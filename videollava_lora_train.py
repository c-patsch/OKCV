import numpy as np
import logging
import os
from contextlib import nullcontext
import random
import PIL
from peft import LoraConfig
from transformers import pipeline, AutoModelForCausalLM
from videoqa_dataset import VideoDialogueDataset
from torchvision.transforms import ToTensor
from functools import partial
from accelerate import PartialState

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, VideoLlavaForConditionalGeneration, AutoImageProcessor

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()
hf_token = os.environ["HF_KEY"]
# def custom_callback(ex):
#     import ipdb; ipdb.set_trace()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


def collator_idx_remover(collator_fn, batch):
    data_dict = collator_fn(batch)
    del data_dict["idx"]
    return data_dict


if __name__ == "__main__":
    model_id = "LanguageBind/Video-LLaVA-7B-hf"
    knowledge_type = ""
    ################
    # Dataset
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    processor = AutoImageProcessor.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if not "Phi" in model_id else "right"

    tokenizer.chat_template = ("{% for message in messages %}{% if loop.first %}USER: <video>{{ message['content'] }}{% elif message['role'] == 'user' %}USER: {{ message['content'] }}{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}{% endif %}{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}{% if messages[-1]['role'] == 'user' %} ASSISTANT:{% endif %}{{ eos_token }}")
    print(model_id)
    print(tokenizer.chat_template)

    train_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final_train.json",
                                         "datasets/queryd/videos", tokenizer, ToTensor(), "mmplug2_captions.json" if "c" in knowledge_type else None,
                                         None, "video_dialogue", "peicewise",
                                         False, "retrieved_text_for_kat_n_40.json" if
                                         "k" in knowledge_type else None, sample_amount=8, image_processor=processor)
    print(f"Dataset size: {len(train_dataset)}")
    eval_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final_val.json",
                                        "datasets/queryd/videos", tokenizer, ToTensor(), "mmplug2_captions.json" if "c" in knowledge_type else None,
                                        None, "video_dialogue", "peicewise", True,
                                        "retrieved_text_for_kat_n_40.json" if
                                        "k" in knowledge_type else None, sample_amount=8, image_processor=processor)

    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        # trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    print(model_config.attn_implementation)
    # device_string = PartialState().process_index
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, token=hf_token, trust_remote_code=True, **model_kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        # training_args.max_grad_norm = 5e-5
        # print(training_args)
        peft_config = get_peft_config(model_config)
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )
        # peft_config.use_rslora = True  # difference between really high grad norm and then NaN and starting out at infinite grad norm
        # peft_config.init_lora_weights = "loftq"
        training_args.optim = "paged_adamw_32bit"
        training_args.fp16 = True
        training_args.bf16 = False
        training_args.learning_rate = 2e-5
        training_args.warmup_ratio = 0.03
        # training_args.group_by_length=True
        training_args.lr_scheduler_type = "constant"
        training_args.logging_steps = 100
        training_args.max_grad_norm = 1.0
        training_args.save_steps = 1017
        training_args.do_eval = False
        training_args.output_dir = f"trained_models/{model_id.split('/')[-1]}{'_captions' if 'c' in knowledge_type else ''}{'_knowledge' if 'k' in knowledge_type else ''}"
        training_args.push_to_hub = False
        training_args.hub_token = hf_token
        training_args.gradient_checkpointing_kwargs={'use_reentrant': False}
        print(training_args.output_dir)
        # print(training_args.num_train_epochs)
        print(training_args.output_dir)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="question",  # need a dummy field
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=partial(collator_idx_remover, VideoDialogueDataset.video_dialogue_collator),
            dataset_kwargs={"skip_prepare_dataset": True},
        )

    # if sft_script_args.train == "True":

    trainer.train()

    with save_context:
        model_id = model_id.replace("/", "_")
        trainer.save_model(f"trained_models/{model_id.split('/')[-1]}{'_captions' if 'c' in knowledge_type else ''}{'_knowledge' if 'k' in knowledge_type else ''}")
    # else:
    #     print("Here")


# accelerate launch --mixed_precision fp16 videollava_lora_train.py --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft"     --model_name_or_path="llava-hf/llava-1.5-7b-hf"     --report_to="none"     --learning_rate=2e-5     --per_device_train_batch_size=1     --gradient_accumulation_steps=1     --output_dir="data/vsft-llava-1.5-7b-hf"     --num_train_epochs=1     --gradient_checkpointing     --remove_unused_columns=False     --torch_dtype=float16 --fp16=True  --use_peft=True     --lora_r=64     --lora_alpha=16     --lora_target_modules=all-linear --log_level="info" --logging_strategy="steps" --logging_steps=1

# accelerate launch --mixed_precision fp16 videollava_lora_train.py --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft"     --model_name_or_path="llava-hf/llava-1.5-7b-hf"     --report_to="none"     --learning_rate=2e-5     --per_device_train_batch_size=1     --gradient_accumulation_steps=1     --output_dir="data/vsft-llava-1.5-7b-hf"     --num_train_epochs=4     --gradient_checkpointing     --remove_unused_columns=False     --torch_dtype=float16 --fp16=True  --use_peft=True     --lora_r=64     --lora_alpha=16     --lora_target_modules=all-linear --log_level="info" --logging_strategy="steps" --logging_steps=1