import json
# test
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, LlavaForConditionalGeneration, AutoProcessor, VideoLlavaForConditionalGeneration
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os
import numpy as np
import PIL
import torch
import re
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from videoqa_dataset import VideoDialogueDataset
from functools import partial
from torchvision.transforms import ToTensor
from torch.profiler import profile, record_function, ProfilerActivity

hf_token = os.environ["HF_KEY"]

def collator_idx_remover(collator_fn, batch):
    data_dict = collator_fn(batch)
    del data_dict["idx"]
    return data_dict


def eval(args):
    model_id = ("trained_models/"
                "Video-LLaVA-7B-hf_knowledge/checkpoint-1017")
                # "Mistral-7B-Instruct-v0.3_captions/checkpoint-7119")
                # "Meta-Llama-3-8B-Instruct_captions_knowledge/checkpoint-7119")
    llama2_chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    llama3_chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    mistral_chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
    phi_chat_template = "{{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"
    video_llava_chat_template = ("{% for message in messages %}{% if loop.first %}USER: <video>{{ message['content'] }}{% elif message['role'] == 'user' %}USER: {{ message['content'] }}{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}{% endif %}{% if not loop.last %}{{ ' ' }}{% endif %}{% endfor %}{% if messages[-1]['role'] == 'user' %} ASSISTANT:{% endif %}{{ eos_token }}")


    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if not "Phi" in model_id else "right"
    tokenizer.chat_template = llama2_chat_template if "Llama-2" in model_id else (llama3_chat_template if "LLama-3" in model_id else (mistral_chat_template if "Mistral" in model_id else (phi_chat_template if "Phi" in model_id else video_llava_chat_template)))

    if "Llama-3" in model_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(model_id)
    print(tokenizer.chat_template)

    train_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final_train.json",
                                         "datasets/queryd/videos", tokenizer, ToTensor(), "mmplug2_captions.json" if "captions" in model_id else None,
                                         None, "dialogue_only", "peicewise", False,
                                         "retrieved_text_for_kat_n_40.json" if "knowledge" in model_id else None
                                         if "knowledge" in model_id else None)
    print(f"Dataset size: {len(train_dataset)}")
    eval_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final_val.json",
                                        "datasets/queryd/videos", tokenizer, ToTensor(), "mmplug2_captions.json" if "captions" in model_id else None,
                                        None, "dialogue_only", "peicewise", False,
                                        "retrieved_text_for_kat_n_40.json" if "knowledge" in model_id else None
                                        if "knowledge" in model_id else None)

    bsz = 2

    data_collator = partial(collator_idx_remover, partial(VideoDialogueDataset.dialogue_only_collator, pad_direction=tokenizer.padding_side))
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False, num_replicas=8, rank=int(args.device_num))

    response_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=bsz, sampler=sampler, num_workers=4, collate_fn=data_collator)

    # ckpt_dirs = os.listdir(output_dir)
    # ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    # last_ckpt = ckpt_dirs[-1]

    device_num = int(args.device_num)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token).cuda(0) if "Video-LLaVA" not in model_id else VideoLlavaForConditionalGeneration.from_pretrained(model_id, token=hf_token, trust_remote_code=True).cuda(0)
    model.eval()
    model_answers = []

    with torch.no_grad():
        with tqdm(total=len(response_dataloader)) as pbar:
            for batch in response_dataloader:
                # batch = data_collator([eval_dataset[j] for j in range(i, i+bsz) if j < len(eval_dataset)])
                batch = {key: value.cuda(0) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                # import ipdb; ipdb.set_trace()
                batch["max_length"] = 2048
                results = model.generate(**batch)
                for j in range(len(results)):
                    model_answers.append(tokenizer.decode(results[j]))
                del results
                pbar.update(1)
    json.dump([eval_dataset.dialogues, model_answers], open(f"{model_id}/model_predictions_device_{args.device_num}.json", "w"))

# have to change data loading so it doesn't give the answer to the model



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device_num", default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval(args)