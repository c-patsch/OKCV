from transformers import CLIPVisionModelWithProjection, AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import pickle as pkl
# import faiss
import os
import cv2
import torch
from PIL import Image
import math
import tqdm
from videoqa_dataset import VideoDialogueDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import os
from functools import partial

hf_token = os.environ["HF_KEY"]

def setup(rank, world_size, master_addr, master_port):
    print(f"Setting up rank: {rank}")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} is setup")


def cleanup():
    dist.destroy_process_group()


def load_wikipedia_embedding():
    wiki_embeddings = pkl.load(open("wikipedia_embeddings/wikipedia_embeddings_dpr.pkl", "rb"))
    d = wiki_embeddings[0][2].shape[0]
    index = faiss.IndexFlatIP(d)
    [index.add(embed[2].reshape(1, -1)) for embed in wiki_embeddings]
    index_text = np.array([i[1] for i in wiki_embeddings])
    return index, index_text

def load_wikidata_embedding():
    wiki_embeddings = pkl.load(open("wikidata_embeddings/wikidata_0.pkl", "rb"))
    d = wiki_embeddings[0][2].shape[0]
    index = faiss.IndexFlatIP(d)
    [index.add(embed[2].reshape(1, -1)) for embed in wiki_embeddings]
    index_text = np.array([i[1] for i in wiki_embeddings])
    return index, index_text

def query_wikipedia_loop(index, index_text, n=40, prefetched=True):
    video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset.json", "datasets/queryd/videos", None, preprocess, None, "clip4clip_embedding" if prefetched else None, "video_only")
    # sampler = torch.utils.data.distributed.DistributedSampler(video_dataset, shuffle=False, drop_last=False)
    video_dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)
    if not prefetched:
        model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
        model = DDP(model.to(0), device_ids=[0])
        model = model.eval()
    with tqdm.tqdm(total=len(video_dataloader)) as pbar:
        for batch in video_dataloader:
            if not prefetched:
                video_embedding = real_time_video_embeddings(0, batch["video_frames"], model)
            else:
                video_embedding = batch["video_embedding"]
            distances, retrieved_indices = index.search(video_embedding, n)
            retrieved_text = index_text[retrieved_indices]
            for retrieved_idx, batch_idx in enumerate(batch["idx"]):
                video_dataloader.dataset[batch_idx] = retrieved_text[retrieved_idx].tolist()
            pbar.update(1)
    video_dataloader.dataset.save(f"retrieved_text_for_kat_n_{n}.json")


def preprocess(n_px, sizes=224):
    return Compose([
        Resize(sizes, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(sizes),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])(n_px)


def pre_fetch_video_embeddings(rank, machine_rank, world_size, master_addr, master_port):
    if world_size > 0:
        setup(machine_rank+rank, world_size, master_addr, master_port)
        print(rank, world_size, machine_rank+rank)

    video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset.json", "datasets/queryd/videos", None, preprocess, None, None, "video_only")
    sampler = torch.utils.data.distributed.DistributedSampler(video_dataset, shuffle=False, drop_last=False)
    video_dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=4)

    video_embeddings = []
    model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    model = DDP(model.to(rank), device_ids=[rank])
    model = model.eval()
    with tqdm.tqdm(total=len(video_dataloader)) as pbar:
        for video_batch in video_dataloader:
            visual_output = real_time_video_embeddings(rank, video_batch["video_frames"], model)
            video_embeddings.append((video_batch["video_path"], video_batch["fps"], visual_output.cpu()))
            pbar.update(1)
    pkl.dump(video_embeddings, open(f"video_embeddings_{rank}.pkl", "wb"))

def real_time_video_embeddings(rank, video, model):
    with torch.no_grad():
        visual_output = model(video[0].to(rank))
    visual_output = visual_output["image_embeds"]
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    visual_output = torch.mean(visual_output, dim=0)
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    return visual_output


# def get_video_caption():

# def get_video_gpt4_answer():

def zero_shot_experiments(experiment_name, model_id, video_dataset, device_number):
    print(experiment_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto" if "phi" in model_id.lower() else torch.bfloat16,
        token=hf_token,
        trust_remote_code=True
    ).cuda()
    model.eval()
    # terminators = [
    #     video_dataset.tokenizer.eos_token_id,
    #     video_dataset.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]
    collator_fn = partial(VideoDialogueDataset.dialogue_only_collator, padding_value=video_dataset.tokenizer.pad_token_id if video_dataset.tokenizer.pad_token_id is not None else 0, pad_direction=video_dataset.tokenizer.padding_side)
    sampler = torch.utils.data.distributed.DistributedSampler(video_dataset, shuffle=False, drop_last=False, num_replicas=8, rank=device_number)
    video_dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=10, sampler=sampler, num_workers=4, collate_fn=collator_fn)
    with tqdm.tqdm(total=len(video_dataloader)) as pbar:
        with torch.no_grad():
            for batch in video_dataloader:
                results = model.generate(batch["input_ids"].squeeze(0).cuda(), attention_mask=batch["attention_mask"].squeeze(0).cuda(), max_new_tokens=256)  # , eos_token_id=terminators)
                results = video_dataset.tokenizer.batch_decode(results, skip_special_tokens=True)
                for batch_idx, dataset_idx in enumerate(batch["idx"]):
                    video_dataloader.dataset[dataset_idx] = results[batch_idx]
                pbar.update(1)
    video_dataloader.dataset.save(f"{experiment_name}_{device_number}.json")

def run_index(world_size, master_addr, master_port, machine_index):
    world_size = world_size # number of machines
    nprocs = torch.cuda.device_count()
    print(nprocs)
    mp.spawn(pre_fetch_video_embeddings,
             args=(nprocs*machine_index, nprocs, master_addr, master_port),
             nprocs=nprocs,
             join=True)
    if machine_index == 0:
        all_embeddings = []
        embedding_files = [i for i in os.listdir() if "video_embeddings_" in i]
        for file in embedding_files:
            all_embeddings += pkl.load(open(file, "rb"))
        pkl.dump(all_embeddings, open("video_embeddings.pkl", "wb"))
        for i in range(len(all_embeddings)):
            pkl.dump(all_embeddings[i], open(f"clip4clip_embedding/{all_embeddings[i][0][0].split('/')[-1]}", "wb"))


if __name__ == "__main__":
    # # Run Embedding Getting
    # world_size = 1
    # master_addr = "localhost"
    # master_port = 8080
    # machine_index = 0
    # run_index(world_size, master_addr, master_port, machine_index)

    # # Run retrieval
    # index, index_text = load_wikidata_embedding()
    # query_wikipedia_loop(index, index_text, 15, True)
    # query_wikipedia(None, None)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, required=True)
    args = parser.parse_args()
    #
    # # Run llama
    # # Zero Knowledge Experiments
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, None, None, "dialogue_only", "peicewise")
    # zero_shot_experiments("llama3_no_knowledge", model_id, video_dataset, args.device_num)
    #
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, None, None, "dialogue_only", "peicewise")
    # zero_shot_experiments("llama2_no_knowledge", model_id, video_dataset, args.device_num)
    #
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, None, None, "dialogue_only", "peicewise", train=False)
    # zero_shot_experiments("mistral_no_knowledge", model_id, video_dataset, args.device_num)

    # model_id = "microsoft/Phi-3-small-8k-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left', trust_remote_code=True)
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, None, None, "dialogue_only", "peicewise", train=False)
    # zero_shot_experiments("phi_no_knowledge", model_id, video_dataset, args.device_num)

    #
    # # Caption-Only Experiments
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise")
    # zero_shot_experiments("llama3_caption", model_id, video_dataset, args.device_num)
    #
    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise")
    # zero_shot_experiments("llama2_caption", model_id, video_dataset, args.device_num)

    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", train=False)
    # zero_shot_experiments("mistral_caption", model_id, video_dataset, args.device_num)

    # model_id = "microsoft/Phi-3-small-8k-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left', trust_remote_code=True)
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", train=False)
    # zero_shot_experiments("phi_caption", model_id, video_dataset, args.device_num)
    #

    # # Caption + WD-Knowledge Experiments
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left"
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", False,"retrieved_text_for_kat_n_40.json")
    # zero_shot_experiments("llama3_caption_knowledge", model_id, video_dataset, args.device_num)

    # model_id = "meta-llama/Llama-2-7b-chat-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", False, "retrieved_text_for_kat_n_40.json")
    # zero_shot_experiments("llama2_caption_knowledge", model_id, video_dataset, args.device_num)

    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left')
    # video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", train=False, knowledge_file="retrieved_text_for_kat_n_40.json")
    # zero_shot_experiments("mistral_caption_knowledge", model_id, video_dataset, args.device_num)

    model_id = "microsoft/Phi-3-small-8k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, padding_side='left', trust_remote_code=True)
    video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset_final.json", "datasets/queryd/videos", tokenizer, preprocess, "mmplug2_captions.json", None, "dialogue_only", "peicewise", train=False, knowledge_file="retrieved_text_for_kat_n_40.json")
    zero_shot_experiments("phi_caption_knowledge", model_id, video_dataset, args.device_num)

