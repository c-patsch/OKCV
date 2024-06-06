import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import pickle as pkl
from tqdm import tqdm


def setup(rank, world_size, master_addr, master_port):
    print(f"Setting up rank: {rank}")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank} is setup")


def cleanup():
    dist.destroy_process_group()


def model_setup(rank, model_path, world_size):
    video_clip = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
    # config = RetrieverConfig(
    #     indexing_dimension=512,
    #     apply_question_mask=True,
    #     apply_passage_mask=True,
    #     extract_cls=False,
    #     projection=True,
    # )
    # model = Retriever(video_clip.text_model, video_clip.text_model, config, model_id="video_clip_representation")  # pooled_representation
    model = video_clip
    print("Load model")
    if model_path:
        state_dict = torch.load(model_path)
        if list(state_dict.keys())[0] not in model.state_dict().keys():
            for key in list(state_dict.keys()):
                state_dict[key.split("module.")[1]] = state_dict.pop(key)
        model.load_state_dict(state_dict)
    print("Model loaded")
    # model = DDP(model.to(rank), device_ids=[rank])
    model.to(rank)
    model.eval()
    return model, model


def reindex(rank, machine_rank, world_size, master_addr, master_port, model_setup, tokenizer_class, tokenizer_parameters, model_path=None, overall_rank=None, filename='wikidata_ontology.pkl', file_size=21015324):
    if world_size > 0:
        setup(overall_rank if overall_rank else machine_rank+rank, world_size, master_addr, master_port)
        print(rank, world_size, machine_rank+rank, filename)

    model, embedding_fn = model_setup(rank, model_path, world_size)
    tokenizer = tokenizer_class.from_pretrained(*tokenizer_parameters)
    device = "cuda:"+str(rank)
    if world_size == 0:
        world_size += 1

    def read_pickle_lazy(filename, tokenizer, max_tokens, rank, world_size):
        print(filename)
        file = pkl.load(open(filename, 'rb'))
        batch = []
        max_len = 0
        for i, line in enumerate(file.items()):
            if i % world_size != rank:
                continue
            try:
                title, text, id = line[1][0], line[1][1], line[0]
                text = text.split(".")[0]
            except:
                print(i, line)
                import sys
                sys.exit()
            max_len = max(max_len, len(tokenizer(text, truncation=True, max_length=77)))
            if max_len * len(batch) >= max_tokens:
                yield batch
                batch = []
                max_len = len(tokenizer(text, truncation=True, max_length=77))
            batch.append([title, text, id])
        if batch:
            yield batch

    max_tokens = 25000


    data = []

    with torch.no_grad():
        with tqdm(total=file_size//world_size) as pbar:
            for i, batch in enumerate(read_pickle_lazy(filename, tokenizer, max_tokens, overall_rank if overall_rank else machine_rank+rank, world_size)):
                inputs = tokenizer(["entity: " + title + " description: " + text[1:] for title, text, _ in batch], return_tensors="pt", truncation=True, padding=True, max_length=77)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                # inputs["input_ids"] = inputs.pop("input_ids")[:, :512].to(device)
                # inputs.update({"apply_mask": model.module.config.apply_question_mask, "extract_cls": model.module.config.extract_cls})
                text_features = embedding_fn(**inputs).text_embeds.detach().cpu().numpy()
                [data.append([id, title + ": " + text, text_features[i]]) for i, (title, text, id) in enumerate(batch)]
                pbar.update(len(batch))

    os.makedirs("wikidata_embeddings", exist_ok=True)
    pkl.dump(data, open(f"wikidata_embeddings/wikidata_{overall_rank if overall_rank else machine_rank+rank}.pkl", "wb"))



def run_index(world_size, master_addr, master_port, machine_index, model_setup, model_path=None, embeddings_name=None, filename='wikidata_ontology.pkl', file_size=187308):
    world_size = world_size # number of machines
    nprocs = torch.cuda.device_count()
    print(nprocs)
    # mp.spawn(reindex,
    #          args=(nprocs*machine_index, world_size*nprocs, master_addr, master_port, model_setup, CLIPTokenizer, ["Searchium-ai/clip4clip-webvid150k"], model_path, None, filename, file_size),
    #          nprocs=nprocs,
    #          join=True)
    #
    reindex(0, nprocs*machine_index, world_size*nprocs, master_addr, master_port, model_setup, CLIPTokenizer, ["Searchium-ai/clip4clip-webvid150k"], model_path, None, filename, file_size)

    if machine_index == 0:
        data = []
        for rank in range(world_size * nprocs):
            with open(f"wikipedia_embeddings/wikipedia_embeddings_dpr_{rank}.pkl", "rb") as f:
                data.extend(pkl.load(f))

        with open(embeddings_name if embeddings_name else "wikipedia_embeddings/wikipedia_embeddings_dpr.pkl", "wb") as f:
            pkl.dump(data, f)

        with open("embeddings".join(embeddings_name.split("embeddings")[:2]) + "embeddings_only" + "embeddings".join(embeddings_name.split("embeddings")[2:]) if embeddings_name else "wikipedia_embeddings/wikipedia_embeddings_only_dpr.pkl", "wb") as f:
            data_embeddings = [(i[2], i[1]) for i in data]
            pkl.dump(data_embeddings, f)

        with open("embeddings".join(embeddings_name.split("embeddings")[:2]) + "wiki_only" + "embeddings".join(embeddings_name.split("embeddings")[2:]) if embeddings_name else "wikipedia_embeddings/wikipedia_embeddings_only_dpr.pkl", "wb") as f:
            data_wikis = [(i[2], i[0]) for i in data]
            pkl.dump(data_wikis, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--master_addr', type=str, required=True)
    parser.add_argument('--master_port', type=int, required=True)
    parser.add_argument('--machine_index', type=int, required=True)
    args = parser.parse_args()
    run_index(args.world_size, args.master_addr, args.master_port, args.machine_index, model_setup)
