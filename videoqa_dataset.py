import pickle as pkl
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import json
from torchvision.transforms import ToTensor
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class VideoDialogueDataset(Dataset):
    def __init__(self, qa_file_path, video_dir, tokenizer, transforms, caption_context_file, video_embedding_dir=None,
                 data_mode="video_only", dialogue_mode="peicewise", train=False, knowledge_file=None, sample_amount=None,
                 image_processor=None):
        super().__init__()
        self.qa_file_path = qa_file_path
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.data_mode = data_mode
        self.dialogue_mode = dialogue_mode
        self.video_embedding_dir = video_embedding_dir

        dataset = json.load(open(qa_file_path, "r"))
        if "stats" in data_mode:
            self.entire_dataset = dataset
        self.extracted_data = [(key, j["conversation"]) for key, value in dataset.items() for j in value]
        if self.dialogue_mode == "peicewise":
            self.extracted_data = [(data[0], data[1][:i]) for data in self.extracted_data for i in range(2, len(data[1])+1, 2)]
        self.dialogues = [i[1] for i in self.extracted_data]

        self.video_paths = [f"{video_dir if self.video_embedding_dir is None else self.video_embedding_dir}/{i[0]}" for i in self.extracted_data]
        if "video_only" in self.data_mode:
            self.video_paths = np.unique(self.video_paths).tolist()
        self.results = ["" for i in range(len((self.video_paths if "video" in data_mode else self.dialogues)))]

        if caption_context_file is not None:
            caption_context_file = json.load(open(caption_context_file, "r"))
            self.captions = {i["video_id"]: i["pred_caption"] for i in caption_context_file}
        else:
            self.captions = None

        if knowledge_file is not None:
            knowledge_file = json.load(open(knowledge_file, "r"))
            self.knowledge = {i[0].split("/")[-1]: "; ".join(i[1]) for i in zip(knowledge_file[0], knowledge_file[1])}
        else:
            self.knowledge = None

        self.train = train
        self.sample_amount = sample_amount
        self.image_processor = image_processor


    def __getitem__(self, idx):
        video_stuff = {}
        dialogue_stuff = {}
        if "video" in self.data_mode:
            if self.video_embedding_dir is None:
                video_stuff = self.get_video(idx)
            else:
                video_stuff = self.get_video_embedding(idx)
        if "dialogue" in self.data_mode:
            dialogue_stuff = self.get_dialogue(idx)
        return {**video_stuff, **dialogue_stuff, "idx": idx}

    def __len__(self):
        return len(self.video_paths)

    def __setitem__(self, idx, result):
        self.results[idx] = result

    def save(self, path):
        json.dump([self.video_paths, self.results], open(path, "w")) if "video" in self.data_mode else \
            json.dump([self.video_paths, self.dialogues, self.results], open(path, "w"))


    def get_video_embedding(self, idx):
        embedding = pkl.load(open(self.video_paths[idx], "rb"))
        return {
            "video_path": embedding[0][0],
            "video_frames": embedding[1].item(),
            "video_embedding": embedding[2]
        }

    def get_video(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frames.append(self.transforms(Image.fromarray(frame[:, :, [2, 1, 0]]).convert("RGB")))
                # frames.append(self.transforms(224, Image.fromarray(frame[:, :, [2, 1, 0]]).convert("RGB")))
            else:
                break
        cap.release()
        if not frames:
            frames = [torch.zeros((3, 224, 224), dtype=torch.float32) for _ in range(6)]
            fps = -1
        if self.sample_amount is None:
            sample_rate = math.ceil(fps) // 2
            return {
                "video_path": video_path,
                "video_frames": torch.stack(frames)[(sample_rate // 2) ::sample_rate] if fps != -1 else torch.stack(frames),
                "fps": fps
            }
        else:
            sample_frames = np.arange(0, len(frames), len(frames) / self.sample_amount).astype(int)
            return {
                "video_path": video_path,
                "video_frames": self.image_processor(torch.stack(frames)[sample_frames], return_tensors="pt")["pixel_values_images"],
                "fps": fps
            }



    def get_dialogue(self, idx):
        dialogue = self.dialogues[idx]

        if self.train:
            chat = [{"role": "user" if i % 2 == 0 else "assistant", "content": dialog} for i, dialog in enumerate(dialogue)]
        else:
            chat = [{"role": "user" if i % 2 == 0 else "assistant", "content": dialog} for i, dialog in enumerate(dialogue)][:-1]

        chat_preamble = f"The {'question' if len(chat)==1 else 'dialogue'} presented below is based off of a video. {'Please give the next turn of the dialogue.' if len(chat) > 1 else 'Please answer the presented question. '}"
        if self.captions is not None:
            video_idx = self.video_paths[idx].split("/")[-1]
            if video_idx in self.captions:
                caption = self.captions[video_idx]
                if "what does the video describe? " in caption.lower():
                    caption = caption.lower().split("what does the video describe? ")[1].capitalize()
                chat_preamble += f"To give you some context on the video, a caption of what is happening in the video is presented: {caption}\n"
        if self.knowledge is not None:
            video_idx = self.video_paths[idx].split("/")[-1]
            chat_preamble += f"Additionally, to give further context on the video, 40 explicit knowledge entities from wikidata was retrieved based off of what was in the video. This should give context for what was in the video and give you greater explicit knowledge of the subject matter.\nKnowledge: {self.knowledge[video_idx]}"
        chat[0]["content"] = f"{chat_preamble}\nDialogue Start:{chat[0]['content']}"
        chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt", add_special_tokens=False)
        inputs["labels"] = inputs["input_ids"]
        return inputs

    @staticmethod
    def video_only_collator(batch):
        return {
            "video_path": [i["video_path"] for i in batch],
            "video_frames": torch.stack([i["video_frames"] for i in batch]),
            "fps": [i["fps"] for i in batch],
        }

    @staticmethod
    def dialogue_only_collator(batch, padding_value=0, pad_direction="left"):
        max_length_inputs = max([i["input_ids"].shape[1] for i in batch])
        max_length_labels = max([i["input_ids"].shape[1] for i in batch])
        if pad_direction == "left":
            input_ids = torch.vstack([torch.nn.functional.pad(i["input_ids"], pad=(max_length_inputs - i["input_ids"].shape[1], 0)) for i in batch])
            attention_mask = torch.vstack([torch.nn.functional.pad(i["attention_mask"], pad=(max_length_inputs - i["attention_mask"].shape[1], 0)) for i in batch])
            labels = torch.vstack([torch.nn.functional.pad(i["labels"], pad=(max_length_labels - i["attention_mask"].shape[1], 0)) for i in batch])
        elif pad_direction == "right":
            input_ids = pad_sequence([i["input_ids"].permute(1, 0) for i in batch], batch_first=True, padding_value=padding_value).squeeze(2)
            attention_mask = pad_sequence([i["attention_mask"].permute(1, 0) for i in batch], batch_first=True, padding_value=padding_value).squeeze(2)
            labels = pad_sequence([i["labels"].permute(1, 0) for i in batch], batch_first=True, padding_value=padding_value).squeeze(2)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "idx": [i["idx"] for i in batch]
        }

    @staticmethod
    def video_dialogue_collator(batch, padding_value=0, pad_direction="left"):
        video_batch = VideoDialogueDataset.video_only_collator(batch)
        dialogue_batch = VideoDialogueDataset.dialogue_only_collator(batch, padding_value, pad_direction)
        return {
            "pixel_values_videos": video_batch["video_frames"],
            "input_ids": dialogue_batch["input_ids"],
            "attention_mask": dialogue_batch["attention_mask"],
            "labels": dialogue_batch["labels"],
            "idx": dialogue_batch["idx"],
        }


if __name__ == "__main__":
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained("llama/Llama-2-7b-hf")
    processor = transformers.AutoImageProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
    video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset.json", "datasets/queryd/videos", tokenizer, ToTensor(), None, None, "video_dialogue", "peicewise", image_processor=processor, sample_amount=8)
    VideoDialogueDataset.video_dialogue_collator([video_dataset[1], video_dataset[2], video_dataset[3], video_dataset[4]])