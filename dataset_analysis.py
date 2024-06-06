from videoqa_dataset import VideoDialogueDataset
import torch
import numpy as np
from torchvision.transforms import ToTensor
import spacy
from collections import Counter, defaultdict
import cv2
import matplotlib.pyplot as plt


nlp = spacy.load("en_core_web_sm")

def parse_time_to_seconds(time_str, idx):
    # Split the time string by ':', handle empty minutes or seconds
    parts = time_str.split(':')
    if len(parts) == 1:
        parts = time_str.split('.')
    if len(parts) == 1:
        parts = time_str.split(';')
    try:
        minutes = int(parts[0]) if parts[0] else 0
        seconds = int(parts[1]) if len(parts) > 1 else 0
    except:
        print(idx)
        return
    return minutes * 60 + seconds


video_video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset.json", "datasets/queryd/videos", None, ToTensor(), None, None, "stats_video_only", "whole")
all_dialogue_turns = [j for i in video_video_dataset.dialogues for j in i]
#
#
# all_tokens = [nlp(i.lower()) for i in all_dialogue_turns]
# Counter([token.pos_ for doc in all_tokens for token in doc])
#
#
print(f"Number of Videos: {len(video_video_dataset)}")
print(f"Number of Dialogues: {len(video_video_dataset.dialogues)}")
print(f"Number of dialogue turns: {len(all_dialogue_turns)}")
print(f"Average number of dialogue turns/video: {len(all_dialogue_turns)/3/len(video_video_dataset)}")
# print(f"Number of tokens: {sum([len(i) for i in all_tokens])}")
# print(f"Sentence Length/turn: {sum([len(i) for i in all_tokens])/len(all_dialogue_turns)}")
# print(f"Total Vocabulary: {len(np.unique([j.text for i in all_tokens for j in i]))}")
#
# tokens_with_multiple_pos = defaultdict(set)
# for doc in all_tokens:
#     for token in doc:
#         tokens_with_multiple_pos[token.text].add(token.pos_)
#
# # Create a dictionary to count the number of tokens for each POS
# pos_token_counts = defaultdict(int)
# for pos_set in tokens_with_multiple_pos.values():
#     for pos in pos_set:
#         pos_token_counts[pos] += 1
# print("POS of each unique token: ")
# print(pos_token_counts)
#
#
# sources = [j["sources"] for i in video_video_dataset.entire_dataset.values() for j in i if j["sources"] and j["sources"]!='"n/a"' and j["sources"]!="n/a" and j["sources"]!="none" and "copilot" not in j["sources"].lower()]
# print(f"Number of Dialogues with sources: {len(sources)}")
# print(f"Number of sources found: {sum([len(source.split(chr(10))) for source in sources])}")
# knowledge_type, knowledge_count = np.unique([j["knowledge_type"] for i in video_video_dataset.entire_dataset.values() for j in i], return_counts=True)
# print(knowledge_type, knowledge_count)  # make into pie chart

temporal_certificates = [[k for k in zip(j["begin_time"], j["end_time"]) if k[0] and k[1]] for i in video_video_dataset.entire_dataset.values() for j in i]
print(f"Number of Valid Temporal Certificates Collected: {len([i for i in temporal_certificates if i])}")
temporal_certificates = [i for i in temporal_certificates if i]
converted_times = [[(parse_time_to_seconds(start.strip(), idx), parse_time_to_seconds(end.strip(), idx)) for start, end in pair] for idx, pair in enumerate(temporal_certificates)]
time_diff = [sum([j[1]-j[0] for j in i]) for i in converted_times]  # have to continue cleaning for negative time differences and numbers that are too large
print(f"Average temporal certificate: {np.mean(time_diff)}")  # make into distribution plot
# import IPython; IPython.embed()

video_lengths = []
for i in range(len(video_video_dataset)):
    cap = cv2.VideoCapture(video_video_dataset.video_paths[i])
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_length_seconds = total_frames / fps
    video_lengths.append(total_length_seconds)
print(f"Average video lengths: {np.mean(video_lengths)}")  # make into distribution plot

for i, length_data in enumerate([time_diff, video_lengths]):
    plt.subplot(1, 2, i+1)
    counts, bins = np.histogram(length_data, bins=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 75, 90, 105, 120, 150, 180, max(length_data)]) if i == 0 else np.histogram(length_data, bins=10)
    bins = bins.astype(int)
    plt.bar(range(len(counts)), counts, width=1, edgecolor="blue", color='lightblue')

    # Set custom x-tick labels
    labels = [f'{bins[i]}-{bins[i+1]-1}' if bins[i+1] - bins[i] > 1 else f'{bins[i]}' for i in range(len(bins)-1)]
    labels[-1] = '180+' if i == 0 else f"{bins[-2]}+"  # Modify last label to show 180+
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(0, 900, 100))

    plt.title('Temporal Certificates Range') if i == 0 else plt.title('Video Lengths')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.tight_layout()

# Show the histogram
plt.show()


# have to remove dialogue turns that are supposed to be blank but the turkers filled in ("well questioned" and "thank you"