import matplotlib.pyplot as plt
from videoqa_dataset import VideoDialogueDataset
import numpy as np
from torchvision.transforms import ToTensor
import spacy
from collections import Counter, defaultdict
import cv2


nlp = spacy.load("en_core_web_sm")
# Loading the dataset
video_video_dataset = VideoDialogueDataset("overall_conversational_videos_dataset.json", "datasets/queryd/videos", None, ToTensor(), None, None, "stats_video_only", "whole")

knowledge_type, knowledge_count = np.unique([j["knowledge_type"] for i in video_video_dataset.entire_dataset.values() for j in i], return_counts=True)
knowledge_type = knowledge_type.astype('<U20')
knowledge_type[knowledge_type=="visual"] = "Visual"
knowledge_type[knowledge_type=="commonsense"] = "Common Sense"
knowledge_type[knowledge_type=="factoid"] = "Factoid"

# Extracting the first word from questions
# question_words, question_counts = np.unique(first_word_question, return_counts=True)

# Calculating the percentage
total_questions = np.sum(knowledge_count)
question_percentages = 100 * knowledge_count / total_questions

# Sorting the slices from greatest to least
indices = np.argsort(knowledge_count)[::-1]
sorted_question_words = knowledge_type[indices]
sorted_question_counts = knowledge_count[indices]
sorted_question_percentages = question_percentages[indices]

# Aggregating small slices into "Other"
new_labels = []
new_counts = []
other_count = 0
other_percent = 0

for word, count, percent in zip(sorted_question_words, sorted_question_counts, sorted_question_percentages):
    if percent <= 1.5:
        other_count += count
        other_percent += percent
    else:
        new_labels.append(f'{word}\n{percent:.1f}%')
        new_counts.append(count)

if other_count > 0:
    new_labels.append(f'Other\n{other_percent:.1f}%')
    new_counts.append(other_count)

# Creating the donut chart
fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
wedges, texts = ax.pie(new_counts, labels=new_labels, wedgeprops=dict(width=0.5, edgecolor='w'), startangle=-40)

# Adding a circle at the center to turn the pie into a donut
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# fig.gca().add_artist(centre_circle)

# Adjusting label positions using the midpoint of each slice
for text, wedge in zip(texts, wedges):
    angle = ((wedge.theta2 - wedge.theta1) / 2 + wedge.theta1)
    print(text, (wedge.theta2 - wedge.theta1))
    # question angle adjustments
    angle = angle + 15 if wedge.theta2 - wedge.theta1 < 46 else angle
    angle = angle - 22 if wedge.theta2 - wedge.theta1 < 40 else angle
    # answer angle adjustments
    # angle = angle - 4 if wedge.theta2 - wedge.theta1 < 35 else angle
    # angle = angle + 9.5 if wedge.theta2 - wedge.theta1 < 32 else angle
    # angle = angle - 3.5 if wedge.theta2 - wedge.theta1 <= 27 else angle
    # angle = angle - 5.5 if wedge.theta2 - wedge.theta1 <= 11 else angle
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    # Align text position based on the slice's size and center
    text.set_position((x * 1.1, y * 1.1))  # adjust 1.2 as needed to better position the labels
    text.set_fontsize(12)
    # text.set_fontname("impact")
    text.set_weight("bold")


ax.text(0, 0, f"Total\nDialogues:\n{total_questions}", ha='center', va='center', fontsize=16, weight="bold", fontname="impact")

# Displaying the chart
plt.show()
