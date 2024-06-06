from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import json
import re
from tqdm import tqdm
import torch
import numpy as np


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, model_answers, reference_answers):
        self.model_answers = model_answers
        self.reference_answers = reference_answers

    def __len__(self):
        return len(self.model_answers)

    def __getitem__(self, idx):
        model_answer = self.model_answers[idx]
        reference_answer = self.reference_answers[idx]
        return model_answer, reference_answer


def extract_model_output_file_llama(model_file):
    model_outputs = json.load(open(model_file, "r"))
    needs_feedback = [(model_outputs[1][i], model_outputs[2][i]) for i in range(len(model_outputs[0])) if model_outputs[2][i]] if len(model_outputs) == 3 else \
        [(model_outputs[0][i], model_outputs[1][i]) for i in range(len(model_outputs[0])) if model_outputs[1][i]]
    responses = [i[1] for i in needs_feedback]
    if "llama2" in model_file.lower():
        pre_responses = [re.findall(r"(?s)\[INST\](.+?)\[\/INST\]", i)[-1] for i in responses]
        responses = [pre_responses[idx].strip() + " " + i.split(pre_responses[idx] + "[/INST]")[-1].strip() for idx, i in enumerate(responses)]
    elif "llama3" in model_file.lower():
        split_response = [responses[i].split("assistant") for i in range(len(responses))]
        pre_responses = [split_response[i][-2].split("user")[-1].strip() for i in range(len(split_response))]
        responses = [pre_responses[i] + " " + split_response[i][-1].strip() for i in range(len(split_response))]

    reference_answers = ["\n".join(i[0][-2:]) for i in needs_feedback]
    return responses, reference_answers

def extract_model_output_file_mistral(model_file):
    model_outputs = json.load(open(model_file, "r"))
    needs_feedback = [(model_outputs[1][i], model_outputs[2][i]) for i in range(len(model_outputs[0])) if model_outputs[2][i]] if len(model_outputs) == 3 else \
        [(model_outputs[0][i], model_outputs[1][i]) for i in range(len(model_outputs[0])) if model_outputs[1][i]]

    reference_answers = [i[0][-2:] for i in needs_feedback]
    responses = ["\n".join([reference_answers[i][0], needs_feedback[i][1].split(reference_answers[i][0])[-1].strip()]) for i in range(len(needs_feedback))]
    reference_answers = ["\n".join(i) for i in reference_answers]
    return responses, reference_answers


def load_model_output_file(model_file, bsz, dist=True):
    all_responses = []
    all_reference_answers = []
    if "trained_models" not in model_file:
        for i in range(0, 8):
            responses, reference_answers = extract_model_output_file_llama(f"{model_file}_{i}.json") if "llama" in model_file else extract_model_output_file_mistral(f"{model_file}_{i}.json")
            all_responses.extend(responses)
            all_reference_answers.extend(reference_answers)
    else:
        responses, reference_answers = extract_model_output_file_llama(model_file) if "llama" in model_file else extract_model_output_file_mistral(model_file)
        all_responses.extend(responses)
        all_reference_answers.extend(reference_answers)
        model_file = model_file.split("/")[1]


    response_dataset = CustomDataset(all_responses, all_reference_answers)
    sampler = torch.utils.data.distributed.DistributedSampler(response_dataset, shuffle=False, drop_last=False, num_replicas=8, rank=args.device_num) if dist else None

    response_dataloader = torch.utils.data.DataLoader(response_dataset, batch_size=bsz, sampler=sampler, num_workers=4) if dist else \
        torch.utils.data.DataLoader(response_dataset, batch_size=bsz, shuffle=False, num_workers=4)
    return response_dataloader, model_file


def main(args):
    instruction = "Having watched a video, a user has a few questions which they ask about the video to get more clarity about the video and topics around the video.",

    rubric_data = {
        "criteria": "Is the model proficient in answering the user's question succinctly and clearly?",
        "score1_description": "The model neglects to answer the question, is very indirect, or is overly wordy with its answer.",
        "score2_description": "The model answers the question, but is either incorrect, misses key details, or is very indirect with how it answers the question.",
        "score3_description": "The model answers mostly matches the reference answers but in an indirect or overly long manner.",
        "score4_description": "The model answers matches the reference answer well but is still overly wordy or indirect with its answer.",
        "score5_description": "The model excels in responding to the user's questions and answers correctly, directly, and with appropriate succinctness."
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    bsz = 3
    # model_file = "mistral_caption_knowledge"
    # model_file = "mistral_caption"
    # model_file = "phi_caption"
    model_file = "trained_models/Llama-2-7b-hf_captions_knowledge/checkpoint-7119/model_predictions.json"
    # model_file = "trained_models/Mistral-7B-Instruct-v0.3/checkpoint-7119/model_predictions.json"
    # model_file = "trained_models/Mistral-7B-Instruct-v0.3_captions/checkpoint-7119/model_predictions.json"
    # model_file = "trained_models/Mistral-7B-Instruct-v0.3_captions_knowledge/checkpoint-7119/model_predictions.json"
    response_dataloader, model_file = load_model_output_file(model_file, bsz)

    judge = PrometheusEval(model_id="prometheus-eval/prometheus-7b-v2.0", absolute_grade_template=ABSOLUTE_PROMPT)
    all_feedbacks = []
    all_scores = []
    with tqdm(total=len(response_dataloader)) as pbar:
        for batch in response_dataloader:
            feedbacks, scores = judge.absolute_grade(
                instructions=[instruction for i in range(len(batch[0]))],
                responses=batch[0],
                params={},
                rubric=score_rubric,
                reference_answers=batch[1]
            )
            all_feedbacks.extend(feedbacks)
            all_scores.extend(scores)
            pbar.update(1)
    json.dump([all_feedbacks, all_scores], open(f"eval_results/{model_file}_prometheus_eval_results_device_{args.device_num}.json", "w"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, required=True)
    args = parser.parse_args()
    main(args)