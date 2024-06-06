from Evaluation.bart_score import BARTScorer
from tqdm import tqdm
from prometheus_eval_results import load_model_output_file
import pickle as pkl
import numpy as np
from bleurt import score

def main(args):
    bart_scorer = BARTScorer(device=f'cuda:{args.device_num}', checkpoint='facebook/bart-large-cnn')
    bart_scorer.load(path='bart.pth')
    scorer = score.BleurtScorer("BLEURT-20")

    for model_file in [
                       "phi_no_knowledge", "phi_caption", "phi_caption_knowledge",
                       "mistral_no_knowledge", "mistral_caption", "mistral_caption_knowledge",
                       "llama2_no_knowledge", "llama2_caption", "llama2_caption_knowledge",
                       "llama3_no_knowledge", "llama3_caption", "llama3_caption_knowledge",
                       "trained_models/Llama-2-7b-hf/checkpoint-7119/model_predictions.json",
                       "trained_models/Llama-2-7b-hf_captions/checkpoint-7119/model_predictions.json",
                       "trained_models/Meta-Llama-3-8B-Instruct/checkpoint-7119/model_predictions.json",
                       "trained_models/Meta-Llama-3-8B-Instruct_captions/checkpoint-7119/model_predictions.json",
                       "trained_models/Meta-Llama-3-8B-Instruct_captions_knowledge/checkpoint-7119/model_predictions.json",
                       "trained_models/Mistral-7B-Instruct-v0.3/checkpoint-7119/model_predictions.json",
                       "trained_models/Mistral-7B-Instruct-v0.3_captions/checkpoint-7119/model_predictions.json",
                       "trained_models/Mistral-7B-Instruct-v0.3_captions_knowledge/checkpoint-7119/model_predictions.json",
                       "trained_models/Llama-2-7b-hf_captions_knowledge/checkpoint-7119/model_predictions.json",
                       ]:
        print(model_file)
        bart_score_a = []
        bart_score_b = []
        bleurt_scores = []
        bsz = 128
        response_dataloader, model_file = load_model_output_file(model_file, bsz, False)

        with tqdm(total=len(response_dataloader)) as pbar:
            for batch in response_dataloader:
                bart_score_a.extend(bart_scorer.score(batch[0], batch[1]))
                bart_score_b.extend(bart_scorer.score(batch[1], batch[0]))
                # bart_scores.extend(bart_scorer.score(batch[0], batch[1]))
                # bleurt_scores.extend(scorer.score(references=batch[0], candidates=batch[1]))
                pbar.update(1)
        pkl.dump([np.array(bart_score_a), np.array(bart_score_b)], open(f"eval_results/{model_file}_bartscore_eval_results_device.pkl", "wb"))
        # pkl.dump(np.array(bleurt_scores), open(f"eval_results/{model_file}_bleurtscore_eval_results_device.pkl", "wb"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_num', type=int, required=True)
    args = parser.parse_args()
    main(args)