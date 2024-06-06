import json
import pickle as pkl
import os
import numpy as np
from collections import Counter

print("BARTScores")
bart_score_results = [i for i in os.listdir("eval_results") if "bartscore" in i]
for i in bart_score_results:
   bart_scores_a, bart_scores_b = pkl.load(open(f"eval_results/{i}", "rb"))
   print(f"{i.split('_bartscore_eval_results_')[0]}: {np.exp(bart_scores_a).mean()}")

print()

print("BleurtScores")
bart_score_results = [i for i in os.listdir("eval_results") if "bleurtscore" in i]
for i in bart_score_results:
   bleurt_scores = pkl.load(open(f"eval_results/{i}", "rb"))
   print(f"{i.split('_bleurtscore_eval_results_')[0]}: {bleurt_scores.mean()}")

print()

print("Prometheus Scores")
prometheus_eval_results = sorted([[key, value] for key, value in Counter([i.split("_prometheus_eval_results_")[0] for i in os.listdir("eval_results") if "prometheus" in i]).items()], key=lambda x: x[0])

for prometheus_res in prometheus_eval_results:
    all_scores = []
    if prometheus_res[1] == 8:
        for i in range(8):
            _, scores = json.load(open(f"eval_results/{prometheus_res[0]}_prometheus_eval_results_device_{i}.json", "r"))
            all_scores.extend(scores)
        print(f"{prometheus_res[0]}: {np.mean([i for i in all_scores if i is not None])}")
    else:
        print(f"{prometheus_res[0]}: In Progress")