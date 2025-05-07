import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_zh_score
)

dataset2metric = {
    "multifieldqa_zh": qa_f1_zh_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    predictions, answers, lengths = [], [], []
    dataset = 'multifieldqa_zh'
    with open('Output/result/path', 'r', encoding='utf-8') as file:  
        qa_data = json.load(file)
    for qa in qa_data:
        predictions.append(qa['llm_ans'])
        answers.append(qa["answers"])
        all_classes = None
        if "length" in qa:
            lengths.append(qa["length"])
    if args.e:
        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
    else:
        score = scorer(dataset, predictions, answers, all_classes)
    scores[dataset] = score
    if args.e:
        out_path = f"judge_experiment/coordination_qa/eval/pred_e_{dataset}.json"
    else:
        out_path = f"judge_experiment/tmp_eval.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
        
# CUDA_VISIBLE_DEVICES=5 python eval.py
