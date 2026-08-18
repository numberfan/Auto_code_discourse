#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 11:16
# @Description  : 留一法交叉验证
import numpy as np

from auto_analysis.assess_func import evaluate_predictions, print_evaluation_report
from auto_analysis.auto_coding import code_full_transcript


def leave_one_out_evaluation(all_files: list, n_votes: int = 3):
    """
    留一法：每次用 1 个文件做测试，其余做 few-shot 来源

    all_files: [{"file_id": "...", "transcript": [...], "gold": [...]}, ...]
    """

    all_results = []

    for i, test_file in enumerate(all_files):
        print(f"\n{'=' * 40}")
        print(f"Testing on: {test_file['file_id']} ({i + 1}/{len(all_files)})")
        print(f"{'=' * 40}")

        # 用该文件的教师话轮做测试
        transcript = test_file["transcript"]
        gold = test_file["gold"]

        # 编码
        predictions = code_full_transcript(transcript, n_votes=n_votes)

        # 评估
        eval_result = evaluate_predictions(gold, predictions)
        eval_result["file_id"] = test_file["file_id"]

        all_results.append(eval_result)
        print_evaluation_report(eval_result)

    # 汇总所有文件的结果
    avg_macro_f1 = np.mean([r["macro_f1"] for r in all_results])
    avg_binary_f1 = np.mean([r["binary_classification"]["f1"] for r in all_results])

    print(f"\n{'=' * 60}")
    print(f"OVERALL LEAVE-ONE-OUT RESULTS")
    print(f"{'=' * 60}")
    print(f"  Average Binary F1: {avg_binary_f1:.3f}")
    print(f"  Average Macro F1:  {avg_macro_f1:.3f}")

    # 找出表现最差的文件
    worst = min(all_results, key=lambda x: x["macro_f1"])
    best = max(all_results, key=lambda x: x["macro_f1"])
    print(f"  Best file:  {best['file_id']} (Macro F1: {best['macro_f1']:.3f})")
    print(f"  Worst file: {worst['file_id']} (Macro F1: {worst['macro_f1']:.3f})")

    return all_results
