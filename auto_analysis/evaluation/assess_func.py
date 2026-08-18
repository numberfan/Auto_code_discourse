#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 11:07
# @Description  : 编码准确性计算、保存结果

import os
import json
import time

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import numpy as np

ALL_CODES = [
    "say_more", "revoice", "press_for_reasoning", "challenge",
    "restate", "agree_disagree", "add_on", "explain_others"
]


def evaluate_predictions(gold_data: list, pred_data: list) -> dict:
    """评估预测结果 """

    gold_map = {item["turn_id"]: set(item["codes"]) for item in gold_data}
    pred_map = {item["turn_id"]: set(item["codes"]) for item in pred_data}

    # === 阶段一：二分类（是否有 APT move）===
    binary_gold = []
    binary_pred = []

    for turn_id in gold_map:
        binary_gold.append(1 if gold_map[turn_id] else 0)
        binary_pred.append(1 if pred_map.get(turn_id, set()) else 0)

    binary_f1 = f1_score(binary_gold, binary_pred, zero_division=0)
    binary_precision = precision_score(binary_gold, binary_pred, zero_division=0)
    binary_recall = recall_score(binary_gold, binary_pred, zero_division=0)

    # === 阶段二：每个 code 的 F1 ===
    per_code_metrics = {}

    for code in ALL_CODES:
        code_gold = [1 if code in gold_map[tid] else 0 for tid in gold_map]
        code_pred = [1 if code in pred_map.get(tid, set()) else 0 for tid in gold_map]

        if sum(code_gold) == 0:
            per_code_metrics[code] = {"f1": None, "support": 0, "note": "No gold samples"}
            continue

        f1 = f1_score(code_gold, code_pred, zero_division=0)
        prec = precision_score(code_gold, code_pred, zero_division=0)
        rec = recall_score(code_gold, code_pred, zero_division=0)

        per_code_metrics[code] = {
            "f1": round(f1, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "support": sum(code_gold)
        }

    # === 汇总 ===
    macro_f1 = np.mean([
        m["f1"] for m in per_code_metrics.values()
        if m["f1"] is not None
    ])

    return {
        "binary_classification": {
            "f1": round(binary_f1, 3),
            "precision": round(binary_precision, 3),
            "recall": round(binary_recall, 3),
        },
        "per_code_metrics": per_code_metrics,
        "macro_f1": round(macro_f1, 3),
        "total_turns_evaluated": len(gold_map),
        "turns_with_apt_moves": sum(1 for v in gold_map.values() if v),
        "turns_without_apt_moves": sum(1 for v in gold_map.values() if not v),
    }


def save_evaluation_report(eval_results: dict, output_dir: str = "report", file_id: str = "eval", timestamp: str = None) -> str:
    """保存评估报告"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    eval_output_path = os.path.join(output_dir, f"evaluation_{file_id}_{timestamp}.json")

    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"评估报告已成功保存至: {eval_output_path}")
    return eval_output_path

def print_evaluation_report(eval_results: dict):
    """打印评估报告"""

    print("=" * 60)
    print("APT CODING EVALUATION REPORT")
    print("=" * 60)

    print(f"\nTotal turns evaluated: {eval_results['total_turns_evaluated']}")
    print(f"  With APT moves: {eval_results['turns_with_apt_moves']}")
    print(f"  Without APT moves: {eval_results['turns_without_apt_moves']}")

    print(f"\n--- Stage 1: Binary Classification (Has APT move?) ---")
    b = eval_results["binary_classification"]
    print(f"  Precision: {b['precision']}")
    print(f"  Recall:    {b['recall']}")
    print(f"  F1:        {b['f1']}")

    print(f"\n--- Stage 2: Per-Code Performance ---")
    print(f"{'Code':<22} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Support'}")
    print("-" * 55)

    for code, m in eval_results["per_code_metrics"].items():
        if m["f1"] is not None:
            print(f"{code:<22} {m['f1']:<8} {m['precision']:<8} {m['recall']:<8} {m['support']}")
        else:
            print(f"{code:<22} {'N/A':<8} {'N/A':<8} {'N/A':<8} {m['support']}")

    print(f"\n--- Overall ---")
    print(f"  Macro F1: {eval_results['macro_f1']}")
    print("=" * 60)