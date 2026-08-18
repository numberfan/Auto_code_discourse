#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/18 09:54
# @Description  :

import json


def error_analysis(gold_data, pred_data):
    gold_map = {item["turn_id"]: set(item["codes"]) for item in gold_data}
    pred_map = {item["turn_id"]: set(item["codes"]) for item in pred_data}

    confusions = []
    for turn_id in gold_map:
        gold = gold_map[turn_id]
        pred = pred_map.get(turn_id, set())
        if gold != pred:
            confusions.append({
                "turn_id": turn_id,
                "gold": list(gold),
                "pred": list(pred),
                "false_positives": list(pred - gold),
                "false_negatives": list(gold - pred)
            })

    # 统计混淆对
    confusion_pairs = {}
    for c in confusions:
        for fn in c["false_negatives"]:
            for fp in c["false_positives"]:
                pair = f"{fn} → {fp}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    print("=== 最常见的混淆模式 ===")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1]):
        print(f"  Gold [{pair.split('→')[0].strip()}] 被误判为 [{pair.split('→')[1].strip()}]: {count} 次")

    return confusions


# 使用
