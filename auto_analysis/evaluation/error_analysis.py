#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/18 09:54
# @Description  : 错误分析与结果保存

import os
import json
import time

def error_analysis(gold_data, pred_data, output_dir="report", file_id="analysis", timestamp=None):
    """执行错误分析并保存详细结果"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    gold_map = {item["turn_id"]: set(item["codes"]) for item in gold_data}
    pred_map = {item["turn_id"]: set(item["codes"]) for item in pred_data}

    # 找到预测数据中的话语内容
    pred_utterance_map = {}
    for item in pred_data:
        pred_utterance_map[item["turn_id"]] = item.get("utterance", "")

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
                "false_negatives": list(gold - pred),
                "utterance": pred_utterance_map.get(turn_id, "")
            })

    # 输出具体案例（前10个）
    print("\n=== 具体错误案例（前10个）===")
    for c in confusions[:10]:
        print(f"Turn {c['turn_id']}: \"{c['utterance']}\"")
        print(f"Gold: {c['gold']}  |  Pred: {c['pred']}")
        print(f"漏检: {c['false_negatives']}  |  误报: {c['false_positives']}")
        print()

    # 统计混淆对
    confusion_pairs = {}
    for c in confusions:
        for fn in c["false_negatives"]:
            for fp in c["false_positives"]:
                pair = f"{fn} → {fp}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    print("=== 最常见的混淆模式 ===")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    for pair, count in sorted_pairs:
        print(f"Gold [{pair.split('→')[0].strip()}] 被误判为 [{pair.split('→')[1].strip()}]: {count} 次")

     # 统计
    pure_misses = [c for c in confusions if not c["false_positives"] and c["false_negatives"]]
    print(f"\n=== 纯漏检（pred为空但gold有）: {len(pure_misses)} 个 ===")
    for c in pure_misses[:5]:
        print(f"Turn {c['turn_id']}: \"{c['utterance']}\"  Gold: {c['gold']}")

    # 保存详细分析结果到文件
    os.makedirs(output_dir, exist_ok=True)
    error_output_path = os.path.join(output_dir, f"error_analysis_{file_id}_{timestamp}.json")

    analysis_report = {
        "file_id": file_id,
        "timestamp": timestamp,
        "total_errors": len(confusions),
        "pure_misses_count": len(pure_misses),
        "confusion_pairs": {pair: count for pair, count in sorted_pairs},
        "detailed_confusions": confusions
    }

    with open(error_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)

    print(f"\n错误分析详情已成功保存至: {error_output_path}")

    return confusions


# 使用
