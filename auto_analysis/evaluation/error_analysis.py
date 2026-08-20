#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/18 09:54
# @Description  : 错误分析与结果保存

import os
import json
import time
TYPE_A_CODES = {"say_more", "press_for_reasoning", "revoice", "challenge"}
TYPE_B_CODES = {"add_on", "explain_others", "agree_disagree", "restate"}
def error_analysis(
    gold_data,
    pred_data,
    output_dir="report",
    file_id="analysis",
    timestamp=None,
    prompt_version=None,
):
    """执行错误分析并保存详细结果"""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    gold_map = {item["turn_id"]: set(item["codes"]) for item in gold_data}
    pred_map = {item["turn_id"]: item for item in pred_data}

    confusions = []
    addressee_errors = 0
    trigger_errors = 0

    for turn_id in gold_map:
        gold_codes = gold_map[turn_id]
        pred_item = pred_map.get(turn_id, {})
        pred_codes = set(pred_item.get("codes", []))
        if gold_codes != pred_codes:
            # 判断是否是 addressee 类型错误
            gold_is_type_a = bool(gold_codes & TYPE_A_CODES)
            gold_is_type_b = bool(gold_codes & TYPE_B_CODES)
            pred_is_type_a = bool(pred_codes & TYPE_A_CODES)
            pred_is_type_b = bool(pred_codes & TYPE_B_CODES)
            type_mismatch = (gold_is_type_a and pred_is_type_b) or (gold_is_type_b and pred_is_type_a)
            if type_mismatch:
                addressee_errors += 1
            # 判断是否是 trigger 错误（gold 有码但 pred 为空，或反之）
            is_trigger_error = (bool(gold_codes) != bool(pred_codes))
            if is_trigger_error:
                trigger_errors += 1
            confusions.append({
                "turn_id": turn_id,
                "gold": sorted(gold_codes),
                "pred": sorted(pred_codes),
                "false_positives": sorted(pred_codes - gold_codes),
                "false_negatives": sorted(gold_codes - pred_codes),
                "utterance": pred_item.get("utterance_full") or pred_item.get("utterance", ""),
                "pred_addressee": pred_item.get("step2_addressee", ""),
                "pred_trigger": pred_item.get("step1_trigger", ""),
                "reasoning": pred_item.get("reasoning", ""),
                "stage1_evidence": pred_item.get("stage1_evidence", ""),
                "stage2_reasoning": pred_item.get("stage2_reasoning", ""),
                "confirmed": pred_item.get("confirmed", False),
                "type_mismatch": type_mismatch,
                "trigger_error": is_trigger_error,
            })

    # === 输出摘要 ===
    total_errors = len(confusions)
    print(f"\n{'=' * 60}")
    print(f"错误分析报告")
    print(f"{'=' * 60}")
    print(f"总错误话轮: {total_errors}")
    print(f"Trigger 错误 (漏检/误检): {trigger_errors}")
    print(f"Addressee 类型错误 (A↔B): {addressee_errors}")
    print(f"同类型内代码错误: {total_errors - trigger_errors - addressee_errors}")
    # 具体案例
    print(f"\n--- 具体错误案例（前10个）---")
    for c in confusions[:10]:
        tag = ""
        if c["trigger_error"]:
            tag = "[TRIGGER]"
        elif c["type_mismatch"]:
            tag = "[ADDRESSEE]"
        print(f"Turn {c['turn_id']}{tag}: \"{c['utterance']}\"")
        print(f"Gold: {c['gold']}  |  Pred: {c['pred']}")
        print(f"Pred addressee: {c['pred_addressee']} | Stage1: {c['reasoning'][:80]}")
        if c["stage2_reasoning"]:
            print(f"Stage2: {c['stage2_reasoning'][:80]}")
        print()
    # 混淆模式
    confusion_pairs = {}
    for c in confusions:
        for fn in c["false_negatives"]:
            for fp in c["false_positives"]:
                pair = f"{fn} → {fp}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        # 纯漏检
        if not c["false_positives"] and c["false_negatives"]:
            for fn in c["false_negatives"]:
                pair = f"{fn} → (missed)"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        # 纯误报
        if not c["false_negatives"] and c["false_positives"]:
            for fp in c["false_positives"]:
                pair = f"(none) → {fp}"
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    print("--- 最常见的混淆模式 ---")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])
    for pair, count in sorted_pairs[:15]:
        print(f"  {pair}: {count} 次")
    # 纯漏检统计
    pure_misses = [c for c in confusions if not c["false_positives"] and c["false_negatives"]]
    print(f"\n--- 纯漏检 (pred=[] 但 gold 有码): {len(pure_misses)} 个 ---")
    for c in pure_misses[:5]:
        print(f"  Turn {c['turn_id']}: \"{c['utterance']}\"  Gold: {c['gold']}")
    # 保存
    os.makedirs(output_dir, exist_ok=True)
    error_output_path = os.path.join(output_dir, f"error_analysis_{file_id}_{timestamp}.json")
    analysis_report = {
        "file_id": file_id,
        "timestamp": timestamp,
        "prompt_version": prompt_version,
        "summary": {
            "total_errors": total_errors,
            "trigger_errors": trigger_errors,
            "addressee_errors": addressee_errors,
            "within_type_errors": total_errors - trigger_errors - addressee_errors,
            "pure_misses": len(pure_misses),
        },
        "confusion_pairs": dict(sorted_pairs),
        "detailed_confusions": confusions,
    }
    with open(error_output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    print(f"\n错误分析详情已保存至: {error_output_path}")
    return confusions
