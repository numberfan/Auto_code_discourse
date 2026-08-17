#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 11:18
# @Description  : excel数据清洗，并保存为json

from collections import Counter
import pandas as pd
import json
import re

# APT code 列名到标准化名称的映射
CODE_COLUMN_MAP = {
    "Say more": "say_more",
    "Revoice": "revoice",
    "Press for reasoning": "press_for_reasoning",
    "Challenge": "challenge",
    "Restate": "restate",
    "Agree/Disagree": "agree_disagree",
    "Add on": "add_on",
    "Explain with others": "explain_others",
}


def clean_excel_to_json(filepath: str, file_id: str) -> dict:
    """将 Excel 文件转换为标准化 JSON"""

    df = pd.read_excel(filepath)

    # 标准化列名
    df.columns = [str(c).strip() for c in df.columns]

    transcript = []
    gold_labels = []

    for _, row in df.iterrows():
        # 跳过汇总行
        if pd.isna(row.get("Number")) or str(row.get("Number")).lower() == "sum":
            continue

        turn_id = int(row["Number"]) if not pd.isna(row.get("Number")) else None
        if turn_id is None:
            continue

        speaker = str(row.get("Speaker", "")).strip().rstrip(":")
        utterance = str(row.get("Utterance", "")).strip()
        timestamp = str(row.get("Timestamp", row.get("Time", ""))).strip()

        # 判断是否是教师
        is_teacher = any(kw in speaker.lower() for kw in ["teacher", "t:"])

        # 提取 APT codes
        codes = []
        for col_name, code_name in CODE_COLUMN_MAP.items():
            # 模糊匹配列名（因为不同文件列名可能有细微差异）
            matching_cols = [c for c in df.columns if code_name.replace("_", " ") in c.lower()
                             or col_name.lower() in c.lower()]
            for mc in matching_cols:
                val = row.get(mc)
                if pd.notna(val) and str(val).strip() == "1":
                    codes.append(code_name)
                    break

        turn = {
            "turn_id": turn_id,
            "timestamp": timestamp,
            "speaker": speaker,
            "utterance": utterance,
            "is_teacher": is_teacher,
            "word_count": len(utterance.split()) if utterance else 0,
        }
        transcript.append(turn)

        if is_teacher:
            gold_labels.append({
                "turn_id": turn_id,
                "codes": codes
            })

    return {
        "file_id": file_id,
        "language": "en",
        "total_turns": len(transcript),
        "teacher_turns": sum(1 for t in transcript if t["is_teacher"]),
        "transcript": transcript,
        "gold": gold_labels,
        "statistics": {
            "coded_turns": sum(1 for g in gold_labels if g["codes"]),
            "uncoded_turns": sum(1 for g in gold_labels if not g["codes"]),
            "code_distribution": dict(Counter(
                code for g in gold_labels for code in g["codes"]
            ))
        }
    }