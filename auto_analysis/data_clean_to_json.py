#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 11:18
# @Description  : excel数据清洗，并保存为json

from collections import Counter
import pandas as pd
import json
import re

# 标准编码及其可能的基础名称（列名清理后应匹配其中之一）
CODE_BASE_NAMES = {
    "say_more": ["say more"],
    "revoice": ["revoice"],
    "press_for_reasoning": ["press for reasoning"],
    "challenge": ["challenge"],
    "restate": ["restate"],
    "agree_disagree": ["agree disagree", "agree/disagree"],
    "add_on": ["add on"],
    "explain_others": ["explain others", "explain with others"],
}


def clean_col_name(s: str) -> str:
    """
    清理列名，用于匹配：
    - 去除括号及其内容
    - 转小写
    - 非字母字符替换为空格
    - 合并多个空格
    """
    s = str(s).strip()
    # 去括号内容
    s = re.sub(r'\([^)]*\)', '', s)
    s = s.lower()
    # 除字母和空格外全部替换为空格
    s = re.sub(r'[^a-z\s]', ' ', s)
    # 合并多个空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_code_to_col_mapping(df_columns) -> dict:
    """
    根据列名模糊匹配，建立标准编码 -> 实际列名 的映射。
    """
    code_to_col = {}
    for code, base_names in CODE_BASE_NAMES.items():
        # 先清理所有基础名称
        cleaned_bases = [clean_col_name(b) for b in base_names]
        for col in df_columns:
            cleaned_col = clean_col_name(col)
            if cleaned_col in cleaned_bases:
                code_to_col[code] = col
                break
    return code_to_col


def clean_excel_to_json(filepath: str, file_id: str) -> dict:
    """将 Excel 文件转换为标准化 JSON"""

    df_raw = pd.read_excel(filepath, header=None)
    # 前两行是标题行
    first_row = df_raw.iloc[0]
    second_row = df_raw.iloc[1]
    column_names = []
    for i in range(df_raw.shape[1]):
        val_second = second_row.iloc[i]
        if pd.notna(val_second) and str(val_second).strip() != '':
            column_names.append(str(val_second).strip())
        else:
            val_first = first_row.iloc[i]
            if pd.notna(val_first) and str(val_first).strip() != '':
                column_names.append(str(val_first).strip())
            else:
                column_names.append(f"Unnamed_{i}")

    df_raw.columns = column_names
    df = df_raw.iloc[2:].reset_index(drop=True)

    # 标准化列名
    df.columns = [str(c).strip() for c in df.columns]

    # 建立 code 到实际列的映射
    code_to_col = build_code_to_col_mapping(df.columns)

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
        for code, col in code_to_col.items():
            val = row.get(col)
            if pd.notna(val):
                val_str = str(val).strip()
                # 兼容数字 1 / 1.0 和文本 "1"
                if val_str in ("1", "1.0", "1.00"):
                    codes.append(code)

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