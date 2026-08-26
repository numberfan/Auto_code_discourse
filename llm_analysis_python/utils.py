#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 14:06
# @Description  :
import json
import re

def pct(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0

def load_text_file(filepath):
    """读取文本文件，若失败则返回空字符串"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"警告：无法读取 {filepath}: {e}")
        return ""

def extract_json(text: str) -> dict:
    """
    从模型输出中提取 JSON 对象。
    当模型返回的文本包含前后缀时，尝试提取第一个完整的 JSON 对象。
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder(strict=False)
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue

    repaired = _repair_truncated_json(text)
    if repaired is not None:
        return repaired
    raise ValueError(f"无法从模型输出中解析 JSON: {text[:200]}...")


def _repair_truncated_json(text: str):
    """Close a JSON object cut off at the end of a model response."""
    start = text.find("{")
    if start < 0:
        return None

    candidate = text[start:]
    stack = []
    in_string = False
    escaped = False
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (char == "]" and stack[-1] != "[") or (char == "}" and stack[-1] != "{"):
                return None
            stack.pop()

    # Complete a truncated scalar so validation can downgrade missing confidence to review.
    stripped = candidate.rstrip()
    if stripped.endswith((":", ",")):
        if stripped.endswith(":"):
            candidate = stripped + " null"
        else:
            candidate = stripped[:-1]
    if in_string:
        candidate += '"'
    candidate += "".join("}" if item == "{" else "]" for item in reversed(stack))
    try:
        value = json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None