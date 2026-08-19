#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 14:06
# @Description  :
import json
import re

def pct(count: int) -> float:
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
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从模型输出中解析 JSON: {text[:200]}...")