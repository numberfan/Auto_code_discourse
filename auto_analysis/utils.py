#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 14:06
# @Description  :
import json


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
    # 首先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 使用正则找到第一个 { 和最后一个 } 之间的内容
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"无法从模型输出中解析 JSON: {text[:200]}...")