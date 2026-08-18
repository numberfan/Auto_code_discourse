#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 10:59
# @Description  : 自动编码：使用 Vertex AI / DeepSeek 等，滑动窗口对目标进行编码，多轮投票后确定编码结果

import time
from collections import Counter
from auto_analysis import config
from auto_analysis.utils import load_text_file, extract_json

# 初始化llm
client = config.get_client()
MODEL_NAME = config.get_model_name()
USE_JSON_MODE = config.supports_json_mode()

SYSTEM_PROMPT_PATH = "prompts/coded_prompt.txt"
FEW_SHOT_EXAMPLES_PATH = "prompts/coded_examples.txt"

SYSTEM_PROMPT = load_text_file(SYSTEM_PROMPT_PATH)
FEW_SHOT_EXAMPLES = load_text_file(FEW_SHOT_EXAMPLES_PATH)


def build_context_window(transcript: list, target_idx: int,
                         before: int = 5, after: int = 2) -> str:
    """构建目标话轮的上下文窗口"""

    context_lines = []

    # 前文
    start = max(0, target_idx - before)
    for i in range(start, target_idx):
        turn = transcript[i]
        context_lines.append(
            f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}"
        )

    # 目标话轮
    target = transcript[target_idx]
    context_lines.append(
        f"\n>>> [Turn {target['turn_id']}] {target['speaker']}: "
        f"{target['utterance']} <<<\n"
    )

    # 后文
    end = min(len(transcript), target_idx + after + 1)
    for i in range(target_idx + 1, end):
        turn = transcript[i]
        context_lines.append(
            f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}"
        )

    return "\n".join(context_lines)

MAX_RETRIES= 3

def code_single_turn(transcript: list, target_idx: int,
                     temperature: float = 0.0) -> dict:
    """对单个教师话轮进行 APT 编码（带重试）"""
    context = build_context_window(transcript, target_idx)

    user_prompt = f"""## CONTEXT:
                    {context}

                    ## TASK:
                    Analyze the TARGET TURN (marked with >>>) above. 
                    Identify any APT moves present.
                    Remember: most teacher utterances contain NO APT moves.

                    ## OUTPUT FORMAT:
                    Return ONLY a valid JSON object (no text outside JSON):
                    {{"turn_id": <number>, "codes": ["<code1>", "<code2>", ...], "reasoning": "..."}}
                    If no APT moves are present, use an empty list: []
                    """

    # 构建消息
    system_content = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt}
    ]

    # 构建请求参数
    kwargs = {
        "model": config.get_model_name(),
        "messages": messages,
        "temperature": temperature,
    }
    if config.supports_json_mode():
        kwargs["response_format"] = {"type": "json_object"}

    # 重试循环
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"  [尝试 {attempt + 1}/{MAX_RETRIES}] 调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt  # 1秒, 2秒, 4秒
                print(f"  等待 {sleep_time} 秒后重试...")
                time.sleep(sleep_time)
            else:
                # 最后一次失败则抛出异常，终止
                raise RuntimeError(f"连续 {MAX_RETRIES} 次调用 LLM 失败，放弃该话轮。")


def code_with_voting(transcript: list, target_idx: int,
                     n_votes: int = 5) -> dict:
    """多次推理投票"""

    all_results = []

    for i in range(n_votes):
        # 第一次用 temperature=0，其余用 0.3-0.5
        temp = 0.0 if i == 0 else 0.4
        result = code_single_turn(transcript, target_idx, temperature=temp)
        all_results.append(result)
        time.sleep(0.5)  # rate limiting

    # 投票逻辑
    code_counts = Counter()
    reasoning_list = []

    for r in all_results:
        codes = r.get("codes", [])
        if not codes:
            code_counts["__NONE__"] += 1
        for code in codes:
            code_counts[code] += 1
        reasoning_list.append(r.get("reasoning", ""))

    # 决策
    n = n_votes
    final_codes = []
    confidence_map = {}
    none_votes = code_counts.pop("__NONE__", 0)

    # 找出投票数最高的 code
    if code_counts:
        max_count = max(code_counts.values())
        max_codes = [c for c, cnt in code_counts.items() if cnt == max_count]
    else:
        max_codes = []
    # 如果达到阈值，正常输出
    for code, count in code_counts.items():
        agreement = count / n
        if agreement >= 0.6:
            final_codes.append(code)
            confidence_map[code] = round(agreement, 2)
    # 若没有达到阈值的 code，但模型给出了非空预测（可能分歧大）
    if not final_codes and max_codes and none_votes < n * 0.6:
        # 取票数最高的 code 作为暂定输出，标记需复核
        final_codes = max_codes[:1]
        confidence_map[final_codes[0]] = round(max_count / n, 2)
        needs_review = True

    # 判断是否需要人工复核
    needs_review = False
    if final_codes and any(v < 0.8 for v in confidence_map.values()):
        needs_review = True
    if not final_codes and none_votes < n * 0.8:
        needs_review = True  # 模型不确定是否该编码

    target = transcript[target_idx]

    return {
        "turn_id": target["turn_id"],
        "speaker": target["speaker"],
        "utterance": target["utterance"][:100] + "..." if len(target["utterance"]) > 100 else target["utterance"],
        "codes": final_codes,
        "confidence": confidence_map,
        "needs_review": needs_review,
        "vote_detail": dict(code_counts),
        "none_votes": none_votes,
        "sample_reasoning": reasoning_list[0]  # 第一次（temperature=0）的推理
    }


def code_full_transcript(transcript: list, n_votes: int = 5) -> list:
    """对整个转录文件的所有教师话轮进行编码"""

    results = []
    teacher_turns = [
        (i, t) for i, t in enumerate(transcript)
        if t.get("is_teacher", False)
    ]

    print(f"Total teacher turns to code: {len(teacher_turns)}")

    for idx, (array_idx, turn) in enumerate(teacher_turns):
        print(f"  Coding turn {turn['turn_id']} ({idx + 1}/{len(teacher_turns)})...")
        result = code_with_voting(transcript, array_idx, n_votes=n_votes)
        results.append(result)
        time.sleep(1)  # rate limiting

    return results
