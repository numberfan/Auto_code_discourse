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


def build_context_window(transcript: list, target_idx: int, before: int = 5, after: int = 0) -> str:
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

def code_single_turn(transcript: list, target_idx: int, temperature: float = 0.0) -> dict:
    """对单个教师话轮进行 APT 编码（带重试）"""
    context = build_context_window(transcript, target_idx)
    target_turn = transcript[target_idx]

    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TASK:\n"
        f"Analyze the TARGET TURN (marked with >>>) above.\n"
        f"Pay special attention to the student utterance immediately BEFORE the target — "
        f"it helps determine if the teacher is addressing the SAME student or OTHER students.\n"
        f"Identify any APT moves present.\n"
        f"Remember: most teacher utterances (~60%) contain NO APT moves.\n\n"
        f"## OUTPUT FORMAT:\n"
        f"Return ONLY a valid JSON object (no text outside JSON):\n"
        f'{{"turn_id": {target_turn["turn_id"]}, "codes": [], "reasoning": "..."}}\n'
        f"If no APT moves are present, use an empty list for codes."
    )

    # 构建消息
    system_content = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt}
    ]

    # 构建请求参数
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    if USE_JSON_MODE:
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


def code_with_voting(transcript: list, target_idx: int, n_votes: int = 5) -> dict:
    all_results = []
    for i in range(n_votes):
        temp = 0.0 if i == 0 else 0.4
        result = code_single_turn(transcript, target_idx, temperature=temp)
        all_results.append(result)
        time.sleep(0.5)

    code_counts = Counter()
    reasoning_list = []
    for r in all_results:
        codes = r.get("codes", [])
        if not codes:
            code_counts["__NONE__"] += 1
        for code in codes:
            code_counts[code] += 1
        reasoning_list.append(r.get("reasoning", ""))

    n = n_votes
    none_votes = code_counts.get("__NONE__", 0)
    non_none_counts = {c: cnt for c, cnt in code_counts.items() if c != "__NONE__"}
    max_code, max_count = max(non_none_counts.items(), key=lambda x: x[1], default=(None, 0))

    final_codes = []
    confidence_map = {}
    needs_review = False

    # 严格多数 (> n/2)
    majority_threshold = n / 2
    if max_count > majority_threshold:
        for code, cnt in non_none_counts.items():
            if cnt == max_count:
                final_codes.append(code)
                confidence_map[code] = round(cnt / n, 2)
    # 未达多数，但非空票 > 空票且至少有 2 票，取最高票并标记复核
    elif max_code is not None and max_count >= 2 and max_count > none_votes:
        final_codes.append(max_code)
        confidence_map[max_code] = round(max_count / n, 2)
        needs_review = True

    # 如果最终有代码，但最低置信度 < 0.6，需要复核
    if final_codes and min(confidence_map.values()) < 0.6:
        needs_review = True
    # 如果有非空票但没入选，需要复核
    elif not final_codes and max_code is not None:
        needs_review = True

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
        "sample_reasoning": reasoning_list[0] if reasoning_list else ""
    }


def code_full_transcript(transcript: list, n_votes: int = 5) -> list:
    """对整个转录文件的所有教师话轮进行编码"""

    results = []
    teacher_turns = [
        (i, t) for i, t in enumerate(transcript)
        if t.get("is_teacher", False)
    ]

    print(f"Total teacher turns to code: {len(teacher_turns)}")
    failed_turns = []

    for idx, (array_idx, turn) in enumerate(teacher_turns):
        print(f"  Coding turn {turn['turn_id']} ({idx + 1}/{len(teacher_turns)})...")
        try:
            result = code_with_voting(transcript, array_idx, n_votes=n_votes)
            results.append(result)
        except Exception as e:
            # 记录失败，填入默认空结果
            print(f"  [错误] Turn {turn['turn_id']} 编码失败: {e}")
            failed_turns.append(turn['turn_id'])
            results.append({
                "turn_id": turn["turn_id"],
                "speaker": turn.get("speaker", ""),
                "utterance": turn.get("utterance", "")[:100],
                "codes": [],
                "confidence": {},
                "needs_review": True,  # 失败需要人工复核
                "vote_detail": {},
                "none_votes": 0,
                "sample_reasoning": f"ERROR: {str(e)}"
            })
        time.sleep(1)  # rate limiting

    if failed_turns:
        print(f"\n警告：以下话轮编码失败，已标记为 needs_review=True: {failed_turns}")

    return results
