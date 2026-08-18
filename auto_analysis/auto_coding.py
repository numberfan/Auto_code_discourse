#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 10:59
# @Description  : 自动编码：使用 Vertex AI / DeepSeek 等，滑动窗口对目标进行编码，多轮投票后确定编码结果
import asyncio
import time
from collections import Counter
from auto_analysis import llm_config
from auto_analysis.utils import load_text_file, extract_json

# 初始化llm
MODEL_NAME = llm_config.get_model_name()
USE_JSON_MODE = llm_config.supports_json_mode()
llm_config.print_status() # 打印模型状态

SYSTEM_PROMPT_PATH = "prompts/coded_prompt.txt"
FEW_SHOT_EXAMPLES_PATH = "prompts/few_show.txt"

SYSTEM_PROMPT = load_text_file(SYSTEM_PROMPT_PATH)
FEW_SHOT_EXAMPLES = load_text_file(FEW_SHOT_EXAMPLES_PATH)


def build_context_window(transcript: list, target_idx: int, before: int = 5, after: int = 0) -> str:
    """构建目标话轮的上下文窗口"""

    context_lines = []
    start = max(0, target_idx - before) # 前文
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

async def code_single_turn(client, transcript: list, target_idx: int, temperature: float = 0.0) -> dict:
    """对单个教师话轮进行 APT 编码（异步）"""
    context = build_context_window(transcript, target_idx)
    target_turn = transcript[target_idx]

    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TASK:\n"
        f"Analyze the TARGET TURN (marked with >>>) above.\n"
        f"Pay special attention to the student utterance immediately BEFORE the target — "
        f"it helps determine if the teacher is addressing the SAME student or OTHER students.\n"
        f"Identify any APT moves present.\n"
        f"Remember: If the teacher is responding to or following up on student talk (even briefly), assign a code. Only leave codes=[] when the utterance is clearly procedural, feedback, or content delivery.\n\n"
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
            print(f"  [调用中] 正在向 {MODEL_NAME} 发送请求 (Turn {target_turn['turn_id']}, 尝试 {attempt + 1}/{MAX_RETRIES})...")
            response = client.chat.completions.create(**kwargs)
            print(f"  [成功] 收到响应 (Turn {target_turn['turn_id']})")
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


async def code_with_voting(client, transcript: list, target_idx: int, n_votes: int = 5) -> dict:
    """异步并发执行多轮投票"""
    async def single_vote(i):
        temp = 0.0 if i == 0 else 0.2
        return await code_single_turn(client, transcript, target_idx, temperature=temp)

    # 并发执行所有投票
    vote_tasks = [single_vote(i) for i in range(n_votes)]
    all_results = await asyncio.gather(*vote_tasks, return_exceptions=True)

    valid_results = []
    for r in all_results:
        if not isinstance(r, Exception):
            valid_results.append(r)

    if not valid_results:
        raise RuntimeError("所有投票任务均失败")

    code_counts = Counter()
    reasoning_list = []
    for r in valid_results:
        codes = r.get("codes", [])
        if not codes:
            code_counts["__NONE__"] += 1
        for code in codes:
            code_counts[code] += 1
        reasoning_list.append(r.get("reasoning", ""))

    n = len(valid_results)
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
    elif max_code is not None and max_count >= 2 and max_count > none_votes:
        final_codes.append(max_code)
        confidence_map[max_code] = round(max_count / n, 2)
        needs_review = True

    if final_codes and min(confidence_map.values()) < 0.6:
        needs_review = True
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


async def _code_full_transcript_async(transcript: list, n_votes: int = 5, max_concurrency: int = 5) -> list:
    """对整个转录文件的所有教师话轮进行编码"""

    client = llm_config.get_async_client()
    semaphore = asyncio.Semaphore(max_concurrency)  # 限制同时并发请求数，避免触发大模型限流

    teacher_turns = [
        (i, t) for i, t in enumerate(transcript)
        if t.get("is_teacher", False)
    ]

    print(f"Total teacher turns to code (Async): {len(teacher_turns)}, Max Concurrency: {max_concurrency}")
    failed_turns = []

    async def process_single_teacher_turn(idx, array_idx, turn):
        async with semaphore:
            print(f"  [开始] 话轮 Turn {turn['turn_id']} ({idx + 1}/{len(teacher_turns)})...")
            try:
                result = await code_with_voting(client, transcript, array_idx, n_votes=n_votes)
                print(f"  [完成] 话轮 Turn {turn['turn_id']} 编码成功")
                return result
            except Exception as e:
                print(f"  [错误] Turn {turn['turn_id']} 编码失败: {e}")
                failed_turns.append(turn['turn_id'])
                return {
                    "turn_id": turn["turn_id"],
                    "speaker": turn.get("speaker", ""),
                    "utterance": turn.get("utterance", "")[:100],
                    "codes": [],
                    "confidence": {},
                    "needs_review": True,
                    "vote_detail": {},
                    "none_votes": 0,
                    "sample_reasoning": f"ERROR: {str(e)}"
                }

    tasks = [
        process_single_teacher_turn(idx, array_idx, turn)
        for idx, (array_idx, turn) in enumerate(teacher_turns)
    ]

    results = await asyncio.gather(*tasks)

    if failed_turns:
        print(f"\n警告：以下话轮编码失败，已标记为 needs_review=True: {failed_turns}")

    return results

def code_full_transcript(transcript: list, n_votes: int = 5) -> list:
    """对整个转录文件的所有教师话轮进行编码（同步入口，供 main.py 直接调用）"""
    return asyncio.run(_code_full_transcript_async(transcript, n_votes=n_votes, max_concurrency=5))