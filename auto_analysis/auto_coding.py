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
llm_config.print_status()  # 打印模型状态

PROMPT_VERSION = "v3"
SYSTEM_PROMPT_PATH = f"prompts/{PROMPT_VERSION}/coded_prompt.txt"
FEW_SHOT_EXAMPLES_PATH = f"prompts/{PROMPT_VERSION}/few_shot.txt"
CONFIRM_PROMPT_PATH = f"prompts/{PROMPT_VERSION}/confirm_prompt.txt"

SYSTEM_PROMPT = load_text_file(SYSTEM_PROMPT_PATH)
FEW_SHOT_EXAMPLES = load_text_file(FEW_SHOT_EXAMPLES_PATH)
CONFIRM_PROMPT_TEMPLATE = load_text_file(CONFIRM_PROMPT_PATH)

TYPE_A_CODES = {"say_more", "press_for_reasoning", "revoice", "challenge"}
TYPE_B_CODES = {"add_on", "explain_others", "agree_disagree", "restate"}


def build_context_window(transcript: list, target_idx: int, before: int = 5, after: int = 2) -> str:
    """构建目标话轮的上下文窗口"""

    context_lines = []
    start = max(0, target_idx - before)  # 前文

    # 找到目标话轮前最后一个学生
    student_before = None
    for i in range(target_idx - 1, -1, -1):
        if not transcript[i].get("is_teacher", False):
            student_before = transcript[i]["speaker"]
            break
    # 找到目标话轮后第一个学生
    student_after = None
    for i in range(target_idx + 1, min(len(transcript), target_idx + after + 1)):
        if not transcript[i].get("is_teacher", False):
            student_after = transcript[i]["speaker"]
            break

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

    # 添加 speaker 标注
    context_lines.append("")
    context_lines.append(
        f"[SPEAKER INFO] Student_Before = {student_before or 'unknown'} | Student_After = {student_after or 'unknown'}")
    if student_before and student_after:
        if student_before == student_after:
            context_lines.append(f"[SPEAKER INFO] Same student continues → likely Type A")
        else:
            context_lines.append(f"[SPEAKER INFO] Different student responds → likely Type B")

    return "\n".join(context_lines)


MAX_RETRIES = 3


async def code_single_turn(client, transcript: list, target_idx: int, temperature: float = 0.0) -> dict:
    """对单个教师话轮进行 APT 编码（异步）"""
    context = build_context_window(transcript, target_idx)
    target_turn = transcript[target_idx]

    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TASK:\n"
        f"Analyze the TARGET TURN (marked with >>>) above.\n"
        f"Follow the 3-step CODING PROCEDURE defined in the system prompt.\n\n"
        f"IMPORTANT for Step 2 (Addressee Check):\n"
        f"- Look at [SPEAKER INFO] at the bottom of the context.\n"
        f"- If Student_After ≠ Student_Before, the teacher is redirecting → Type B.\n"
        f"- If Student_After = Student_Before, the teacher stays with same student → Type A.\n"
        f"- Use the CONTENT of responses only to verify, not to determine the code.\n\n"
        f"## OUTPUT FORMAT:\n"
        f"Return ONLY a valid JSON object (no text outside JSON):\n"
        f'{{"turn_id": {target_turn["turn_id"]}, "step1_trigger": "yes"|"no", '
        f'"step2_addressee": "same_student"|"other_student"|"none", '
        f'"step2_evidence": "...", '
        f'"codes": [], "reasoning": "..."}}\n'
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
            print(
                f"[调用中] 正在向 {MODEL_NAME} 发送请求 (Turn {target_turn['turn_id']}, 尝试 {attempt + 1}/{MAX_RETRIES})...")
            response = await client.chat.completions.create(**kwargs)
            print(f"[成功] 收到响应 (Turn {target_turn['turn_id']})")
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"[尝试 {attempt + 1}/{MAX_RETRIES}] 调用失败: {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt  # 1秒, 2秒, 4秒
                print(f"等待 {sleep_time} 秒后重试...")
                await asyncio.sleep(sleep_time)
            else:
                # 最后一次失败则抛出异常
                raise RuntimeError(f"连续 {MAX_RETRIES} 次调用 LLM 失败，放弃该话轮。")


def is_confident(result):
    trigger = result.get("step1_trigger")
    codes = result.get("codes", [])

    if trigger == "no":
        return len(codes) == 0
    if not codes:  # 如果 trigger 是 yes 但没代码，说明模型不确定
        return False

    # 检查 addressee 与 codes 类型一致
    addressee = result.get("step2_addressee")
    if addressee not in ("same_student", "other_student"):
        return False

    codes_set = set(codes)
    if addressee == "same_student" and not codes.issubset(TYPE_A_CODES):
        return False
    if addressee == "other_student" and not codes.issubset(TYPE_B_CODES):
        return False

    # 检查 reasoning 中是否表达不确定
    reasoning = result.get("reasoning", "").lower()
    uncertainty_words = ["probably", "maybe", "uncertain", "possibly", "either", "not sure", "could be"]
    if any(w in reasoning for w in uncertainty_words):
        return False
    return True


async def code_with_voting(client, transcript: list, target_idx: int, n_votes: int = 2) -> dict:
    """异步并发执行多轮投票"""

    # 先单次，temperature=0
    first_result = await code_single_turn(client, transcript, target_idx, temperature=0.0)

    # 快速路径：单次结果足够自信
    if is_confident(first_result):
        target = transcript[target_idx]
        return {
            "turn_id": target["turn_id"],
            "speaker": target["speaker"],
            "utterance": target["utterance"][:100],
            "codes": first_result.get("codes", []),
            "confidence": {c: 1.0 for c in first_result.get("codes", [])} if first_result.get("codes") else {
                "no_apt": 1.0},
            "needs_review": False,
            "vote_detail": {"__NONE__": 0} if not first_result.get("codes") else {c: 1 for c in
                                                                                  first_result.get("codes", [])},
            "none_votes": 0,
            "step1_trigger": first_result.get("step1_trigger", "yes"),
            "step2_addressee": first_result.get("step2_addressee", "none"),
            "sample_reasoning": first_result.get("reasoning", "")
        }

    # 否则执行多轮投票（n_votes 可为 3）
    async def single_vote(i):
        temp = 0.0 if i == 0 else 0.3
        return await code_single_turn(client, transcript, target_idx, temperature=temp)

    extra_votes = max(1, n_votes - 1)  # 至少再投1次
    vote_tasks = [single_vote(i) for i in range(extra_votes)]
    all_results = await asyncio.gather(*vote_tasks, return_exceptions=True)
    valid_results = [r for r in all_results if not isinstance(r, Exception)]

    needs_review = False

    if not valid_results:
        raise RuntimeError("所有投票任务均失败")
    n = len(valid_results)

    # 如果模型没输出 step1_trigger / step2_addressee，从 codes 推断
    for r in valid_results:
        if "step1_trigger" not in r:
            r["step1_trigger"] = "yes" if r.get("codes") else "no"
        if "step2_addressee" not in r:
            codes = set(r.get("codes", []))
            if codes & TYPE_A_CODES:
                r["step2_addressee"] = "same_student"
            elif codes & TYPE_B_CODES:
                r["step2_addressee"] = "other_student"
            else:
                r["step2_addressee"] = "none"

    # Step 1: Trigger 投票
    trigger_yes = sum(1 for r in valid_results if r.get("step1_trigger") == "yes")
    trigger_no = n - trigger_yes

    if trigger_yes > n / 2:
        trigger_decision = "yes"
    elif trigger_yes > 0 and trigger_yes == trigger_no:
        trigger_decision = "uncertain"
    elif trigger_yes >= 2 and trigger_no > trigger_yes:
        trigger_decision = "uncertain"
    else:
        trigger_decision = "no"

    if trigger_decision == "no":
        # 多数认为无 APT
        target = transcript[target_idx]
        return {
            "turn_id": target["turn_id"],
            "speaker": target["speaker"],
            "utterance": target["utterance"][:100],
            "codes": [],
            "confidence": {"no_apt": round((n - trigger_yes) / n, 2)},
            "needs_review": False,
            "vote_detail": {"__NONE__": n - trigger_yes},
            "none_votes": n - trigger_yes,
            "step1_trigger": "no",
            "step2_addressee": "none",
            "sample_reasoning": valid_results[0].get("reasoning", "")
        }
    elif trigger_decision == "uncertain":
        needs_review = True

    # Step 2: Addressee 投票
    addressee_votes = [r.get("step2_addressee", "") for r in valid_results]
    addressee_votes_valid = [a for a in addressee_votes if a in ("same_student", "other_student")]  # 过滤无效值
    if not addressee_votes_valid:
        addressee_decision = "other_student"
    else:
        vote_counts = Counter(addressee_votes_valid)
        max_count = max(vote_counts.values())
        top_addressees = [a for a, cnt in vote_counts.items() if cnt == max_count]
        if len(top_addressees) > 1:
            needs_review = True
            addressee_decision = "other_student"
        else:
            addressee_decision = top_addressees[0]

    # 根据 addressee 定义允许的 codes
    if addressee_decision == "same_student":
        allowed_codes = TYPE_A_CODES
    else:
        allowed_codes = TYPE_B_CODES

    # Step 3: 对每个投票的 codes 进行过滤
    filtered_code_lists = []
    for r in valid_results:
        raw_codes = r.get("codes", [])
        filtered = [c for c in raw_codes if c in allowed_codes]
        filtered_code_lists.append(filtered)

    # 聚合过滤后的 codes
    code_counts = Counter()
    for codes in filtered_code_lists:
        if not codes:
            code_counts["__NONE__"] += 1
        for code in codes:
            code_counts[code] += 1
    none_votes = code_counts.get("__NONE__", 0)
    non_none_counts = {c: cnt for c, cnt in code_counts.items() if c != "__NONE__"}
    final_codes = []
    confidence_map = {}

    # 选择出现次数最多的 codes，要求至少 2 票且超过 none_votes
    if non_none_counts:
        max_count = max(non_none_counts.values())
        if max_count > n / 2:  # 严格多数
            for code, cnt in non_none_counts.items():
                if cnt > n / 2:
                    final_codes.append(code)
                    confidence_map[code] = round(cnt / n, 2)
        else:  # 相对多数下，输出所有票数 >=2 且 > none_votes 的 code
            for code, cnt in non_none_counts.items():
                if cnt >= 2 and cnt > none_votes:
                    final_codes.append(code)
                    confidence_map[code] = round(cnt / n, 2)
            if final_codes:
                needs_review = True
    if not final_codes and non_none_counts:
        needs_review = True
    # 检查投票分歧是否过大（例如 confidence < 0.6）
    if final_codes and min(confidence_map.values()) < 0.6:
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
        "step1_trigger": trigger_decision,
        "step2_addressee": addressee_decision,
        "sample_reasoning": valid_results[0].get("reasoning", "")
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
            print(f"[开始] 话轮 Turn {turn['turn_id']} ({idx + 1}/{len(teacher_turns)})...")
            try:
                result = await code_with_voting(client, transcript, array_idx, n_votes=n_votes)

                # 二次确认：仅对 needs_review=True 的话轮
                if result.get("needs_review", False):
                    print(f"[二次确认] Turn {turn['turn_id']} 进入确认阶段...")
                    confirm_result = await confirm_uncertain_turn(
                        client, transcript, array_idx, result
                    )
                    if confirm_result and "codes" in confirm_result:
                        if result['step2_addressee'] == "same_student":
                            allowed = TYPE_A_CODES
                        elif result['step2_addressee'] == "other_student":
                            allowed = TYPE_B_CODES
                        else:
                            allowed = set()

                        filtered = [c for c in confirm_result['codes'] if c in allowed]
                        if filtered:
                            result["codes"] = filtered
                        else:
                            if not confirm_result["codes"]:
                                result["codes"] = []

                        result["confirmed"] = True
                        result["confirm_reasoning"] = confirm_result.get("reasoning", "")
                        result["needs_review"] = False  # 已确认
                        print(f"[确认完成] Turn {turn['turn_id']} → {confirm_result['codes']}")

                print(f"[完成] 话轮 Turn {turn['turn_id']} 编码成功")
                return result
            except Exception as e:
                print(f"[错误] Turn {turn['turn_id']} 编码失败: {e}")
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
    """对整个转录文件的所有教师话轮进行编码（同步入口）"""
    return asyncio.run(_code_full_transcript_async(transcript, n_votes=n_votes, max_concurrency=5))


async def confirm_uncertain_turn(client, transcript: list, target_idx: int, first_pass_result: dict) -> dict:
    """对 needs_review=True 的话轮进行二次确认编码"""

    context = build_context_window(transcript, target_idx, before=7, after=2)  # 给更多上下文
    target_turn = transcript[target_idx]

    # 构建确认 prompt，注入第一轮结果
    confirm_context = (
        f"## FIRST-PASS ANALYSIS:\n"
        f"- Vote detail: {first_pass_result.get('vote_detail', {})}\n"
        f"- Addressee decision: {first_pass_result.get('step2_addressee', 'unknown')}\n"
        f"- Candidate codes: {first_pass_result.get('codes', [])}\n"
        f"- Sample reasoning: {first_pass_result.get('sample_reasoning', '')}\n\n"
        f"## TRANSCRIPT CONTEXT:\n{context}\n\n"
        f"## TARGET: Turn {target_turn['turn_id']}\n\n"
        f"Make your FINAL decision. Output ONLY valid JSON:\n"
        f'{{"turn_id": {target_turn["turn_id"]}, "codes": [], "reasoning": "..."}}'
    )

    messages = [
        {"role": "system", "content": CONFIRM_PROMPT_TEMPLATE},
        {"role": "user", "content": confirm_context}
    ]

    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,  # 确认阶段用 0 温度，要确定性
    }
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
        result = extract_json(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"[二次确认失败] Turn {target_turn['turn_id']}: {e}")
        return None
