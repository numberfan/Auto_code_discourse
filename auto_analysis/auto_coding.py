#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 10:59
# @Description  : 单次预测+条件复查
import asyncio
from auto_analysis import llm_config
from auto_analysis.utils import load_text_file, extract_json

# 初始化llm
MODEL_NAME = llm_config.get_model_name()
USE_JSON_MODE = llm_config.supports_json_mode()
llm_config.print_status()  # 打印模型状态

# prompt
PROMPT_VERSION = "v3"
SYSTEM_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/coded_prompt.txt")
FEW_SHOT_EXAMPLES = load_text_file(f"prompts/{PROMPT_VERSION}/few_shot.txt")
CONFIRM_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/confirm_prompt.txt")

# APT codes
TYPE_A_CODES = {"say_more", "press_for_reasoning", "revoice", "challenge"}
TYPE_B_CODES = {"add_on", "explain_others", "agree_disagree", "restate"}
ALL_VALID_CODES = TYPE_A_CODES | TYPE_B_CODES

# llm重连
MAX_RETRIES = 3


# 上下文窗口
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

    # 前文
    for i in range(start, target_idx):
        turn = transcript[i]
        context_lines.append(f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}")

    # 目标话轮
    target = transcript[target_idx]
    context_lines.append(f"\n>>> [Turn {target['turn_id']}] {target['speaker']}: {target['utterance']} <<<\n")

    # 后文
    end = min(len(transcript), target_idx + after + 1)
    for i in range(target_idx + 1, end):
        turn = transcript[i]
        context_lines.append(f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}")

    # speaker 标注
    context_lines.append("")
    context_lines.append(
        f"[SPEAKER INFO] Student_Before = {student_before or 'unknown'} | Student_After = {student_after or 'unknown'}")
    if student_before and student_after:
        if student_before == student_after:
            context_lines.append(f"[SPEAKER INFO] Same student continues → likely Type A")
        else:
            context_lines.append(f"[SPEAKER INFO] Different student responds → likely Type B")

    return "\n".join(context_lines)


# 置信度判断
def is_confident(result: dict) -> bool:
    """判断单次预测结果是否足够确信"""
    trigger = result.get("step1_trigger")
    codes = result.get("codes", [])
    addressee = result.get("step2_addressee", "")
    reasoning = result.get("reasoning", "").lower()

    # Case 1: 无触发且无代码 → 确信
    if trigger == "no" and len(codes) == 0:
        return True

    # Case 2: 有触发但没代码 → 不确信
    if trigger == "yes" and not codes:
        return False

    # Case 3: addressee 缺失或无效 → 不确信
    if addressee not in ("same_student", "other_student"):
        return False

    # Case 4: codes 与 addressee 类型不一致 → 不确信
    codes_set = set(codes)
    if not codes_set.issubset(ALL_VALID_CODES):
        return False

    if addressee == "same_student" and not codes_set.issubset(TYPE_A_CODES):
        return False
    if addressee == "other_student" and not codes_set.issubset(TYPE_B_CODES):
        return False

    # Case 5: 高风险代码进入复查。错误分析显示 add_on/challenge 容易自信误报，
    # 因此即使格式一致，也要求二次确认。
    high_risk_codes = {"add_on", "challenge"}
    if codes_set & high_risk_codes:
        return False

    # Case 6: 推理中有犹豫词 → 不确信
    uncertainty_markers = ["probably", "maybe", "uncertain", "possibly",
                           "either", "not sure", "could be", "unclear"]
    if any(w in reasoning for w in uncertainty_markers):
        return False

    return True


# Layer 1: 单次预测
async def predict_single(client, transcript: list, target_idx: int) -> dict:
    """单次预测：temp=0，返回解析后的 JSON dict"""
    context = build_context_window(transcript, target_idx)
    target_turn = transcript[target_idx]
    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TASK:\n"
        f"Analyze the TARGET TURN (marked with >>>) above.\n"
        f"Follow the 3-step CODING PROCEDURE defined in the system prompt.\n\n"
        f"IMPORTANT for Step 2 (Addressee Check):\n"
        f"- Look at [SPEAKER INFO] at the bottom of the context.\n"
        f"- Student_After ≠ Student_Before is evidence for Type B, but not enough by itself.\n"
        f"- Student_After = Student_Before is evidence for Type A, but not enough by itself.\n"
        f"- First verify the teacher is building on a specific student's idea, not asking a new lesson question.\n"
        f"- Use the CONTENT of responses to verify whether the turn is peer-linked or topic-advancing.\n\n"
        f"## OUTPUT FORMAT:\n"
        f"Return ONLY a valid JSON object (no text outside JSON):\n"
        f'{{"turn_id": {target_turn["turn_id"]}, "step1_trigger": "yes"|"no", '
        f'"step2_addressee": "same_student"|"other_student"|"none", '
        f'"step2_evidence": "...", '
        f'"codes": [], "reasoning": "..."}}\n'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES},
        {"role": "user", "content": user_prompt}
    ]
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  [预测] Turn {target_turn['turn_id']} (尝试 {attempt + 1}/{MAX_RETRIES})...")
            response = await client.chat.completions.create(**kwargs)
            return extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"  [失败] 尝试 {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Turn {target_turn['turn_id']}: 连续 {MAX_RETRIES} 次预测失败")


# Layer 2: 条件复查
async def confirm_prediction(client, transcript: list, target_idx: int, first_result: dict) -> dict:
    """对不确信的预测进行二次确认，返回最终 codes + addressee"""
    context = build_context_window(transcript, target_idx, before=7, after=3)
    target_turn = transcript[target_idx]
    user_prompt = (
        f"## FIRST-PASS RESULT:\n"
        f"- step1_trigger: {first_result.get('step1_trigger', 'unknown')}\n"
        f"- step2_addressee: {first_result.get('step2_addressee', 'unknown')}\n"
        f"- step2_evidence: {first_result.get('step2_evidence', '')}\n"
        f"- codes: {first_result.get('codes', [])}\n"
        f"- reasoning: {first_result.get('reasoning', '')}\n\n"
        f"## TRANSCRIPT CONTEXT:\n{context}\n\n"
        f"## TARGET: Turn {target_turn['turn_id']}\n"
        f'"{target_turn["utterance"]}"\n\n'
        f"Make your FINAL decision. Output ONLY valid JSON:\n"
        f'{{"turn_id": {target_turn["turn_id"]}, '
        f'"addressee": "same_student"|"other_student"|"none", '
        f'"codes": [], "reasoning": "..."}}'
    )
    messages = [
        {"role": "system", "content": CONFIRM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        print(f"  [复查] Turn {target_turn['turn_id']}...")
        response = await client.chat.completions.create(**kwargs)
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"  [复查失败] Turn {target_turn['turn_id']}: {e}")
        return None


# 核心流程：预测+条件复查
async def code_single_teacher_turn(client, transcript: list, target_idx: int) -> dict:
    """对单个教师话轮完成完整编码流程"""
    target = transcript[target_idx]
    # Layer 1: 单次预测
    first_result = await predict_single(client, transcript, target_idx)
    # 判断置信度
    if is_confident(first_result):
        return _build_output(target, first_result, confirmed=False, needs_review=False)
    # Layer 2: 条件复查
    print(f"  [不确信] Turn {target['turn_id']}，进入复查...")
    confirm_result = await confirm_prediction(client, transcript, target_idx, first_result)
    if confirm_result and "codes" in confirm_result:
        # 用 confirm 返回的 addressee 做类型过滤
        addressee = confirm_result.get("addressee", "none")
        if addressee == "same_student":
            allowed = TYPE_A_CODES
        elif addressee == "other_student":
            allowed = TYPE_B_CODES
        else:
            allowed = ALL_VALID_CODES  # 无法判断时不过滤
        final_codes = [c for c in confirm_result["codes"] if c in allowed]
        return _build_output(
            target, first_result,
            override_codes=final_codes,
            override_addressee=addressee,
            confirmed=True,
            needs_review=False,
            confirm_reasoning=confirm_result.get("reasoning", "")
        )
    else:
        # 复查也失败，保留首次结果，标记人工复查
        return _build_output(target, first_result, confirmed=False, needs_review=True)


def _build_output(target: dict, first_result: dict, *,
                  override_codes=None, override_addressee=None,
                  confirmed: bool, needs_review: bool,
                  confirm_reasoning: str = "") -> dict:
    """统一构建输出格式"""
    codes = override_codes if override_codes is not None else first_result.get("codes", [])
    addressee = override_addressee if override_addressee is not None else first_result.get("step2_addressee", "none")
    return {
        "turn_id": target["turn_id"],
        "speaker": target["speaker"],
        "utterance": target["utterance"][:120] + ("..." if len(target["utterance"]) > 120 else ""),
        "codes": codes,
        "step1_trigger": first_result.get("step1_trigger", "unknown"),
        "step2_addressee": addressee,
        "reasoning": first_result.get("reasoning", ""),
        "confirmed": confirmed,
        "confirm_reasoning": confirm_reasoning,
        "needs_review": needs_review,
    }


# 编码入口
async def _code_full_transcript_async(transcript: list, max_concurrency: int = 5) -> list:
    """对所有教师话轮并发编码"""
    if max_concurrency < 1:
        raise ValueError("max_concurrency 必须大于等于 1")

    client = llm_config.get_async_client()
    semaphore = asyncio.Semaphore(max_concurrency)
    teacher_turns = [
        (array_idx, turn) for array_idx, turn in enumerate(transcript)
        if turn.get("is_teacher", False)
    ]
    print(f"教师话轮总数: {len(teacher_turns)}，并发上限: {max_concurrency}")
    failed_turns = []

    async def _process(idx, array_idx, turn):
        async with semaphore:
            try:
                result = await code_single_teacher_turn(client, transcript, array_idx)
                status = "✓" if not result["needs_review"] else "需复查"
                print(f"  [{idx + 1}/{len(teacher_turns)}] Turn {turn['turn_id']} → "
                      f"{result['codes']} {status}")
                return result
            except Exception as e:
                print(f"  [✗] Turn {turn['turn_id']} 编码失败: {e}")
                failed_turns.append(turn["turn_id"])
                return {
                    "turn_id": turn["turn_id"],
                    "speaker": turn.get("speaker", ""),
                    "utterance": turn.get("utterance", "")[:120],
                    "codes": [],
                    "step1_trigger": "unknown",
                    "step2_addressee": "unknown",
                    "reasoning": f"ERROR: {e}",
                    "confirmed": False,
                    "confirm_reasoning": "",
                    "needs_review": True,
                }

    tasks = [
        _process(idx, array_idx, turn)
        for idx, (array_idx, turn) in enumerate(teacher_turns)
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    if failed_turns:
        print(f"\n失败话轮 (已标记 needs_review): {failed_turns}")
    return list(results)


def code_full_transcript(transcript: list, max_concurrency: int = 5) -> list:
    """同步入口"""
    return asyncio.run(_code_full_transcript_async(transcript, max_concurrency=max_concurrency))
