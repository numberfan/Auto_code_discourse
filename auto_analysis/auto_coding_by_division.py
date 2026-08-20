#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 10:59
# @Description  : APT coding pipeline with separate trigger, code, and review stages.

import asyncio
from auto_analysis import llm_config
from auto_analysis.utils import extract_json, load_text_file

MODEL_NAME = llm_config.get_model_name()
USE_JSON_MODE = llm_config.supports_json_mode()
llm_config.print_status()

PROMPT_VERSION = "v5"
STAGE1_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/stage1_trigger_addressee.txt")
STAGE1_FEW_SHOT = load_text_file(f"prompts/{PROMPT_VERSION}/stage1_few_shot.txt")
STAGE2A_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/stage2a_type_a.txt")
TYPE_A_FEW_SHOT = load_text_file(f"prompts/{PROMPT_VERSION}/type_a_few_shot.txt")
STAGE2B_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/stage2b_type_b.txt")
TYPE_B_FEW_SHOT = load_text_file(f"prompts/{PROMPT_VERSION}/type_b_few_shot.txt")
STAGE3_PROMPT = load_text_file(f"prompts/{PROMPT_VERSION}/stage3_confirm.txt")

if not all((STAGE1_PROMPT, STAGE1_FEW_SHOT, STAGE2A_PROMPT, TYPE_A_FEW_SHOT,
            STAGE2B_PROMPT, TYPE_B_FEW_SHOT, STAGE3_PROMPT)):
    raise RuntimeError(f"未能完整加载 {PROMPT_VERSION} 分阶段 Prompt 文件")

TYPE_A_CODES = {"say_more", "press_for_reasoning", "revoice", "challenge"}
TYPE_B_CODES = {"add_on", "explain_others", "agree_disagree", "restate"}
ALL_VALID_CODES = TYPE_A_CODES | TYPE_B_CODES
MAX_RETRIES = 3
STAGE1_MAX_OUTPUT_TOKENS = 2048
STAGE2_MAX_OUTPUT_TOKENS = 2048
STAGE3_MAX_OUTPUT_TOKENS = 4096


def _build_context_window(transcript, target_idx, before=5, after=2):
    context_lines = []
    start = max(0, target_idx - before)

    student_before = next(
        (transcript[i]["speaker"] for i in range(target_idx - 1, -1, -1)
         if not transcript[i].get("is_teacher", False)),
        None,
    )
    student_after = next(
        (transcript[i]["speaker"] for i in range(
            target_idx + 1, min(len(transcript), target_idx + after + 1))
         if not transcript[i].get("is_teacher", False)),
        None,
    )

    for i in range(start, target_idx):
        turn = transcript[i]
        context_lines.append(f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}")

    target = transcript[target_idx]
    context_lines.append(
        f"\n>>> [Turn {target['turn_id']}] {target['speaker']}: "
        f"{target['utterance']} <<<\n"
    )

    for i in range(target_idx + 1, min(len(transcript), target_idx + after + 1)):
        turn = transcript[i]
        context_lines.append(f"[Turn {turn['turn_id']}] {turn['speaker']}: {turn['utterance']}")

    context_lines.append(
        f"\n[SPEAKER INFO] Student_Before = {student_before or 'unknown'} | "
        f"Student_After = {student_after or 'unknown'}"
    )
    if student_before and student_after:
        relation = "Same student continues" if student_before == student_after else "Different student responds"
        context_lines.append(f"[SPEAKER INFO] {relation}")
    return "\n".join(context_lines)


def _request_kwargs(messages, max_tokens):
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


async def _call_llm_json(client, messages, max_tokens, label):
    kwargs = _request_kwargs(messages, max_tokens)
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise ValueError("模型返回空内容")
            return extract_json(content)
        except ValueError as exc:
            # 解析失败通常不可通过重试修复，避免重复消耗 token。
            raise RuntimeError(f"{label}: 模型返回无效 JSON: {exc}") from exc
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"{label}: LLM 调用失败: {exc}") from exc
            await asyncio.sleep(2 ** attempt)


def _require_string(result, key, allowed, label):
    value = result.get(key)
    if value not in allowed:
        raise ValueError(f"{label}: {key}={value!r} 无效")
    return value


def _validate_stage1(result, turn_id):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 1: turn_id 不匹配: {result.get('turn_id')!r}")
    trigger = _require_string(result, "trigger", {"yes", "no"}, "Stage 1")
    addressee = _require_string(
        result, "addressee", {"same_student", "other_student", "none"}, "Stage 1"
    )
    if trigger == "no" and addressee != "none":
        raise ValueError("Stage 1: trigger=no 时 addressee 必须为 none")
    if trigger == "yes" and addressee == "none":
        raise ValueError("Stage 1: trigger=yes 时必须给出 addressee")
    return result


def _validate_stage2(result, turn_id, allowed_codes):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 2: turn_id 不匹配: {result.get('turn_id')!r}")
    code = result.get("code")
    if code not in allowed_codes:
        raise ValueError(f"Stage 2: code={code!r} 不属于当前分支")
    return result


def _validate_stage3(result, turn_id):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 3: turn_id 不匹配: {result.get('turn_id')!r}")
    trigger = _require_string(result, "final_trigger", {"yes", "no"}, "Stage 3")
    addressee = _require_string(
        result, "final_addressee", {"same_student", "other_student", "none"}, "Stage 3"
    )
    codes = result.get("final_code")
    if not isinstance(codes, list) or len(codes) > 1 or any(code not in ALL_VALID_CODES for code in codes):
        raise ValueError("Stage 3: final_code 必须是至多一个合法代码的数组")
    if trigger == "no" and (addressee != "none" or codes):
        raise ValueError("Stage 3: trigger=no 时不能有 addressee 或 code")
    if trigger == "yes" and addressee == "none":
        raise ValueError("Stage 3: trigger=yes 时必须给出 addressee")
    if addressee == "same_student" and any(code not in TYPE_A_CODES for code in codes):
        raise ValueError("Stage 3: same_student 与 Type B code 不匹配")
    if addressee == "other_student" and any(code not in TYPE_B_CODES for code in codes):
        raise ValueError("Stage 3: other_student 与 Type A code 不匹配")
    return result


async def predict_stage1(client, transcript, target_idx):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx)
    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TARGET TURN (marked with >>>):\n"
        f">>> [{target['turn_id']}] {target['speaker']}: {target['utterance']} <<<\n\n"
        "Determine trigger and addressee. Return ONLY JSON with keys: "
        "turn_id, trigger, addressee, discussion_active, evidence, reasoning."
    )
    messages = [
        {"role": "system", "content": STAGE1_PROMPT + "\n\n" + STAGE1_FEW_SHOT},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE1_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 1")
    return _validate_stage1(result, target["turn_id"])


async def predict_stage2(client, transcript, target_idx, addressee):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx)
    if addressee == "same_student":
        system_prompt = STAGE2A_PROMPT + "\n\n" + TYPE_A_FEW_SHOT
        allowed_codes = TYPE_A_CODES
        branch = "Type A"
    else:
        system_prompt = STAGE2B_PROMPT + "\n\n" + TYPE_B_FEW_SHOT
        allowed_codes = TYPE_B_CODES
        branch = "Type B"
    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TARGET TURN:\n{target['utterance']}\n\n"
        f"Assign exactly one {branch} code. Return ONLY JSON with keys: turn_id, code, reasoning."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE2_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 2")
    return _validate_stage2(result, target["turn_id"], allowed_codes)


async def predict_stage3(client, transcript, target_idx, stage1_result, stage2_result, issue_description):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx, before=7, after=3)
    user_prompt = (
        f"## FIRST-PASS RESULT:\n"
        f"Stage 1: trigger={stage1_result.get('trigger')}, addressee={stage1_result.get('addressee')}\n"
        f"Stage 2: code={stage2_result.get('code') if stage2_result else None}\n"
        f"Confidence issue: {issue_description}\n\n"
        f"## TRANSCRIPT CONTEXT:\n{context}\n\n"
        f"## TARGET: Turn {target['turn_id']} - {target['utterance']}\n\n"
        "Make your FINAL decision. Return ONLY JSON with keys: turn_id, final_trigger, "
        "final_addressee, final_code, override_reason."
    )
    messages = [
        {"role": "system", "content": STAGE3_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(
        client, messages, STAGE3_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 3"
    )
    return _validate_stage3(result, target["turn_id"])


def need_review(stage1_result, stage2_result):
    uncertainty_markers = [
        "probably", "maybe", "uncertain", "possibly", "either",
        "not sure", "could be", "unclear",
    ]
    stage1_reasoning = stage1_result.get("reasoning", "").lower()
    stage2_reasoning = stage2_result.get("reasoning", "").lower()
    high_risk_codes = {"add_on", "revoice", "challenge"}
    return (
        any(marker in stage1_reasoning for marker in uncertainty_markers)
        or any(marker in stage2_reasoning for marker in uncertainty_markers)
        or stage2_result.get("code") in high_risk_codes
    )

def _build_output(target, *, trigger, addressee, codes, reasoning, confirmed, needs_review, stage1_evidence="", stage2_reasoning="", confirm_reasoning="",):
    return {
        "turn_id": target["turn_id"],
        "speaker": target["speaker"],
        "utterance": target["utterance"][:120] + ("..." if len(target["utterance"]) > 120 else ""),
        "utterance_full": target["utterance"],
        "codes": codes,
        "step1_trigger": trigger,
        "step2_addressee": addressee,
        "reasoning": reasoning,
        "stage1_evidence": stage1_evidence,
        "stage2_reasoning": stage2_reasoning,
        "confirmed": confirmed,
        "confirm_reasoning": confirm_reasoning,
        "needs_review": needs_review,
    }


async def code_single_teacher_turn(client, transcript, target_idx):
    target = transcript[target_idx]
    try:
        stage1 = await predict_stage1(client, transcript, target_idx)
        trigger = stage1["trigger"]
        addressee = stage1["addressee"]
        if trigger == "no":
            return _build_output(
                target, trigger="no", addressee="none", codes=[],
                reasoning=stage1.get("reasoning", ""), confirmed=True, needs_review=False,
            )

        stage2 = await predict_stage2(client, transcript, target_idx, addressee)
        if need_review(stage1, stage2):
            stage3 = await predict_stage3(
                client, transcript, target_idx, stage1, stage2,
                "low confidence or high-risk code",
            )
            return _build_output(
                target,
                trigger=stage3["final_trigger"],
                addressee=stage3["final_addressee"],
                codes=stage3["final_code"],
                reasoning=stage1.get("reasoning", ""),
                stage1_evidence=stage1.get("evidence", ""),
                stage2_reasoning=stage2.get("reasoning", ""),
                confirmed=True,
                needs_review=False,
                confirm_reasoning=stage3.get("override_reason", ""),
            )
        return _build_output(
            target, trigger=trigger, addressee=addressee, codes=[stage2["code"]],
            reasoning=stage1.get("reasoning", ""),
            stage1_evidence=stage1.get("evidence", ""),
            stage2_reasoning=stage2.get("reasoning", ""),
            confirmed=True, needs_review=False,
        )
    except Exception as exc:
        print(f"[错误] Turn {target['turn_id']}: {exc}")
        return _build_output(
            target, trigger="unknown", addressee="unknown", codes=[],
            reasoning=f"ERROR: {exc}", confirmed=False, needs_review=True,
        )


async def _code_full_transcript_async(transcript, max_concurrency=5):
    if max_concurrency < 1:
        raise ValueError("max_concurrency 必须大于等于 1")
    client = llm_config.get_async_client()
    semaphore = asyncio.Semaphore(max_concurrency)
    teacher_turns = [(i, turn) for i, turn in enumerate(transcript) if turn.get("is_teacher", False)]
    print(f"教师话轮总数: {len(teacher_turns)}，并发上限: {max_concurrency}")

    async def process(index, array_idx, turn):
        async with semaphore:
            result = await code_single_teacher_turn(client, transcript, array_idx)
            status = "✓" if not result["needs_review"] else "需复查"
            print(f"  [{index + 1}/{len(teacher_turns)}] Turn {turn['turn_id']} → {result['codes']} {status}")
            return result

    try:
        results = await asyncio.gather(*(process(i, idx, turn) for i, (idx, turn) in enumerate(teacher_turns)))
    finally:
        await client.close()
    return list(results)


def code_full_transcript(transcript, max_concurrency=5):
    return asyncio.run(_code_full_transcript_async(transcript, max_concurrency))
