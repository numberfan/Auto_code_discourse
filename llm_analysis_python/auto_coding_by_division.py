#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2026/8/17 10:59
# @Description  : APT coding pipeline with separate trigger, code, and review stages.

import asyncio
from llm_analysis_python import llm_config
from llm_analysis_python.utils import extract_json, load_text_file

MODEL_NAME = llm_config.get_model_name()
USE_JSON_MODE = llm_config.supports_json_mode()
llm_config.print_status()

PROMPT_VERSION = "v8.1"
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
STAGE2_MAX_OUTPUT_TOKENS = 4096
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


def _get_immediate_student_response(transcript, target_idx):
    """Return only the first turn after the target, if it is a student turn."""
    response_idx = target_idx + 1
    if response_idx >= len(transcript):
        return None
    response = transcript[response_idx]
    if response.get("is_teacher", False):
        return None
    return response


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
    confidence = result.get("confidence")
    if confidence is None:
        result["confidence"] = "low"
    else:
        _require_string(result, "confidence", {"high", "medium", "low"}, "Stage 1")
    addressee = _require_string(
        result, "addressee", {"same_student", "other_student", "none"}, "Stage 1"
    )
    response_status = _require_string(
        result, "response_status", {"present", "absent", "weak", "unclear"}, "Stage 1"
    )
    response_relation = _require_string(
        result, "response_relation", {"same_student", "other_student", "none", "unknown"}, "Stage 1"
    )
    response_turn_id = result.get("response_turn_id")
    if response_turn_id is not None and not isinstance(response_turn_id, int):
        raise ValueError("Stage 1: response_turn_id 必须是整数或 null")
    if response_status == "absent" and response_relation != "none":
        raise ValueError("Stage 1: response_status=absent 时 response_relation 必须为 none")
    if response_status == "absent" and response_turn_id is not None:
        raise ValueError("Stage 1: response_status=absent 时 response_turn_id 必须为 null")
    if response_status == "present" and response_relation == "none":
        raise ValueError("Stage 1: response_status=present 时必须标注 response_relation")
    if response_status == "present" and response_turn_id is None:
        raise ValueError("Stage 1: response_status=present 时必须给出 response_turn_id")
    if trigger == "no" and addressee != "none":
        raise ValueError("Stage 1: trigger=no 时 addressee 必须为 none")
    if trigger == "yes" and addressee == "none":
        raise ValueError("Stage 1: trigger=yes 时必须给出 addressee")
    return result


def _validate_stage2(result, turn_id, allowed_codes):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 2: turn_id 不匹配: {result.get('turn_id')!r}")
    _require_string(result, "confidence", {"high", "medium", "low"}, "Stage 2")
    codes = result.get("codes")
    if not isinstance(codes, list) or not codes:
        raise ValueError("Stage 2: codes 必须至少包含一个合法代码")
    if len(set(codes)) != len(codes) or any(code not in allowed_codes for code in codes):
        raise ValueError(f"Stage 2: codes={codes!r} 不属于当前分支或包含重复代码")
    had_multiple_codes = len(codes) > 1
    result["codes"] = _normalize_codes(codes, allowed_codes)
    if len(result["codes"]) > 1:
        result["codes"] = [result["codes"][0]]
    if had_multiple_codes:
        result["confidence"] = "low"
    return result


def _validate_stage3(result, turn_id):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 3: turn_id 不匹配: {result.get('turn_id')!r}")
    trigger = _require_string(result, "final_trigger", {"yes", "no"}, "Stage 3")
    addressee = _require_string(
        result, "final_addressee", {"same_student", "other_student", "none"}, "Stage 3"
    )
    codes = result.get("final_codes")
    if not isinstance(codes, list) or any(code not in ALL_VALID_CODES for code in codes):
        raise ValueError("Stage 3: final_codes 必须是合法代码数组")
    allowed_codes = (
        TYPE_A_CODES if addressee == "same_student"
        else TYPE_B_CODES if addressee == "other_student"
        else ALL_VALID_CODES
    )
    result["final_codes"] = _normalize_codes(codes, allowed_codes)
    if len(result["final_codes"]) > 1:
        result["final_codes"] = [result["final_codes"][0]]
    if trigger == "no" and (addressee != "none" or codes):
        raise ValueError("Stage 3: trigger=no 时不能有 addressee 或 code")
    if trigger == "yes" and addressee == "none":
        raise ValueError("Stage 3: trigger=yes 时必须给出 addressee")
    if trigger == "yes" and not result["final_codes"]:
        raise ValueError("Stage 3: trigger=yes 时必须至少给出一个 code")
    return result


def _normalize_codes(codes, allowed_codes):
    """Remove residual default codes when a more specific code is present."""
    normalized = list(dict.fromkeys(code for code in codes if code in allowed_codes))
    if allowed_codes == TYPE_A_CODES:
        specific_codes = TYPE_A_CODES - {"say_more"}
        if set(normalized) & specific_codes:
            normalized = [code for code in normalized if code != "say_more"]
    elif allowed_codes == TYPE_B_CODES:
        specific_codes = TYPE_B_CODES - {"add_on"}
        if set(normalized) & specific_codes:
            normalized = [code for code in normalized if code != "add_on"]
    return normalized


async def predict_stage1(client, transcript, target_idx):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx, after=1)
    response = _get_immediate_student_response(transcript, target_idx)
    if response:
        response_evidence = (
            f"[RESPONSE EVIDENCE] Immediate next student turn only: "
            f"[Turn {response['turn_id']}] {response['speaker']}: {response['utterance']}"
        )
    else:
        response_evidence = (
            "[RESPONSE EVIDENCE] No immediate student response follows this teacher turn. "
            "Do not use later turns as evidence."
        )
    user_prompt = (
        f"## CONTEXT:\n{context}\n\n"
        f"## TARGET TURN (marked with >>>):\n"
        f">>> [{target['turn_id']}] {target['speaker']}: {target['utterance']} <<<\n\n"
        f"## RESPONSE EVIDENCE:\n{response_evidence}\n\n"
        "Determine trigger and addressee. Return ONLY JSON with keys: "
        "turn_id, trigger, addressee, response_status, response_turn_id, "
        "response_relation, confidence, evidence, reasoning."
    )
    messages = [
        {"role": "system", "content": STAGE1_PROMPT + "\n\n" + STAGE1_FEW_SHOT},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE1_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 1")
    return _validate_stage1(result, target["turn_id"])


async def predict_stage2(client, transcript, target_idx, addressee):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx, after=1)
    response = _get_immediate_student_response(transcript, target_idx)
    response_evidence = (
        f"[Turn {response['turn_id']}] {response['speaker']}: {response['utterance']}"
        if response else "No immediate student response follows the target teacher turn."
    )
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
        f"## IMMEDIATE RESPONSE EVIDENCE (use only this student response):\n{response_evidence}\n\n"
        f"Assign exactly one primary {branch} code. Return ONLY JSON with keys: turn_id, codes, confidence, reasoning."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE2_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 2")
    return _validate_stage2(result, target["turn_id"], allowed_codes)


async def predict_stage3(client, transcript, target_idx, stage1_result, stage2_result, issue_description):
    target = transcript[target_idx]
    context = _build_context_window(transcript, target_idx, before=7, after=1)
    response = _get_immediate_student_response(transcript, target_idx)
    response_evidence = (
        f"[Turn {response['turn_id']}] {response['speaker']}: {response['utterance']}"
        if response else "No immediate student response follows the target teacher turn."
    )
    user_prompt = (
        f"## FIRST-PASS RESULT:\n"
        f"Stage 1: trigger={stage1_result.get('trigger')}, addressee={stage1_result.get('addressee')}\n"
        f"Response status={stage1_result.get('response_status')}, "
        f"response_turn_id={stage1_result.get('response_turn_id')}, "
        f"response_relation={stage1_result.get('response_relation')}\n"
        f"Stage 2: codes={stage2_result.get('codes') if stage2_result else None}\n"
        f"Confidence issue: {issue_description}\n\n"
        f"## TRANSCRIPT CONTEXT:\n{context}\n\n"
        f"## IMMEDIATE RESPONSE EVIDENCE (use only this student response):\n{response_evidence}\n\n"
        f"## TARGET: Turn {target['turn_id']} - {target['utterance']}\n\n"
        "Make your FINAL decision. Return ONLY JSON with keys: turn_id, final_trigger, "
        "final_addressee, final_codes, override_reason."
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
    high_risk_codes = {"add_on", "revoice", "press_for_reasoning", "challenge"}
    return (
        stage1_result.get("confidence") == "low"
        or stage2_result.get("confidence") == "low"
        or bool(set(stage2_result.get("codes", [])) & high_risk_codes)
    )

def _build_output(target, *, trigger, addressee, codes, reasoning, confirmed,
                  needs_review, stage1_evidence="", stage1_confidence="",
                  stage2_reasoning="", stage2_confidence="", confirm_reasoning="",
                  response_status="unknown", response_turn_id=None,
                  response_relation="unknown",
                  reviewed=False):
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
        "stage1_confidence": stage1_confidence,
        "response_status": response_status,
        "response_turn_id": response_turn_id,
        "response_relation": response_relation,
        "stage2_reasoning": stage2_reasoning,
        "stage2_confidence": stage2_confidence,
        "confirmed": confirmed,
        "reviewed": reviewed,
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
                reasoning=stage1.get("reasoning", ""),
                stage1_evidence=stage1.get("evidence", ""),
                stage1_confidence=stage1.get("confidence", ""),
                response_status=stage1.get("response_status", "unknown"),
                response_turn_id=stage1.get("response_turn_id"),
                response_relation=stage1.get("response_relation", "unknown"),
                confirmed=True, needs_review=False,
            )

        stage2 = await predict_stage2(client, transcript, target_idx, addressee)
        if need_review(stage1, stage2):
            try:
                stage3 = await predict_stage3(
                    client, transcript, target_idx, stage1, stage2,
                    "low confidence or high-risk code",
                )
                return _build_output(
                    target,
                    trigger=stage3["final_trigger"],
                    addressee=stage3["final_addressee"],
                    codes=stage3["final_codes"],
                    reasoning=stage1.get("reasoning", ""),
                    stage1_evidence=stage1.get("evidence", ""),
                    stage1_confidence=stage1.get("confidence", ""),
                    response_status=stage1.get("response_status", "unknown"),
                    response_turn_id=stage1.get("response_turn_id"),
                    response_relation=stage1.get("response_relation", "unknown"),
                    stage2_reasoning=stage2.get("reasoning", ""),
                    stage2_confidence=stage2.get("confidence", ""),
                    confirmed=True,
                    needs_review=False,
                    reviewed=True,
                    confirm_reasoning=stage3.get("override_reason", ""),
                )
            except Exception as exc:
                print(f"[复查失败] Turn {target['turn_id']}: {exc}，保留 Stage 2 结果")
                return _build_output(
                    target, trigger=trigger, addressee=addressee,
                    codes=stage2["codes"], reasoning=stage1.get("reasoning", ""),
                    stage1_evidence=stage1.get("evidence", ""),
                    stage1_confidence=stage1.get("confidence", ""),
                    response_status=stage1.get("response_status", "unknown"),
                    response_turn_id=stage1.get("response_turn_id"),
                    response_relation=stage1.get("response_relation", "unknown"),
                    stage2_reasoning=stage2.get("reasoning", ""),
                    stage2_confidence=stage2.get("confidence", ""),
                    confirmed=False, needs_review=True,
                    confirm_reasoning=f"Stage 3 failed: {exc}",
                )
        return _build_output(
            target, trigger=trigger, addressee=addressee, codes=stage2["codes"],
            reasoning=stage1.get("reasoning", ""),
            stage1_evidence=stage1.get("evidence", ""),
            stage1_confidence=stage1.get("confidence", ""),
            response_status=stage1.get("response_status", "unknown"),
            response_turn_id=stage1.get("response_turn_id"),
            response_relation=stage1.get("response_relation", "unknown"),
            stage2_reasoning=stage2.get("reasoning", ""),
            stage2_confidence=stage2.get("confidence", ""),
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
