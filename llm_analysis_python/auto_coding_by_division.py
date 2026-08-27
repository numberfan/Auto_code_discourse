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

PROMPT_VERSION = "v11"
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
HIGH_RISK_CODES = {"add_on", "revoice", "press_for_reasoning", "challenge"}
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
            raise RuntimeError(f"{label}: 模型返回无效 JSON: {exc}") from exc
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"{label}: LLM 调用失败: {exc}") from exc
            await asyncio.sleep(2 ** attempt)


def _clean_text(value, default=""):
    if isinstance(value, str):
        text = value.strip()
        return text or default
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _choice(result, key, allowed, default=None):
    value = result.get(key)
    if value not in allowed:
        value = default
    if value not in allowed:
        raise ValueError(f"{key}={value!r} 无效")
    result[key] = value
    return value


def _default_trigger_basis(trigger, addressee, invitation_status):
    if trigger == "no":
        return "teacher_led"
    if addressee == "other_student" or invitation_status != "none":
        return "peer_linked_invitation"
    return "direct_follow_up"


def _default_invitation_status(trigger, addressee, trigger_basis):
    if trigger == "yes" and addressee == "other_student" and trigger_basis == "peer_linked_invitation":
        return "implicit_peer_linked"
    return "none"


def _default_evidence_mode(trigger_basis, response_status):
    if trigger_basis == "peer_linked_invitation":
        return "invitation_based"
    if response_status == "present":
        return "response_based"
    return "invitation_based"


def _default_addressee_basis(addressee):
    if addressee == "same_student":
        return "same student follow-up to prior contribution"
    if addressee == "other_student":
        return "invites other students into the same discussion"
    return "no addressee because trigger is no"


def _normalize_stage1(result, target):
    result = dict(result)
    result["turn_id"] = target["turn_id"]

    trigger = _choice(result, "trigger", {"yes", "no"}, default="no")
    addressee = _choice(
        result, "addressee", {"same_student", "other_student", "none"}, default="none"
    )
    confidence = _choice(result, "confidence", {"high", "medium", "low"}, default="low")

    response_status = _choice(
        result, "response_status", {"present", "absent", "weak", "unclear"}, default="absent"
    )
    if response_status == "absent":
        response_relation = "none"
        response_turn_id = None
    else:
        response_relation = _choice(
            result,
            "response_relation",
            {"same_student", "other_student", "none", "unknown"},
            default="unknown",
        )
        response_turn_id = result.get("response_turn_id")
        if not isinstance(response_turn_id, int):
            response_turn_id = None
    result["response_status"] = response_status
    result["response_relation"] = response_relation
    result["response_turn_id"] = response_turn_id

    raw_trigger_basis = result.get("trigger_basis")
    raw_invitation_status = result.get("invitation_status")
    invitation_status_default = _default_invitation_status(trigger, addressee, "peer_linked_invitation")
    invitation_status = raw_invitation_status if raw_invitation_status in {"explicit_peer_linked", "implicit_peer_linked", "none"} else invitation_status_default
    trigger_basis = raw_trigger_basis if raw_trigger_basis in {"direct_follow_up", "peer_linked_invitation", "new_question", "teacher_led"} else _default_trigger_basis(trigger, addressee, invitation_status)
    invitation_status = _choice(
        {"value": invitation_status},
        "value",
        {"explicit_peer_linked", "implicit_peer_linked", "none"},
        default=_default_invitation_status(trigger, addressee, trigger_basis),
    )
    evidence_mode = _choice(
        {"value": result.get("evidence_mode")},
        "value",
        {"response_based", "invitation_based"},
        default=_default_evidence_mode(trigger_basis, response_status),
    )

    if trigger == "no":
        addressee = "none"
        if trigger_basis not in {"new_question", "teacher_led"}:
            trigger_basis = "teacher_led"
        invitation_status = "none"
    else:
        if addressee == "none":
            addressee = "other_student" if trigger_basis == "peer_linked_invitation" else "same_student"
        if trigger_basis in {"new_question", "teacher_led"}:
            trigger_basis = _default_trigger_basis(trigger, addressee, invitation_status)
        if addressee != "other_student":
            invitation_status = "none"
        elif trigger_basis == "peer_linked_invitation" and invitation_status == "none":
            invitation_status = "implicit_peer_linked"

    result["trigger"] = trigger
    result["addressee"] = addressee
    result["confidence"] = confidence
    result["trigger_basis"] = trigger_basis
    result["invitation_status"] = invitation_status
    result["evidence_mode"] = evidence_mode
    result["addressee_basis"] = _clean_text(result.get("addressee_basis"), _default_addressee_basis(addressee))
    result["evidence"] = _clean_text(result.get("evidence"), target.get("utterance", "")[:60] or "no evidence")
    result["reasoning"] = _clean_text(result.get("reasoning"), "no reasoning")
    return result


def _validate_stage1(result, target):
    result = _normalize_stage1(result, target)
    if result.get("turn_id") != target["turn_id"]:
        raise ValueError(f"Stage 1: turn_id 不匹配: {result.get('turn_id')!r}")
    return result


def _normalize_codes(codes, allowed_codes):
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


def _validate_stage2(result, turn_id, allowed_codes):
    if result.get("turn_id") != turn_id:
        raise ValueError(f"Stage 2: turn_id 不匹配: {result.get('turn_id')!r}")
    _choice(result, "confidence", {"high", "medium", "low"}, default="low")
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
    result["reasoning"] = _clean_text(result.get("reasoning"), "no reasoning")
    return result


def _validate_stage3(result, turn_id):
    result = dict(result)
    result["turn_id"] = turn_id
    final_trigger = _choice(result, "final_trigger", {"yes", "no"}, default="no")
    final_addressee = _choice(
        result, "final_addressee", {"same_student", "other_student", "none"}, default="none"
    )
    codes = result.get("final_codes")
    if not isinstance(codes, list):
        codes = []
    if final_trigger == "no":
        result["final_addressee"] = "none"
        result["final_codes"] = []
    else:
        if final_addressee == "none":
            if any(code in TYPE_B_CODES for code in codes):
                final_addressee = "other_student"
            else:
                final_addressee = "same_student"
            result["final_addressee"] = final_addressee
        allowed_codes = TYPE_A_CODES if final_addressee == "same_student" else TYPE_B_CODES
        result["final_codes"] = _normalize_codes(codes, allowed_codes)
        if len(result["final_codes"]) > 1:
            result["final_codes"] = [result["final_codes"][0]]
        if not result["final_codes"]:
            raise ValueError("Stage 3: trigger=yes 时必须至少给出一个 code")
    result["override_reason"] = _clean_text(result.get("override_reason"), "risk-based review")
    return result


def _contains_any(text, patterns):
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in patterns)


def _derive_risk_flags(target, stage1_result, stage2_result):
    flags = []
    utterance = target.get("utterance", "")
    lowered = utterance.lower()
    response_status = stage1_result.get("response_status")
    trigger_basis = stage1_result.get("trigger_basis")
    evidence_mode = stage1_result.get("evidence_mode")
    addressee = stage1_result.get("addressee")
    codes = stage2_result.get("codes", []) if stage2_result else []

    if stage1_result.get("confidence") != "high":
        flags.append(f"stage1_confidence:{stage1_result.get('confidence', 'unknown')}")
    if stage2_result and stage2_result.get("confidence") != "high":
        flags.append(f"stage2_confidence:{stage2_result.get('confidence', 'unknown')}")
    for code in codes:
        if code in HIGH_RISK_CODES:
            flags.append(f"high_risk_code:{code}")
    if trigger_basis == "peer_linked_invitation" and response_status != "present":
        flags.append("invitation_without_response")
    if evidence_mode == "invitation_based" and _contains_any(
        lowered, ["what do you think", "anything else", "anyone else", "how else", "what else"]
    ):
        flags.append("peer_invitation_question_form")
    if addressee == "other_student" and _contains_any(
        lowered, ["why", "reason", "evidence", "justify", "justification"]
    ):
        flags.append("why_to_other_student")
    if _contains_any(
        lowered, ["best use", "do you think", "is that right", "that's the reason why", "that is the reason why"]
    ):
        flags.append("revoice_or_challenge_boundary")
    if _contains_any(lowered, ["why", "how", "what difference", "what effect", "because what"]):
        flags.append("say_more_or_reasoning_boundary")
    if trigger_basis == "peer_linked_invitation" and _contains_any(
        lowered, ["let's go back", "now", "really quick"]
    ):
        flags.append("topic_shift_but_peer_linked")
    return list(dict.fromkeys(flags))


def need_review(stage1_result, stage2_result, target):
    risk_flags = _derive_risk_flags(target, stage1_result, stage2_result)
    return bool(risk_flags), risk_flags


def _build_basis(stage1_result, stage2_result=None, stage3_result=None):
    return {
        "stage1": {
            "trigger": stage1_result.get("trigger"),
            "addressee": stage1_result.get("addressee"),
            "trigger_basis": stage1_result.get("trigger_basis", ""),
            "addressee_basis": stage1_result.get("addressee_basis", ""),
            "evidence_mode": stage1_result.get("evidence_mode", ""),
            "invitation_status": stage1_result.get("invitation_status", ""),
            "response_status": stage1_result.get("response_status", "unknown"),
            "response_turn_id": stage1_result.get("response_turn_id"),
            "response_relation": stage1_result.get("response_relation", "unknown"),
            "evidence": stage1_result.get("evidence", ""),
            "reasoning": stage1_result.get("reasoning", ""),
            "confidence": stage1_result.get("confidence", ""),
        },
        "stage2": {
            "codes": stage2_result.get("codes", []) if stage2_result else [],
            "reasoning": stage2_result.get("reasoning", "") if stage2_result else "",
            "confidence": stage2_result.get("confidence", "") if stage2_result else "",
        },
        "stage3": {
            "reviewed": bool(stage3_result),
            "final_trigger": stage3_result.get("final_trigger") if stage3_result else None,
            "final_addressee": stage3_result.get("final_addressee") if stage3_result else None,
            "final_codes": stage3_result.get("final_codes") if stage3_result else [],
            "override_reason": stage3_result.get("override_reason", "") if stage3_result else "",
        },
    }


def _format_issue_description(risk_flags):
    return ", ".join(risk_flags) if risk_flags else "no explicit risk flags"


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
        "Use dual evidence: (1) the teacher turn itself and its link to the prior student idea; "
        "(2) the immediate response evidence shown above. The immediate response is supporting evidence, "
        "not a hard gate for an explicit peer-linked invitation. Return ONLY JSON with keys: "
        "turn_id, trigger, addressee, response_status, response_turn_id, response_relation, "
        "trigger_basis, addressee_basis, evidence_mode, invitation_status, confidence, evidence, reasoning."
    )
    messages = [
        {"role": "system", "content": STAGE1_PROMPT + "\n\n" + STAGE1_FEW_SHOT},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE1_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 1")
    return _validate_stage1(result, target)


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


async def predict_stage3(client, transcript, target_idx, stage1_result, stage2_result, issue_description, risk_flags):
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
        f"Trigger basis={stage1_result.get('trigger_basis')}, addressee basis={stage1_result.get('addressee_basis')}\n"
        f"Evidence mode={stage1_result.get('evidence_mode')}, invitation status={stage1_result.get('invitation_status')}\n"
        f"Response status={stage1_result.get('response_status')}, "
        f"response_turn_id={stage1_result.get('response_turn_id')}, "
        f"response_relation={stage1_result.get('response_relation')}\n"
        f"Stage 2: codes={stage2_result.get('codes') if stage2_result else None}\n"
        f"Risk flags: {', '.join(risk_flags) if risk_flags else 'none'}\n"
        f"Review issue: {issue_description}\n\n"
        f"## TRANSCRIPT CONTEXT:\n{context}\n\n"
        f"## IMMEDIATE RESPONSE EVIDENCE (use only this student response):\n{response_evidence}\n\n"
        f"## TARGET: Turn {target['turn_id']} - {target['utterance']}\n\n"
        "Make your FINAL decision using dual evidence. The immediate response is supporting evidence, "
        "not a hard gate for an explicit peer-linked invitation. Return ONLY JSON with keys: turn_id, "
        "final_trigger, final_addressee, final_codes, override_reason."
    )
    messages = [
        {"role": "system", "content": STAGE3_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = await _call_llm_json(client, messages, STAGE3_MAX_OUTPUT_TOKENS, f"Turn {target['turn_id']} Stage 3")
    return _validate_stage3(result, target["turn_id"])


def _build_output(target, *, trigger, addressee, codes, reasoning, confirmed,
                  needs_review, stage1_evidence="", stage1_confidence="",
                  stage2_reasoning="", stage2_confidence="", confirm_reasoning="",
                  response_status="unknown", response_turn_id=None,
                  response_relation="unknown", trigger_basis="",
                  addressee_basis="", evidence_mode="",
                  invitation_status="none", risk_flags=None,
                  basis=None, reviewed=False):
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
        "trigger_basis": trigger_basis,
        "addressee_basis": addressee_basis,
        "evidence_mode": evidence_mode,
        "invitation_status": invitation_status,
        "stage2_reasoning": stage2_reasoning,
        "stage2_confidence": stage2_confidence,
        "risk_flags": risk_flags or [],
        "basis": basis or {},
        "confirmed": confirmed,
        "reviewed": reviewed,
        "confirm_reasoning": confirm_reasoning,
        "needs_review": needs_review,
    }


async def code_single_teacher_turn(client, transcript, target_idx):
    target = transcript[target_idx]
    try:
        stage1 = await predict_stage1(client, transcript, target_idx)
        if stage1["trigger"] == "no":
            return _build_output(
                target,
                trigger="no",
                addressee="none",
                codes=[],
                reasoning=stage1.get("reasoning", ""),
                stage1_evidence=stage1.get("evidence", ""),
                stage1_confidence=stage1.get("confidence", "low"),
                response_status=stage1.get("response_status", "unknown"),
                response_turn_id=stage1.get("response_turn_id"),
                response_relation=stage1.get("response_relation", "unknown"),
                trigger_basis=stage1.get("trigger_basis", "teacher_led"),
                addressee_basis=stage1.get("addressee_basis", ""),
                evidence_mode=stage1.get("evidence_mode", "invitation_based"),
                invitation_status=stage1.get("invitation_status", "none"),
                risk_flags=[],
                basis=_build_basis(stage1),
                confirmed=True,
                needs_review=False,
            )

        stage2 = await predict_stage2(client, transcript, target_idx, stage1["addressee"])
        needs_review, risk_flags = need_review(stage1, stage2, target)
        if needs_review:
            try:
                stage3 = await predict_stage3(
                    client,
                    transcript,
                    target_idx,
                    stage1,
                    stage2,
                    _format_issue_description(risk_flags),
                    risk_flags,
                )
                return _build_output(
                    target,
                    trigger=stage3["final_trigger"],
                    addressee=stage3["final_addressee"],
                    codes=stage3["final_codes"],
                    reasoning=stage1.get("reasoning", ""),
                    stage1_evidence=stage1.get("evidence", ""),
                    stage1_confidence=stage1.get("confidence", "low"),
                    response_status=stage1.get("response_status", "unknown"),
                    response_turn_id=stage1.get("response_turn_id"),
                    response_relation=stage1.get("response_relation", "unknown"),
                    trigger_basis=stage1.get("trigger_basis", ""),
                    addressee_basis=stage1.get("addressee_basis", ""),
                    evidence_mode=stage1.get("evidence_mode", ""),
                    invitation_status=stage1.get("invitation_status", "none"),
                    stage2_reasoning=stage2.get("reasoning", ""),
                    stage2_confidence=stage2.get("confidence", ""),
                    risk_flags=risk_flags,
                    basis=_build_basis(stage1, stage2, stage3),
                    confirmed=True,
                    needs_review=False,
                    reviewed=True,
                    confirm_reasoning=stage3.get("override_reason", ""),
                )
            except Exception as exc:
                print(f"[复查失败] Turn {target['turn_id']}: {exc}，保留 Stage 2 结果")
                return _build_output(
                    target,
                    trigger=stage1["trigger"],
                    addressee=stage1["addressee"],
                    codes=stage2["codes"],
                    reasoning=stage1.get("reasoning", ""),
                    stage1_evidence=stage1.get("evidence", ""),
                    stage1_confidence=stage1.get("confidence", "low"),
                    response_status=stage1.get("response_status", "unknown"),
                    response_turn_id=stage1.get("response_turn_id"),
                    response_relation=stage1.get("response_relation", "unknown"),
                    trigger_basis=stage1.get("trigger_basis", ""),
                    addressee_basis=stage1.get("addressee_basis", ""),
                    evidence_mode=stage1.get("evidence_mode", ""),
                    invitation_status=stage1.get("invitation_status", "none"),
                    stage2_reasoning=stage2.get("reasoning", ""),
                    stage2_confidence=stage2.get("confidence", ""),
                    risk_flags=risk_flags,
                    basis=_build_basis(stage1, stage2),
                    confirmed=False,
                    needs_review=True,
                    confirm_reasoning=f"Stage 3 failed: {exc}",
                )

        return _build_output(
            target,
            trigger=stage1["trigger"],
            addressee=stage1["addressee"],
            codes=stage2["codes"],
            reasoning=stage1.get("reasoning", ""),
            stage1_evidence=stage1.get("evidence", ""),
            stage1_confidence=stage1.get("confidence", "low"),
            response_status=stage1.get("response_status", "unknown"),
            response_turn_id=stage1.get("response_turn_id"),
            response_relation=stage1.get("response_relation", "unknown"),
            trigger_basis=stage1.get("trigger_basis", ""),
            addressee_basis=stage1.get("addressee_basis", ""),
            evidence_mode=stage1.get("evidence_mode", ""),
            invitation_status=stage1.get("invitation_status", "none"),
            stage2_reasoning=stage2.get("reasoning", ""),
            stage2_confidence=stage2.get("confidence", ""),
            risk_flags=risk_flags,
            basis=_build_basis(stage1, stage2),
            confirmed=True,
            needs_review=False,
        )
    except Exception as exc:
        print(f"[错误] Turn {target['turn_id']}: {exc}")
        return _build_output(
            target,
            trigger="unknown",
            addressee="unknown",
            codes=[],
            reasoning=f"ERROR: {exc}",
            confirmed=False,
            needs_review=True,
            trigger_basis="teacher_led",
            addressee_basis="pipeline error fallback",
            evidence_mode="invitation_based",
            invitation_status="none",
            risk_flags=["pipeline_error"],
            basis={"error": str(exc)},
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
