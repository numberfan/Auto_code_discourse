# APT Coding Pipeline v5 — Architecture
## Flow:
For each teacher turn:
│
├─► Stage 1: Trigger + Addressee (stage1_trigger_addressee.txt)
│   ├─ Output: {trigger, addressee, discussion_active, evidence}
│   │
│   ├─ If trigger = "no" → Final output: codes = []
│   │
│   ├─ If trigger = "yes" AND addressee = "same_student"
│   │   └─► Stage 2A (stage2a_type_a.txt)
│   │       └─ Output: one of [revoice, challenge, press_for_reasoning, say_more]
│   │
│   └─ If trigger = "yes" AND addressee = "other_student"
│       └─► Stage 2B (stage2b_type_b.txt)
│           └─ Output: one of [explain_others, agree_disagree, restate, add_on]
│
└─► (Optional) Stage 3: Review (stage3_review.txt)
└─ Only triggered when Stage 1 or Stage 2 confidence is low

## Confidence Thresholds (for Stage 3 routing):
- Stage 1: If model expresses uncertainty in reasoning → route to review
- Stage 2: If reasoning mentions multiple possible codes → route to review

## Context Window:
- Input to all stages: 5 turns before + target turn + 2 turns after
- Include speaker labels for all turns

## Token Estimation per turn:
- Stage 1: ~800 tokens prompt + ~200 context + ~150 output ≈ 1150 tokens
- Stage 2A/2B: ~600 tokens prompt + ~200 context + ~100 output ≈ 900 tokens
- Total per coded turn: ~2050 tokens (vs current ~2500)
- Savings: ~18% fewer tokens overall, with better accuracy