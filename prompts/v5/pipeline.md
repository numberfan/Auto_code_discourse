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
└─► Stage 3 is kept as a manual/offline review prompt, not part of the default
    automatic path. Re-enable it only after an ablation shows higher F1.

## Context Window:
- Input to all stages: 5 turns before + target turn + 2 turns after
- Include speaker labels for all turns

## Token Estimation per turn:
- Stage 1: ~800 tokens prompt + ~200 context + ~150 output ≈ 1150 tokens
- Stage 2A/2B: ~600 tokens prompt + ~200 context + ~100 output ≈ 900 tokens
- Total per coded turn: ~2050 tokens for trigger=yes turns
- Trigger=no turns stop after Stage 1 to keep the default path simple