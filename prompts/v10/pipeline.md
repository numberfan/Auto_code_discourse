# APT Coding Pipeline v10

v10 adds response-aware trigger evidence to the v9 pipeline.

- The program identifies only the first student turn immediately after each teacher turn.
- Stage 1 receives that response as explicit causal evidence and must not use later turns.
- Stages 2 and 3 receive the same immediate response evidence and must not use later turns.
- Brief prompts such as `Yes?` require a substantive immediate continuation to count as `say_more`; explicit invitations that continue an active peer-linked discussion may count as `add_on` even before a response is recorded.
- New questions, rapid-fire checks, instructions, and teacher-led content checks remain separate from APT follow-ups.
- Stage 2 uses one primary code; reasoning questions are reviewed as a high-risk boundary; JSON truncation repair and Stage 3 conflict fallback are retained.

Reports are stored under `coded_discourse/evaluation/v10/`.
