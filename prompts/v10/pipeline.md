# APT Coding Pipeline v10

v10 adds response-aware trigger evidence to the v9 pipeline.

- The program identifies only the first student turn immediately after each teacher turn.
- Stage 1 receives that response as explicit causal evidence and must not use later turns.
- Brief prompts such as `Yes?` require a substantive immediate continuation to count as `say_more`.
- New questions, rapid-fire checks, instructions, and teacher-led content checks remain separate from APT follow-ups.
- Stage 2 uses one primary code; JSON truncation repair and Stage 3 conflict fallback are retained.

Reports are stored under `coded_discourse/evaluation/v10/`.
