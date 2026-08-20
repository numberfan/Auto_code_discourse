# APT Coding Pipeline v8

v8 uses the v5 high-recall trigger policy with v7 structured confidence and multi-code outputs.

- Stage 1 recognizes direct follow-up and explicit continuation of an active peer-linked discussion.
- Stage 2 supports multiple codes but removes residual `say_more` or `add_on` when a more specific code is present.
- Revoice includes echo-confirmation turns; reasoning-effect questions are `press_for_reasoning`.
- Stage 3 reviews high-risk or low-confidence turns.
- If Stage 3 fails, validated Stage 2 results are preserved and marked for review.

Reports are stored under `coded_discourse/evaluation/v8/`.
