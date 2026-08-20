# APT Coding Pipeline v9

v9 keeps v8's single-primary-code and recovery behavior, while adding an explicit Stage 1 APT eligibility gate and a peer-reference rescue path.

- Stage 1 first separates student-linked follow-up from teacher-led instruction, management, rapid-fire checking, and new recall questions.
- Explicit references to a peer answer, such as "what they said" or "respond to what they have said", can restore a Type B follow-up when the discussion is student-linked.
- Stage 2 assigns exactly one primary code.
- Stage 3 rechecks trigger eligibility before reviewing high-risk or low-confidence results.
- Stage 3 filters codes to the final addressee type; if no compatible code remains, the validated Stage 2 result is preserved and marked for review.
- Truncated JSON is repaired only when it can be safely closed; valid multi-code responses are reduced to one primary code and marked low confidence.

Reports are stored under `coded_discourse/evaluation/v9/`.
