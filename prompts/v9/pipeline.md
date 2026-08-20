# APT Coding Pipeline v9

v9 keeps v8's single-primary-code and recovery behavior, while adding an explicit Stage 1 APT eligibility gate.

- Stage 1 first separates student-linked follow-up from teacher-led instruction, management, rapid-fire checking, and new recall questions.
- Stage 2 assigns exactly one primary code.
- Stage 3 rechecks trigger eligibility before reviewing high-risk or low-confidence results.
- Truncated JSON is repaired only when it can be safely closed; valid multi-code responses are reduced to one primary code and marked low confidence.

Reports are stored under `coded_discourse/evaluation/v9/`.
