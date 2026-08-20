# APT Coding Pipeline v6

For each teacher turn:

1. Stage 1 determines whether the turn responds to a specific student contribution and identifies the primary addressee. It also returns structured confidence.
2. If `trigger=no`, the final code is `[]` and no further model call is made.
3. If `trigger=yes`, Stage 2 uses the selected Type A or Type B prompt and returns a `codes` array. Multiple distinct codes are allowed within the selected type.
4. Stage 3 runs only when Stage 1 or Stage 2 reports `confidence=low`, or when Stage 2 returns `add_on`, `revoice`, or `challenge`.
5. Stage 3 returns `final_codes`; codes must be legal for the final addressee type.

Output and reports are tagged with `prompt_version=v6`. Evaluation and error analysis are saved together under `coded_discourse/evaluation/v6/`.