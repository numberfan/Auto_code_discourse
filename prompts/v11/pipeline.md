# APT Coding Pipeline v11

v11 keeps the APT same-student / other-student split, adds dual-evidence trigger reasoning, and upgrades review to risk-driven arbitration.

- Stage 1 uses two evidence sources: the teacher turn itself and the immediate student response evidence.
- Immediate response evidence is supporting evidence, not a hard gate for an explicit peer-linked invitation.
- Stage 1 records structured trigger basis, addressee basis, evidence mode, and invitation status.
- Stage 2 still returns one primary code within the same-student or other-student branch.
- Stage 3 reviews cases flagged by data-driven risk patterns, especially trigger false negatives, add_on boundaries, and revoice/challenge/reasoning confusions.
- Prediction outputs store `risk_flags` and `basis` for downstream error analysis.

Reports are stored under `coded_discourse/evaluation/v11/`.
