# APT Coding Pipeline v8.1

v8.1 keeps v8's high-recall trigger policy and adds structured response evidence and one primary code.

- Stage 1 recognizes direct follow-up and explicit continuation of an active peer-linked discussion.
- Immediate response evidence helps identify the response relation but is not required for an explicit discussion invitation.
- Stage 2 returns one primary code and uses the same immediate response evidence without treating it as a trigger gate.
- Revoice includes echo-confirmation turns; comparison and ordinary factual elaboration are say_more unless the teacher explicitly asks for reason, evidence, or justification.
- Stage 3 reviews low-confidence and high-risk turns and preserves validated Stage 2 results if review fails.

Reports are stored under `coded_discourse/evaluation/v8.1/`.
