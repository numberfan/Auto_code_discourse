# APT Coding Pipeline v7

Stage 1 determines trigger and primary addressee. It recognizes explicit continuation of an active peer-linked discussion, while rejecting unrelated new questions.

Stage 2 assigns one or more codes within the selected Type A or Type B branch. Multiple codes require clearly separate actions; default codes are not appended mechanically.

Stage 3 reviews low-confidence or high-risk results. If Stage 3 fails validation, the implementation preserves the Stage 2 result and marks the turn for review instead of replacing it with an empty prediction.

Reports are stored under `coded_discourse/evaluation/v7/`.
