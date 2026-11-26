# Round 6 — rejected calibration candidate

The working round-4 prompt remains unchanged. Shinka evaluated a baseline and one bounded mutation on 622 learning cases, including the disclosed round-5 test. Training improved by one case (591/622 → 592/622), but reused source-group-disjoint calibration dev regressed from 104/112 to 101/112. The unchanged improvement gate rejected the candidate. No new teacher calls or independent test were run.

These are adaptive calibration results, not independent quality estimates. The retained prompt’s latest independent result remains 136/142 (95.8%), preceded by 133/142 (93.7%) in the prior round. Repeatability on eight new short cases was 7/8 across three calls. Stable Pangram-level equivalence is not demonstrated.

The next useful experiment requires additional teacher-labeled hard boundary cases and an untouched final evaluation. The pending request is to raise only Pangram’s cumulative limit from $100 to $120. Existing limits have not changed. Do not launch another teacher batch based on elapsed time or automatic goal continuations.

Artifacts: protocol.json, train_change_diagnostic.json, validation.json, selection.json. Raw generation/evaluation records: ../runs/round6/. Public active runtime and report remain in the parent folder.
