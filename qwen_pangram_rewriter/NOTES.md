# Rewriter system-prompt search — running log

Model pg120, temp 0.7 unless noted. pass_emu = emu<50 AND task>=6 AND length_ok.
Cost: ~$0.012/text on OpenRouter (emulator + task check) -> screens use n=1 on opt (40 texts, ~$0.5).

## p00_empty (baseline, opt n=2, 80 texts)
pass 21% | emu human 24% | emu mean 61.5 | task 6.26 | len ok 100% | both samples pass on only 2/40 tasks
- by target length: <150w pass 27%, 150-210 21%, 210+ 12%.
- Passing texts: first-person anecdotes/reviews, plain run-on "And ..." chains.
- Failing (88-92): expository/report drafts (history, school projects, argument essays) — model keeps the
  neat topic-sentence + enumerations + "Overall, ..." conclusion structure.
- Task check: mostly 6, 14/80 below 6 (drops / small inventions, spoken fillers).

## Screens, opt n=1 (40 texts each; SE of a pass rate ~7pp)
| prompt | pass | emu<50 | emu mean | task | len |
|---|---|---|---|---|---|
| p01_plain (short: keep facts, plain voice, varied sentences, no filler/summary) | 30% | 30% | 59.1 | 7.18 | 100% |
| p02_rules (p01 + explicit AI-tell list, anti fake-casual, small imperfections ok) | 32.5% | 32.5% | 54.1 | 6.93 | 100% |
| p03_persona ("you lived this, retyping quickly for a forum") | 17.5% | 27.5% | 54.5 | 6.70 | 100% |
- Any system prompt that stresses "keep every fact / add nothing" raises task mean ~6.3 -> ~7.
- p03 persona pushes the model's spoken/transcript habit ("I think this person is a baker"): inventions, task<6 on 8/40.
- Hard on every prompt: literary fiction drafts (bakery/library-book stories), expository explainers
  (attendance registers, rain gauge, siphon), corporate email/notice, 200w+ reviews with heavy imagery.
- p02 failure read: the model paraphrases line by line and keeps the draft skeleton; collapses paragraphs into
  one block; on fiction it narrates as a spoken transcript.

## Emulator calibration vs REAL Pangram (12 existing texts, $1.49)
emu <=25: 4/4 Human | emu 62-70: 1/4 Human (2 "AI Detected" mixed) | emu >=76: 0/4 Human.
-> emulator is a usable proxy; <50 threshold is about right. Also prior GRPO cache: every bakery-story rewrite
was AI on real Pangram; >300w texts nearly always AI.

| p04_retell ("write from memory, reorder/merge", genre notes, keep paragraphs) | 15% | 20% | 64.4 | 6.78 | 100% |
| p05_short (compact version of p04) | 22.5% | 27.5% | 57.1 | 6.20 | 100% |
- Retelling made output *cleaner and more grammatical* -> worse. Asking for paragraphs hurt: multi-paragraph
  outputs avg emu 77 vs 58 single-block (p04); same direction in p00/p02.
- p02 vs p04 on same tasks: the low-emu (12-18) versions are terse and rough: dropped subjects/articles
  ("float drops, valve opens"), comma splices, missing apostrophes, sometimes all lowercase. Clean grammatical
  versions of the same content score 62-66. Next: push the "typed fast, unedited" register (p06 moderate, p07 heavy).

| p06_typed (terse, fragments, dropped subjects, 1-2 blocks) | 20% | 35% | 53.3 | 6.08 | 100% |
| p07_sloppy (p06 + lowercase, missing apostrophes) | 30% | 32.5% | 54.4 | 5.83 | 100% |
- Rough register lowers emu mean a bit but compresses (words/target ~0.8) and drops facts: task<6 on 9-11/40.
  Among emu-human texts, p06 fails task on 6/14. Net pass no better than p01/p02.
- Across all 7 variants ~15 tasks never go below emu 55: the six literary bakery/library stories, the
  keyboard reviews with heavy similes, the remote-work essays, attendance-register explainers, classroom
  report, Pro-plan email. Common factor: the draft's crafted imagery/punchy structure survives the rewrite.
  Next (p08): flatten figurative language into literal statements, forbid compression, keep 1-2 blocks.

| p08_flat (flatten similes/metaphors into literal statements, don't compress) | 22.5% | 32.5% | 55.3 | 6.90 | 100% |
- p08 read: the model largely ignores "flatten imagery" and copies the draft's phrasing nearly verbatim on
  stories ("ghost pale on the spine", "the ovens are breathing") -> 84-88.
| p02 @ temp 1.0 | 35% | **80%** | **26.7** | 4.75 | 92.5% |
- Temperature is by far the biggest detector lever: emu-human 32.5% -> 80%. But fidelity collapses: task<6
  on 23/40 — meta comments about the draft ("I think there is a typo in the draft"), garbled clauses,
  small inventions, overshoot (ratio 1.1-1.2, 3 length fails). Plan: find the middle temperature and/or
  add fidelity rules that hold up under higher temperature.

| p02 @ temp 0.85 | 35% | 42.5% | 46.0 | 6.15 | 100% |
| p09_nocopy (never reuse 4+ word runs from the draft) @0.7 | 30% | 35% | 54.3 | 6.65 | 100% |
- 0.85 is a reasonable middle: emu-human +10pp vs 0.7, task -0.8. p09's no-copy rule did not move the hard
  tasks (stories still 66-90). Prompt wording at temp 0.7 seems to plateau at ~30-35% pass.
- p10_faithful: drop roughness instructions (let temperature provide the irregularity), make fidelity rules
  explicit against the specific high-temp failures (meta comments, guesses, POV/tense drift, garbling, padding).

| p10_faithful (fidelity-first, no roughness asks) @1.0 | 30% | 65% | 34.3 | 5.10 | 92.5% |
| p01_plain @0.85 | 22.5% | 40% | 47.2 | 5.83 | 100% |
| p01_plain @1.0 | 25% | 57.5% | 36.2 | 4.93 | 97.5% |
- Explicit anti-drift rules don't survive temp 1.0 (25/40 task<6). The task 4-5s at high temp are real slips
  (reordered causality, invented "all of us", dropped "Staff"), not checker harshness.
- p01 loses fidelity faster than p02 when temperature rises; p02's longer rule list seems to anchor it.

| p02 @0.85, n=2 (80 texts) | 32.5% | 43.8% | 48.0 | 6.14 | 100% |
| **p11_rules_fid** (p02 + "never comment on the draft / don't guess / same narrator, tense, audience / don't pad") @0.85 | **40%** | 52.5% | 44.0 | 6.10 | 100% |
- p02@0.85 at n=2: both samples pass on 8/40 tasks, at least one on 18/40 -> sample-to-sample variance is large.
- p11 is the best screen but only +7.5pp vs p02@0.85, i.e. within noise. Moving to real Pangram on val:
  baseline, p11@0.85, and one low-temp candidate.

## Real Pangram on val (n=1, 20 texts)
| prompt | temp | Pangram Human | pass_pangram | emu<50 | task |
|---|---|---|---|---|---|
| p00_empty | 0.7 | 35% (7/20) | 30% | 25% | 6.25 |
| p11_rules_fid | 0.85 | 30% (6/20) | 25% | 30% | 5.95 |
- No improvement on val: the opt gain did not transfer (differences = 1 text, pure noise).
- Emulator vs Pangram agreement 36/40 -> emulator is a reliable proxy here.
- val is 6/20 bakery/library stories + 4/20 keyboard reviews; all stories fail under both prompts.
- Category view over all opt runs: stories pass 0/~40 at every prompt and temperature (even temp 1.0 gets emu-human
  only 17-33% and task collapses to ~4). Non-story/non-keyboard tasks: p11@0.85 50% pass vs p02@0.7 43%.
- Next: story-specific rule (p12) screened on a 6-story subset of opt (data/optstory.jsonl, n=2).

| p11 @0.85 on optstory (6 stories, n=2) | 0/12 pass | emu-human 0/12 | emu 71.9 | task 6.5 |
| p12_story (p11 + "retell stories plainly, literal, no imagery") @0.85 on optstory | 0/12 | 0/12 | 70.7 | 5.67 |
- p12 partly obeys (chained, flatter narration) but still 62-72 and loses fidelity. Conclusion: literary-fiction
  rewrites are out of reach for system-prompting this model; dropped p12.
- Next: val n=3 on the emulator only (cheap) for p00, p11@0.85, p02@0.7 to get 60 texts per prompt.

## val n=3 on the emulator (60 texts each)
| prompt | temp | pass | emu<50 | emu mean | task | task<6 |
|---|---|---|---|---|---|---|
| p00_empty | 0.7 | 13.3% | 16.7% | 66.1 | 6.40 | 8 |
| p02_rules | 0.7 | 20.0% | 20.0% | 63.3 | **7.28** | 6 |
| p11_rules_fid | 0.85 | **21.7%** | **28.3%** | 53.1 | 6.20 | 13 |
- Pooled opt+val pass: p00 18% (25/140), p02@0.7 25% (25/100), p11@0.85 29% (29/100).
- Winner on the detector objective: p11@0.85. p02@0.7 is the fidelity-safe alternative (higher task, fewer drift
  errors, smaller detector gain). Sample-to-sample consistency is poor for all (both/all samples pass on 1-2/20 val tasks).

## Final (6 held-out rewrites, n=3, REAL Pangram)
| prompt | temp | Pangram Human | pass_pangram | emu<50 | task |
|---|---|---|---|---|---|
| p00_empty | 0.7 | 22% (4/18) | 22% | 28% | 6.17 |
| p11_rules_fid | 0.85 | 17% (3/18) | 17% | 28% | 6.56 |
- 3/6 final tasks are bakery stories: 0/18 Human across both prompts.
- pandas question: emu 22-40 but Pangram AI on 5/6 -> emulator blind spot for technical Q&A.
- Real Pangram pooled val+final: p00 11/38 (29%), p11@0.85 9/38 (24%). No measurable real-detector gain.
Stopped here: OpenRouter $9.25/12, Pangram $10.45/12.
