# pg120 rewriter system prompt — report (2026-11-16)

No system prompt makes pg120 rewrites robustly "Human Written" on real Pangram 4. Best: `prompts/p11_rules_fid.txt` @ temp 0.85
(emulator pass 18% -> 29% pooled opt+val), but on real Pangram it is within noise of no prompt (val+final 9/38 vs 11/38).

- Real Pangram, val (20, n=1): p00_empty@0.7 7/20 Human; p11@0.85 6/20.
- Real Pangram, final (6 tasks, n=3): p00_empty 4/18; p11 3/18. No task passed on all 3 samples.
- Temperature is the main detector lever (temp 1.0: emu<50 65-80%) but fidelity collapses (task ~4.8-5.1). 0.85 is the usable middle.
- "Keep every fact, add nothing" raises task mean ~6.3 -> ~7 at 0.7. Asking for rough/typed register drops facts.
- Never pass: short literary stories (bakery/library drafts), drafts >210 words, simile-heavy reviews, corporate emails.
  Technical Q&A fools the emulator but not real Pangram.
- Emulator agreed with real Pangram on 36/40 val texts.
- Next lever: best-of-N with a detector check (done in rewriter_app auto mode), or more GRPO on stories/long texts with a stronger fidelity reward.
- Spend: OpenRouter $9.25/12, Pangram $10.45/12. Full log: NOTES.md; all texts: runs/*.json.
