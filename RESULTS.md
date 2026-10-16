# Benchmark results

One row per benchmark run. All numbers are copied from the linked files in `bench/results/`; the files
hold the full tables, designs, caveats and reproduce commands. All runs are from 2026-10-16.

Columns: **top-1** = S1 top-1 agreement with the label, next to the majority-label baseline. **Fixed 0.9** =
disagreement of the hand-picked "trust p_max >= 0.9" rule (definition differs per file, see the note under each
group). **LTT a=0.05** = coverage of session-level Learn-then-Test (hb unless noted) at alpha = 0.05, delta = 0.1,
and "held" = splits (when it served) in which the test disagreement stayed within alpha. Labels are always the
logged action of a teacher agent or human, not gold. All splits are overlapping splits of one dataset, so
"held" counts are not independent trials.

## tau-bench (sierra-research, MIT): next action, tasks as sessions

| Run | Dataset / S1 | Decisions / sessions | S1 cost | Top-1 vs majority | Fixed 0.9 disagreement | LTT a=0.05 coverage (held) | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tau, airline | Jev | 2,565 / 43 tasks | $1.2374 (airline + retail) | 64.0% vs 46.2% | 15.6% | row-level: 6.0% (0/21); session-level: 0.0% (never served) | [tau](bench/results/2026-10-16-tau.md), [tau-session](bench/results/2026-10-16-tau-session.md) |
| tau, retail | Jev | 12,301 / 111 tasks | (see above) | 74.5% vs 45.5% | 6.6% | row-level: 26.5% (45/50); session-level: 0.2% (1/1) | same |
| tau, airline | Laya (local, `--compact`) | 2,565 / 43 tasks | $0 | 8.7% vs 46.2% | 90.6% | 0.0% (never served) | same |
| tau, retail | Laya (local, `--compact`) | 12,301 / 111 tasks | $0 | 10.4% vs 45.5% | 89.8% | 0.0% (never served) | same |
| tau + policy digest, retail | Jev | 12,301 / 111 tasks | $1.4215 (pilot $0.0052 + full run $1.4163, both domains) | 75.7% (was 74.5%) | 3.0% at p_max >= 0.9 (was 6.7%) | 11.4% (50/50) at 50/50 split; 14.0% (50/50) at 70/30; wsr 70/30: 16.1% (50/50) | [digest](bench/results/2026-10-16-digest.md) |
| tau + policy digest, airline | Jev | 2,565 / 43 tasks | (see above) | 64.2% (was 64.0%) | 19.8% at p_max >= 0.9 (was 15.5%) | no config serves in any view | same |
| tau coverage levers, retail | Jev (cached predictions) | 12,301 / 111 tasks | $0 (no new calls) | as tau | – | baseline 0.2% (1/1); best lever nowrite 2.2% (11/11, exploratory); wsr 0.0% | [coverage-exp](bench/results/2026-10-16-coverage-exp.md) |
| tau coverage levers, airline | Jev (cached predictions) | 2,565 / 43 tasks | $0 | as tau | – | 0.0% under every lever | same |
| tau probe, retail | mmBERT-base frozen + linear head (local) vs Jev(digest) | 12,301 / 111 tasks | $0 | probe 67.9% vs Jev(digest) 75.5% (primary split) | probe 18.9% vs Jev(digest) 3.0% (at p_max >= 0.9) | probe 0.0% in every view; Jev(digest) 11.6% at 20/50/30 (50/50) | [probe](bench/results/2026-10-16-probe.md) |
| tau probe, airline | same | 2,565 / 43 tasks | $0 | probe 48.9% vs Jev(digest) 64.6% | probe 38.5% vs Jev(digest) 19.4% (at p_max >= 0.9) | nothing can serve (10-21 calibration tasks) | same |

Notes: in `tau` the fixed-0.9 figure is pooled disagreement on served decisions over the 50 splits; in `digest` and
`probe` it is disagreement at p_max >= 0.9 on the rows given in those files. The row-level LTT numbers use the
older grid (no 1.00 rung) in `tau` and the new grid in `tau-session` (airline row-level a=0.05: 0/21 held in `tau`,
30/49 in `tau-session`); the session-level numbers are the ones the bound is about. Row-level LTT is not valid when decisions cluster in sessions (the 0/21 is from the original grid). The `probe` row's majority
baseline is not restated in the probe file (full-data baselines are in the `tau` rows).

## ABCD (asappresearch, MIT repo): customer-service dialogues

| Run | Dataset / S1 | Decisions / sessions | S1 cost | Top-1 vs majority | Fixed 0.9 disagreement | LTT a=0.05 coverage (held) | File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ABCD 30 buttons | Jev | 36,482 / 10,042 | $1.693 | 43.6% vs 19.7% (official); 44.7% vs 19.4% (random). Last-button baseline 41.8% / 42.5% | 26.5% (official); 28.7% (random) | hb: 0.4% (1/1) official, 0.2% (16/16) random; wsr: 1.4% (0/1) official, 0.0% random | [abcd](bench/results/2026-10-16-abcd.md) |
| ABCD binary (button or reply) | Jev | 5,005 / 382 | $0.1114 | 69.1% vs 72.6% (random); 67.7% vs 74.1% (official) | 22.8% (random); 20.1% (official) | hb: 4.0% (12/12) random, 6.4% (1/1) official; all of it from one trivial type (first agent step, always `reply`); wsr 0.0% | [abcd-binary](bench/results/2026-10-16-abcd-binary.md) |
| ABCD binary pilot, Llama-3.1-8B | logprob LLM S1 | 208 / 16 | $0.00366 | 59.6% vs 73.6% | – (never reaches p_max >= 0.9; coverage 0.0%) | not run (pre-registered gate failed) | [llm-s1](bench/results/2026-10-16-llm-s1.md) |
| ABCD binary pilot, Gemma-3-27B | logprob LLM S1 | 208 / 16 | $0.00322 | 65.9% vs 73.6% | 28.2% (78.4% of rows) | not run | same |
| ABCD binary pilot, Qwen3-235B-A22B | logprob LLM S1 | 208 / 16 (54 failed calls) | $0.00655 | 69.5% on answered rows vs 73.4%; 51.4% counting failures | 23.2% (45.7% of all rows) | not run | same |

Notes: ABCD "official" = train+dev calibrate, test on the test sessions (one split); "random" = 20 seeded 50/50
session splits. Held counts in the ABCD rows are on session-averaged test disagreement. The three pilot models
were not run on the full set because none passed the pre-registered gate (top-1 above majority and at most 5%
errors); the pilot total was $0.01344.

## Reading the table

- Fixed thresholds on S1 confidence gave double-digit disagreement in most runs, including on Jev.
- Where S1 was good (Jev on tau retail), session-level LTT at alpha = 0.05 still served 0.2% without a policy digest and
  11.4% to 16.1% with one, and the session floor (about 45 error-free sessions at alpha = 0.05) binds before S1
  quality does.
- No S1 tried on ABCD binary (Jev and three small LLMs via logprobs) beat the majority label.
