# ABCD binary decision: Jev button-or-reply predictions (2026-10-16)

**Hypothesis.** System 1 is weak at wide next-action choices (ABCD 30 buttons: Jev top-1 43.6%, LTT alpha=0.05 coverage <= 1.4%) but strong at narrow ones, so a binary decision should reach useful guaranteed coverage.

**Pre-registered bar** (bench/experiments/abcd_binary_prereg.md, commit 8507085 at 16:08:06, first Jev call 16:08:10): session-level LTT alpha=0.05 coverage >= 30% with session-averaged held-when-serving >= ~90% in the random view.

**Design.** Decision = every agent step (reply or button press); label `button` / `reply`; decision_type = `button_or_reply_after_<customer|agent|action|none>` (speaker of the previous turn); context = dialogue so far, same format as the 30-button run, current step and later turns excluded. Whole sessions, random order (seed 7), until >= 5,000 decisions: 5,005 decisions (reply 3,631 / button 1,374), 382 sessions (train 292, dev 35, test 55). Official view: calibration = 327 train+dev sessions, test = 55 test sessions (718 rows), a very small test set. Random view: 20 seeded 50/50 session splits (about 191 sessions each side). Methods: fixed 0.8/0.9/0.95, LTT hb (primary) and wsr (opt-in, hashed session order; its guarantee is a conditional, order-dependent bound, see the 30-button report) at alpha 0.02/0.05/0.10, delta 0.1. Evaluation script bench/experiments/abcd.py unchanged (its "last-button" column is here the per-type most-frequent label, i.e. the per-type baseline).

**Labels.** The human agent's logged action in a crowdsourced role-play, not policy-optimal.

**Jev run.** `typesafe/jev-1.13`, 5,005 calls, `--compact` (option text cut to its first 60 chars and context to its last 1,800 chars; this differs from the pre-registered full option text, and I do not know whether the 30-button run used compact), 8 workers, spend $0.1114, median latency 508 ms, 0 errors. Total OpenRouter spend on this key is now about $4.46 of $5.

## Findings

- Bar not met: LTT hb alpha=0.05 serves 4.0% in the random view (6.4% official), against the 30% bar; held when serving is 12/12 where served (served in 12/20 splits). wsr alpha=0.05 serves nothing (0/20; official 0/1). Alpha=0.10: hb and wsr both 6.7% random (20/20 held), 6.4% official.
- All of that coverage comes from one type, `after_none` (the first agent step; 334 of 5,005 decisions, label `reply` in all of them, S1 top-1 100%, majority 100%). The three other types are served in 0/20 splits at every alpha, so the guaranteed coverage is a constant rule's coverage, not evidence of S1 skill.
- Jev top-1 on the total row is 69.1% (random) / 67.7% (official), below the majority baseline 72.6% / 74.1% and the per-type baseline (identical to majority here, 72.6% / 74.1%). Per type, random view: after_action 57.3% vs 78.0%, after_customer 71.0% vs 68.7%, after_agent 67.4% vs 66.9%.
- Fixed 0.9 serves 17.6% (random) / 18.7% (official) with pooled disagreement 22.8% / 20.1%; fixed 0.8 serves 37.9% with 22.8% disagreement. No fixed threshold gives low disagreement; 0.9 disagreement is 22.1% (after_customer), 29.2% (after_agent), 16.7% (after_action) in the random view.
- Versus the 30-button run (LTT alpha=0.05 coverage 0.0-1.4%, top-1 43.6% vs majority 19.7%): the binary task has a higher guaranteed coverage (4.0-6.4%) only through the trivial type, and S1 adds nothing over the majority label (the 30-button run beat the majority baseline by 24 points).
- Caveats: one dataset, one sample of 382 sessions, random splits not independent, compact prompt, "held" is the empirical test rate, not the true risk. I did not test why S1's confidence does not separate reply from button.

## Tables

### View: official (1 split)

| method | coverage | pooled disagreement | held when serving (session avg) | held (rows, not the guaranteed quantity) | served in |
| --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 39.0% | 23.9% | – | – | 1/1 |
| fixed 0.9 | 18.7% | 20.1% | – | – | 1/1 |
| fixed 0.95 | 8.8% | 27.0% | – | – | 1/1 |
| LTT-hb a=0.02 | 0.0% | – | 0/0 | 0/0 | 0/1 |
| LTT-hb a=0.05 | 6.4% | 0.0% | 1/1 | 1/1 | 1/1 |
| LTT-hb a=0.1 | 6.4% | 0.0% | 1/1 | 1/1 | 1/1 |
| LTT-wsr a=0.02 | 0.0% | – | 0/0 | 0/0 | 0/1 |
| LTT-wsr a=0.05 | 0.0% | – | 0/0 | 0/0 | 0/1 |
| LTT-wsr a=0.1 | 6.4% | 0.0% | 1/1 | 1/1 | 1/1 |

Per decision type (test rows pooled over splits; method cells are coverage / pooled disagreement / held (session avg) k of splits served / served in k of splits):

| type | n | S1 top-1 | majority | last-button | dis. at p_max>=0.9 | fixed 0.8 | fixed 0.9 | fixed 0.95 | LTT-hb a=0.02 | LTT-hb a=0.05 | LTT-hb a=0.1 | LTT-wsr a=0.02 | LTT-wsr a=0.05 | LTT-wsr a=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| button_or_reply_after_action | 153 | 53.6% | 80.4% | 80.4% | 10.5% (12.4% of rows) | 37.3% / 28.1% / in 1/1 | 12.4% / 10.5% / in 1/1 | 2.0% / 33.3% / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 |
| button_or_reply_after_agent | 149 | 63.1% | 65.8% | 65.8% | 28.6% (23.5% of rows) | 45.0% / 28.4% / in 1/1 | 23.5% / 28.6% / in 1/1 | 10.7% / 31.2% / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 |
| button_or_reply_after_customer | 370 | 71.4% | 71.6% | 71.6% | 18.8% (21.6% of rows) | 42.2% / 20.5% / in 1/1 | 21.6% / 18.8% / in 1/1 | 11.9% / 25.0% / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 |
| button_or_reply_after_none | 46 | 100.0% | 100.0% | 100.0% | – (0.0% of rows) | 0.0% / – / in 0/1 | 0.0% / – / in 0/1 | 0.0% / – / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 100.0% / 0.0% / held 1/1 / in 1/1 | 100.0% / 0.0% / held 1/1 / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 100.0% / 0.0% / held 1/1 / in 1/1 |
| total | 718 | 67.7% | 74.1% | 74.1% | 20.1% (18.7% of rows) | 39.0% / 23.9% / in 1/1 | 18.7% / 20.1% / in 1/1 | 8.8% / 27.0% / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 6.4% / 0.0% / held 1/1 / in 1/1 | 6.4% / 0.0% / held 1/1 / in 1/1 | 0.0% / – / held 0/0 / in 0/1 | 0.0% / – / held 0/0 / in 0/1 | 6.4% / 0.0% / held 1/1 / in 1/1 |

### View: random (20 splits)

| method | coverage | pooled disagreement | held when serving (session avg) | held (rows, not the guaranteed quantity) | served in |
| --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 37.9% | 22.8% | – | – | 20/20 |
| fixed 0.9 | 17.6% | 22.8% | – | – | 20/20 |
| fixed 0.95 | 8.5% | 24.8% | – | – | 20/20 |
| LTT-hb a=0.02 | 0.0% | – | 0/0 | 0/0 | 0/20 |
| LTT-hb a=0.05 | 4.0% | 0.0% | 12/12 | 12/12 | 12/20 |
| LTT-hb a=0.1 | 6.7% | 0.0% | 20/20 | 20/20 | 20/20 |
| LTT-wsr a=0.02 | 0.0% | – | 0/0 | 0/0 | 0/20 |
| LTT-wsr a=0.05 | 0.0% | – | 0/0 | 0/0 | 0/20 |
| LTT-wsr a=0.1 | 6.7% | 0.0% | 20/20 | 20/20 | 20/20 |

Per decision type (test rows pooled over splits; method cells are coverage / pooled disagreement / held (session avg) k of splits served / served in k of splits):

| type | n | S1 top-1 | majority | last-button | dis. at p_max>=0.9 | fixed 0.8 | fixed 0.9 | fixed 0.95 | LTT-hb a=0.02 | LTT-hb a=0.05 | LTT-hb a=0.1 | LTT-wsr a=0.02 | LTT-wsr a=0.05 | LTT-wsr a=0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| button_or_reply_after_action | 11442 | 57.3% | 78.0% | 78.0% | 16.7% (12.8% of rows) | 34.3% / 24.4% / in 20/20 | 12.8% / 16.7% / in 20/20 | 3.0% / 12.1% / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 |
| button_or_reply_after_agent | 9682 | 67.4% | 66.9% | 66.9% | 29.2% (20.1% of rows) | 45.5% / 24.8% / in 20/20 | 20.1% / 29.2% / in 20/20 | 10.4% / 40.1% / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 |
| button_or_reply_after_customer | 25603 | 71.0% | 68.7% | 68.7% | 22.1% (21.1% of rows) | 41.6% / 21.4% / in 20/20 | 21.1% / 22.1% / in 20/20 | 11.3% / 21.1% / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 |
| button_or_reply_after_none | 3349 | 100.0% | 100.0% | 100.0% | – (0.0% of rows) | 0.0% / – / in 0/20 | 0.0% / – / in 0/20 | 0.0% / – / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 60.0% / 0.0% / held 12/12 / in 12/20 | 100.0% / 0.0% / held 20/20 / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 100.0% / 0.0% / held 20/20 / in 20/20 |
| total | 50076 | 69.1% | 72.6% | 72.6% | 22.8% (17.6% of rows) | 37.9% / 22.8% / in 20/20 | 17.6% / 22.8% / in 20/20 | 8.5% / 24.8% / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 4.0% / 0.0% / held 12/12 / in 12/20 | 6.7% / 0.0% / held 20/20 / in 20/20 | 0.0% / – / held 0/0 / in 0/20 | 0.0% / – / held 0/0 / in 0/20 | 6.7% / 0.0% / held 20/20 / in 20/20 |

Notes

- Held when serving (session avg): test disagreement measured as the mean over served sessions of each session's rate, the quantity LTT bounds; the rows column pools rows instead.
- The total row pools types: LTT guarantees each type separately, so a per-type guarantee does not imply the total holds.
- Label = the human agent's logged button (MTurk role-play), not policy-optimal. Data distributed in the MIT-licensed asappresearch/abcd repo.
- Official view: calibrate on train+dev sessions, test on test sessions, one run; held when serving is 0/1 or 1/1.
- Random view: seeded 50/50 session splits over one dataset, so the splits are not independent trials.
- Each type is calibrated on its own; total pools all types (coverage per split = served / test rows, then averaged).
- Majority = most frequent button over all calibration rows; last-button = most frequent button for the type in calibration. Both learn from calibration rows only.

S1 errors (never succeeded): 0
