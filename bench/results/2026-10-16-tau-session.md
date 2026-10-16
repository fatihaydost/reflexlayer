# tau-bench: session-level vs row-level Learn-then-Test (2026-10-16)

- Same data and cached S1 predictions as `2026-10-16-tau.md` (no new S1 calls): 14,866 decisions, 154 tasks
  (43 airline, 111 retail), splits by task (50 overlapping splits), delta = 0.1.
- Tasks act as sessions. **LTT session** = the new calibrator (`ltt_threshold`): per threshold, each task's
  disagreement rate counts once; Hoeffding-Bentkus test on the task-averaged rate; >= 30 tasks required.
  **LTT row** = the previous row-level test, reproduced by giving every decision its own session.
- Both use the new grid 1.00..0.50 (the 1.00 rung is new), so LTT row differs from the earlier table:
  airline alpha = 0.05 was 0/21 held when serving there, 30/49 here.
- Two "held" columns: *held when serving* measures test disagreement over pooled rows; *held, task-avg*
  measures the task-averaged rate, which is what LTT session promises.
- Reproduce: `uv run python bench/evaluate.py --pred jev=bench/data/pred-jev.jsonl --pred laya=bench/data/pred-laya.jsonl --json bench/results/2026-10-16-tau-session.json`

## Findings

- Airline: LTT session never serves at any alpha. A calibration half has 21 tasks, below both the 30-session
  floor and the ~45 error-free sessions the bound needs at alpha = 0.05. This is the honest answer for 43 tasks.
- Retail, alpha = 0.1: LTT session serves in every split (threshold mean 0.98, 19.9% coverage, 2.7% pooled
  disagreement) and held 50/50 on both measures. LTT row serves more (42.5% coverage) at 8.6% pooled disagreement
  and held 42/50.
- Retail, alpha = 0.05: LTT session serves in 1 of 50 splits (0.2% coverage); LTT row 25.9% coverage, held 45/50
  pooled. alpha <= 0.05 needs more sessions than tau-bench has.
- Session averaging dilutes one bad long session: on airline, LTT row at alpha = 0.05 held 30/49 on pooled rows but
  48/49 task-averaged, because airline:44 (10 of the 15 high-confidence errors in the first run) counts once.
  The guarantee is per typical session, not per decision.
- Laya: no method serves (unchanged).

### jev · tau_airline_next_action

2565 decisions, 43 tasks, top-1 agreement with teacher label 64.0%, majority baseline ('respond') 46.2%, p_max p10/p50/p90 0.50/0.80/0.99, ≥0.50: 90%

| method | threshold | no threshold | served in | coverage | pooled disagreement | p10–p90 | held when serving | held, task-avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 0.80 | 0.0% | 100.0% | 51.8% | 21.2% | 17.9%–23.8% | – | – |
| fixed 0.9 | 0.90 | 0.0% | 100.0% | 33.5% | 15.6% | 12.0%–18.7% | – | – |
| fixed 0.95 | 0.95 | 0.0% | 100.0% | 21.7% | 8.8% | 5.7%–11.4% | – | – |
| LTT session a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT session a=0.05 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.05 | 0.99 | 2.0% | 98.0% | 8.8% | 4.5% | 0.0%–7.6% | 30/49 | 48/49 |
| LTT session a=0.1 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.1 | 0.98 | 0.0% | 100.0% | 13.4% | 9.1% | 0.0%–13.7% | 37/50 | 45/50 |

### jev · tau_retail_next_action

12301 decisions, 111 tasks, top-1 agreement with teacher label 74.5%, majority baseline ('respond') 45.5%, p_max p10/p50/p90 0.52/0.81/1.00, ≥0.50: 93%

| method | threshold | no threshold | served in | coverage | pooled disagreement | p10–p90 | held when serving | held, task-avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 0.80 | 0.0% | 100.0% | 52.0% | 11.5% | 10.8%–12.2% | – | – |
| fixed 0.9 | 0.90 | 0.0% | 100.0% | 35.0% | 6.6% | 6.0%–7.3% | – | – |
| fixed 0.95 | 0.95 | 0.0% | 100.0% | 26.9% | 3.7% | 2.9%–4.4% | – | – |
| LTT session a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.02 | 1.00 | 50.0% | 50.0% | 6.3% | 1.8% | 1.3%–2.4% | 22/25 | 22/25 |
| LTT session a=0.05 | 1.00 | 98.0% | 2.0% | 0.2% | 1.8% | 1.8%–1.8% | 1/1 | 1/1 |
| LTT row a=0.05 | 0.95 | 0.0% | 100.0% | 25.9% | 3.6% | 2.0%–5.0% | 45/50 | 49/50 |
| LTT session a=0.1 | 0.98 | 0.0% | 100.0% | 19.9% | 2.7% | 0.3%–4.1% | 50/50 | 50/50 |
| LTT row a=0.1 | 0.86 | 0.0% | 100.0% | 42.5% | 8.6% | 7.1%–10.1% | 42/50 | 42/50 |

### laya · tau_airline_next_action

2565 decisions, 43 tasks, top-1 agreement with teacher label 8.7%, majority baseline ('respond') 46.2%, p_max p10/p50/p90 0.63/1.00/1.00, ≥0.50: 95%

| method | threshold | no threshold | served in | coverage | pooled disagreement | p10–p90 | held when serving | held, task-avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 0.80 | 0.0% | 100.0% | 82.0% | 91.0% | 89.1%–92.8% | – | – |
| fixed 0.9 | 0.90 | 0.0% | 100.0% | 74.9% | 90.6% | 88.5%–92.4% | – | – |
| fixed 0.95 | 0.95 | 0.0% | 100.0% | 68.0% | 90.2% | 87.8%–92.1% | – | – |
| LTT session a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT session a=0.05 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.05 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT session a=0.1 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.1 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |

### laya · tau_retail_next_action

12301 decisions, 111 tasks, top-1 agreement with teacher label 10.4%, majority baseline ('respond') 45.5%, p_max p10/p50/p90 0.59/1.00/1.00, ≥0.50: 95%

| method | threshold | no threshold | served in | coverage | pooled disagreement | p10–p90 | held when serving | held, task-avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed 0.8 | 0.80 | 0.0% | 100.0% | 79.7% | 89.8% | 89.3%–90.4% | – | – |
| fixed 0.9 | 0.90 | 0.0% | 100.0% | 73.9% | 89.8% | 89.2%–90.4% | – | – |
| fixed 0.95 | 0.95 | 0.0% | 100.0% | 69.3% | 89.8% | 89.2%–90.4% | – | – |
| LTT session a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.02 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT session a=0.05 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.05 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT session a=0.1 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |
| LTT row a=0.1 | – | 100.0% | 0.0% | 0.0% | – | – | 0/0 | 0/0 |

Notes

- LTT bounds the population disagreement with probability >= 1-δ over calibration draws (δ = 0.1). It does not promise every test half is <= α: a threshold whose true risk sits just under α will exceed α on a finite test half fairly often.
- LTT session treats each task as one session: it bounds the task-averaged disagreement with a Hoeffding-Bentkus test over >= 30 tasks. LTT row treats every decision as its own session (the previous row-level test); rows within a task are correlated, so its guarantee does not hold here.
- Threshold columns average splits where a threshold was found; disagreement figures (pooled, p10–p90, held when serving) use serving splits only; coverage averages all splits (no threshold counts as 0).
- held when serving = held/serving: among splits with coverage > 0, how many had test disagreement <= α (fixed thresholds have no α). Pooled disagreement = sum of disagreements / sum of served rows over splits.
- held, task-avg = the same count with test disagreement measured as the mean over served tasks of each task's disagreement rate, the quantity LTT session bounds.
- Labels are the teacher's next action on verified-successful runs, not gold.
- Splits are by task: 50 overlapping splits over one dataset, so shares are not independent trials.
- The threshold grid is 1.00..0.50.

S1 errors (never succeeded): jev 0, laya 0
