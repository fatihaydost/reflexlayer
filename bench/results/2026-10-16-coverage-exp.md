# tau-bench: coverage levers for session-level LTT (2026-10-16)

- Same data and cached Jev predictions as `2026-10-16-tau-session.md` (no new S1 calls): retail 12,301 decisions /
  111 tasks, airline 2,565 / 43. Tasks are sessions; 50 task splits (half calibration, half test), delta = 0.1.
- Reproduce: `uv run python bench/experiments/coverage.py --json bench/results/2026-10-16-coverage-exp.json`

## Levers

Fixed from domain knowledge before evaluation (not verifiable from git history: code and results share one commit, 344371b; nothing tuned on test halves):

- **rule**: if S1 picks a write tool and the last turn is not a `[user]` turn matching
  `\b(yes|yeah|yep|confirm|confirmed|go ahead|proceed|sure|correct|that's right)\b` (case-insensitive), S1 does not
  serve that row (it counts as not covered, in calibration and test). Write tools: retail `cancel_pending_order`,
  `exchange_delivered_order_items`, `modify_pending_order_address`, `modify_pending_order_items`,
  `modify_pending_order_payment`, `modify_user_address`, `return_delivered_order_items` (all whose description asks for
  explicit yes/no confirmation); airline `book_reservation`, `cancel_reservation`, `send_certificate`,
  `update_reservation_baggages`, `update_reservation_flights`, `update_reservation_passengers` (DB-changing names; the
  airline descriptions do not mention confirmation).
- **types**: each domain's type split by the kind of the last turn (after `[user]`, after `[tool result]`, other).
  Each sub-type gets its own LTT threshold, calibrated at delta, so the domain-level failure probability can reach k*delta for k sub-types; coverage is pooled over all test rows of the domain. No decision ends
  after `[agent called]` (a tool result always follows), so there are two sub-types; every one has all tasks.
- **wsr**: `ltt_threshold(..., bound="wsr")`: Waudby-Smith & Ramdas betting p-value on session-averaged disagreement,
  predictable plug-in bet with fixed-n tuning, truncation c = 0.75 (their experiments' value), sessions fed in sorted-id order at the time (calibrate.py now orders them by a keyed hash; the WSR numbers below predate that change and are order-dependent; a review saw WSR thresholds 0.94-0.98 over 20 hashed orders vs 0.95 in id order on ABCD pull-up-account, and no id trend on tau retail).

Exploratory (added **after** seeing that the rule blocked few errors; not fixed in advance):

- **nowrite**: S1 never serves a write tool; every write goes to the LLM.

## Results

| config | domain | α | coverage | pooled disagreement | held when serving | held, task-avg | served in | per sub-type coverage (splits with threshold) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| baseline | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| rule | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| rule | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| types | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| types | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| wsr | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| wsr | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| rule+types | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| rule+types | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| rule+wsr | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| rule+wsr | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| types+wsr | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| types+wsr | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| rule+types+wsr | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| rule+types+wsr | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| nowrite | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| nowrite | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| nowrite+types | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| nowrite+types | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| nowrite+types+wsr | airline | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| nowrite+types+wsr | airline | 0.1 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| baseline | retail | 0.05 | 0.2% | 1.8% | 1/1 | 1/1 | 1/50 |  |
| baseline | retail | 0.1 | 19.9% | 2.7% | 50/50 | 50/50 | 50/50 |  |
| rule | retail | 0.05 | 0.2% | 1.8% | 1/1 | 1/1 | 1/50 |  |
| rule | retail | 0.1 | 19.9% | 2.6% | 50/50 | 50/50 | 50/50 |  |
| types | retail | 0.05 | 0.2% | 2.0% | 1/1 | 1/1 | 1/50 | tool 0.0% (0), user 0.5% (1) |
| types | retail | 0.1 | 21.4% | 2.4% | 50/50 | 50/50 | 50/50 | tool 11.5% (50), user 33.0% (50) |
| wsr | retail | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| wsr | retail | 0.1 | 29.0% | 4.6% | 50/50 | 50/50 | 50/50 |  |
| rule+types | retail | 0.05 | 0.2% | 2.0% | 1/1 | 1/1 | 1/50 | tool 0.0% (0), user 0.5% (1) |
| rule+types | retail | 0.1 | 20.6% | 2.4% | 50/50 | 50/50 | 50/50 | tool 9.9% (50), user 33.3% (50) |
| rule+wsr | retail | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 |  |
| rule+wsr | retail | 0.1 | 29.1% | 4.5% | 50/50 | 50/50 | 50/50 |  |
| types+wsr | retail | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| types+wsr | retail | 0.1 | 27.4% | 4.1% | 50/50 | 50/50 | 50/50 | tool 13.8% (50), user 43.4% (50) |
| rule+types+wsr | retail | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| rule+types+wsr | retail | 0.1 | 27.0% | 4.1% | 50/50 | 50/50 | 50/50 | tool 12.7% (50), user 43.9% (50) |
| nowrite | retail | 0.05 | 2.2% | 2.1% | 11/11 | 11/11 | 11/50 |  |
| nowrite | retail | 0.1 | 22.0% | 2.6% | 50/50 | 50/50 | 50/50 |  |
| nowrite+types | retail | 0.05 | 1.7% | 2.4% | 11/11 | 11/11 | 11/50 | tool 0.0% (0), user 3.7% (11) |
| nowrite+types | retail | 0.1 | 19.3% | 2.5% | 50/50 | 50/50 | 50/50 | tool 9.9% (50), user 30.3% (50) |
| nowrite+types+wsr | retail | 0.05 | 0.0% | – | 0/0 | 0/0 | 0/50 | tool 0.0% (0), user 0.0% (0) |
| nowrite+types+wsr | retail | 0.1 | 25.0% | 4.3% | 50/50 | 50/50 | 50/50 | tool 12.7% (50), user 39.5% (50) |

- tau_airline_next_action: nowrite blocks 313 rows (167 agreed), removes 24 of 131 disagreements at p_max >= 0.9.
- tau_airline_next_action: rule blocks 185 of 2565 rows (69 of them agreed with the teacher); removes 18 of 131 disagreements at p_max >= 0.9. Tasks per last-turn kind: {'tool': 42, 'user': 43}; rows: {'tool': 1252, 'user': 1313}.
- tau_retail_next_action: nowrite blocks 2141 rows (1138 agreed), removes 158 of 291 disagreements at p_max >= 0.9.
- tau_retail_next_action: rule blocks 880 of 12301 rows (181 of them agreed with the teacher); removes 44 of 291 disagreements at p_max >= 0.9. Tasks per last-turn kind: {'tool': 111, 'user': 111}; rows: {'tool': 6645, 'user': 5656}.

## Findings

- **The goal (retail, α = 0.05, 30–40 % coverage) is not met by any configuration.** Best: nowrite at 2.2 % coverage,
  serving in 11/50 splits, held 11/11. Every fixed-in-advance config stays at 0.0–0.2 %.
- Why α = 0.05 fails: with ~55 calibration sessions, Hoeffding-Bentkus passes only if the session-averaged
  disagreement is ≤ 0.2 % (essentially zero errors). On the full retail data (diagnostic, not evaluation) the
  session-averaged disagreement at threshold 1.00 is 0.7 % (0.6 % with nowrite); it is sample-limited at the top rung.
- It is also S1-limited: with nowrite, threshold 0.85 gives 34 % coverage but 6.1 % session-averaged disagreement, so
  30–40 % at α = 0.05 is unreachable with this S1 at any calibration size. Threshold 0.90 (26.8 %, 4.3 %) and 0.95
  (20.0 %, 1.5 %) are under α but need ~440 (WSR) and ~90 (WSR) / ~140 (HB) calibration sessions.
- **The rule as specified barely moves anything** (retail blocks 880 rows, removes 44 of 291 high-confidence
  disagreements): in 114 of the 158 write-vs-respond errors the user's last turn already contains "yes" or similar,
  but answering a choice the agent offered, not confirming the agent's summary. nowrite removes all 158 and is the
  only lever that makes α = 0.05 serve at all.
- **WSR is the strongest lever at α = 0.10** (19.9 % → 29.0 % coverage, held 50/50, pooled 4.6 %), because session rates
  have far less variance than the Bernoulli worst case HB assumes. At α = 0.05 it never serves: with zero losses the
  capped bet needs ~60 sessions versus ~45 for HB, and calibration halves have ~55.
- **types** helps a little at α = 0.10 (21.4 %): after-user decisions serve at 33 %, after-tool at 11.5 %. Combining
  types with WSR does not beat WSR alone (27.4 %): each sub-type has the same ~55 sessions but fewer rows per session,
  so session rates get noisier.
- Airline never serves under any lever: 21–22 calibration tasks is below the 30-session floor.

## Caveats

- 50 overlapping splits over one dataset; held counts are not independent trials.
- nowrite was chosen after seeing the rule's result; treat its numbers as exploratory, to be confirmed on new data.
- The WSR truncation c = 0.75 and the session order (by id) were fixed a priori (the order is now a keyed hash, see above); other choices trade the zero-loss
  floor against variance adaptivity and were not tried.
- Labels are the teacher's next action on successful runs, not gold.
