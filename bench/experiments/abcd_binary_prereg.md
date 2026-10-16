# ABCD binary decision: pre-registration (2026-10-16)

Committed before any Jev call on this data.

**Hypothesis.** System 1 (Jev) is weak at wide next-action choices (ABCD 30 buttons: top-1 43.6%, LTT alpha=0.05 coverage <= 1.4%) but strong at narrow ones. Test on a binary decision.

**Success bar.** Session-level LTT alpha=0.05 coverage >= 30% with session-averaged held-when-serving >= ~90%, in the random view (20 seeded 50/50 session splits).

**Decision.** Every agent step in ABCD (speaker "agent" = reply, "action" = button press). Label `button` if action else `reply`. Options: button = "Press a dashboard button (look up, verify, search, update, etc.) before replying"; reply = "Write a natural-language reply to the customer now". Question: "A customer-service agent follows company guidelines. What do they do next: press a dashboard button, or reply to the customer?"
decision_type = `button_or_reply_after_<customer|agent|action|none>` (speaker of the previous turn). Context = dialogue so far, same format as convert_abcd.py, current step and later turns excluded.

**Sample.** Whole sessions in random order (seed 7) until >= 5,000 decisions (`convert_abcd.py --binary`). No type merging.

**Evaluation.** bench/experiments/abcd.py unchanged: official view (calibration = train+dev sessions in sample, test = test sessions) and 20 random 50/50 session splits; fixed 0.8/0.9/0.95; LTT hb (primary) and wsr (opt-in) at alpha in {0.02, 0.05, 0.10}, delta 0.1. Baselines: majority and per-type most-frequent label from calibration rows.

**Run.** typesafe/jev-1.13 via OpenRouter, 8 workers, cumulative budget cap $0.30.
