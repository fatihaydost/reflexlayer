# Reflex

Reflex is a calibration and guard layer that lets a fast "System 1" (S1) model answer an agent's finite-choice
decision (which tool, which team, continue or stop) only when a session-level Learn-then-Test (LTT) bound says
its disagreement with your LLM stays within a budget alpha. Otherwise the LLM decides. Reflex does not make S1
better; it tells you, with a stated error budget, when you may trust it.

Research project, not maintained as a product. The code works and is tested (`uv run pytest -q -W error`), but
the benchmark results below are mostly negative and there is no roadmap. MIT licensed.

## What we found

Numbers are from `bench/results/` (all runs 2026-10-16); the full table is in [RESULTS.md](RESULTS.md).
Labels are a teacher agent's or a human's logged action, not gold, so "disagreement" means disagreement with that label.

- **Hand-picked thresholds are unsafe.** Trusting S1 when p_max >= 0.9 gave 15.6% disagreement on tau-bench airline
  and 6.6% on retail with Jev, and 26.5% on ABCD 30-button, 22.8% on ABCD binary
  ([tau](bench/results/2026-10-16-tau.md), [abcd](bench/results/2026-10-16-abcd.md),
  [abcd-binary](bench/results/2026-10-16-abcd-binary.md)). Many high-confidence retail errors were write actions: of
  291 disagreements at p_max >= 0.9, 158 were S1 choosing return/cancel/modify/exchange.
- **Row-level LTT is not valid when decisions cluster in sessions.** On tau-bench airline, one task (airline:44)
  carried 10 of 15 high-confidence disagreements; on the original grid row-level LTT held in 0/21 serving splits
  (final grid: 30/49 pooled, 48/49 task-averaged). Session-level calibration removes the independence assumption ([tau](bench/results/2026-10-16-tau.md), [tau-session](bench/results/2026-10-16-tau-session.md)).
- **The price of validity is coverage.** Session-level LTT needs about 45 error-free sessions at alpha = 0.05 and
  never served on airline (43 tasks). On retail at alpha = 0.05 Jev's LTT-served coverage was 0.2%; a policy digest
  in the context lifted it to 11.4% to 16.1%, still short of the 30-40% we hoped for
  ([digest](bench/results/2026-10-16-digest.md), [coverage levers](bench/results/2026-10-16-coverage-exp.md)).
- **Other S1 models did not help.** Laya (local) agreed with the label on 8.7% / 10.4% of airline / retail decisions
  against a 46.2% / 45.5% majority baseline and was never served ([tau](bench/results/2026-10-16-tau.md)). A frozen
  mmBERT-base probe reached 67.9% on retail vs Jev(digest) 75.5% and 0.0% LTT-served coverage in every view
  ([probe](bench/results/2026-10-16-probe.md)).
- **ABCD 30-button:** Jev top-1 was 43.6% vs 19.7% majority, but LTT alpha = 0.05 coverage was 0.0% to 1.4%
  ([abcd](bench/results/2026-10-16-abcd.md)).
- **ABCD binary (button or reply): no S1 beat the majority baseline.** Jev 69.1% vs 72.6% (random view). Its 4.0%
  LTT-served coverage came entirely from one trivial type that is always `reply`. Three LLMs read via first-token
  logprobs (Llama-3.1-8B 59.6%, Gemma-3-27B 65.9%, Qwen3-235B 69.5% on answered rows, vs 73.6% majority on a
  208-decision pilot) failed the pre-registered gate, so no full run was made
  ([abcd-binary](bench/results/2026-10-16-abcd-binary.md), [llm-s1](bench/results/2026-10-16-llm-s1.md)).

## How it works

1. **Shadow** (default): S1 predicts, your LLM decides, both are logged per decision type and session.
2. **Calibrate**: per decision type, walk thresholds from 1.00 down and keep the lowest one whose session-level bound still holds
   (Hoeffding-Bentkus). A WSR betting bound is available as `ltt_threshold(bound='wsr')` with hashed session order and
   is used in the benchmark scripts; it is not exposed by `calibrate()` or the CLI.
3. **Activate** per decision type: S1 answers above the threshold.
4. **Audits and drift fallback**: 5% of served decisions are still sent to the LLM; if audited agreement degrades,
   the type reverts to shadow and its threshold is withdrawn. Types marked destructive are never served by S1.

### Guarantee and its limits

- **The unit is the session.** Decisions inside one conversation are correlated, so Reflex never treats them as
  independent evidence. The budget bounds the *session-averaged* disagreement: over sessions with at least one
  decision above the threshold, the mean of each session's S1-LLM disagreement rate is <= alpha with probability
  >= 1 - delta (Learn-then-Test, fixed sequence over thresholds 1.00..0.50, Hoeffding-Bentkus bound per threshold).
  Each session weighs the same, so one long bad session is diluted by many good ones; it is not a per-decision rate.
- **Pass `session_id`** (SDK argument, gateway header, replay field). Decisions logged without one are ignored by
  `calibrate`; a type with no session ids never gets a threshold.
- **Activation needs many sessions.** At least 30 distinct sessions above the threshold, and even with zero
  disagreement the bound needs n >= ln(1/delta) / -ln(1 - alpha) sessions: about 45 at alpha = 0.05, delta = 0.1,
  114 at alpha = 0.02 (22 at alpha = 0.1, where the 30-session floor binds). With a few disagreeing sessions it
  needs several times more.
- Evidence so far is held counts on overlapping splits of two datasets (the bound behaved as intended on these splits); there is no live validation.
- The budget is meant to hold if the shadow/audit pairs are representative of live traffic (exchangeable).
- Re-running `calibrate` repeatedly on a growing log is a form of peeking that delta does not cover.
- Agreement with the LLM is not correctness: S1 inherits the LLM's mistakes.
- Content is hashed with unsalted sha256 (short inputs can be guessed) and is still sent to the S1 provider.

## Install and quickstart

Python >= 3.12, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fatihaydost/reflexlayer && cd reflexlayer
uv sync
uv run rflx --help        # subcommands: gateway calibrate activate shadow mark-destructive report replay
```

### SDK

```python
from reflexlayer import DecisionLog, OpenAICompatLLM, Router, SystemOneAdapter

router = Router(
    s1=SystemOneAdapter("http://127.0.0.1:8790/v1/systemone", "multilingual"),
    llm=OpenAICompatLLM("https://api.openai.com/v1", "gpt-4.1-mini", api_key="..."),
    log=DecisionLog("reflex.db"),
)
d = router.decide("Which team should handle `message`?", {"billing": "payments, refunds", "tech": "bugs"},
                  {"message": "I was charged twice"}, decision_type="support-triage", session_id="conv-123")
print(d.choice, d.tier)
```

### Gateway

An OpenAI-compatible proxy. Point your client at `http://127.0.0.1:8788/v1`. Requests whose `response_format` is a
single string enum, or `tool_choice: "required"` over argument-free tools, are decisions; everything else passes
through unchanged. Header `x-reflex-decision-type` names the decision type; `x-reflex-session-id` names the
conversation (required for calibration). `x-reflex-*` headers are not forwarded upstream.

```bash
# S1 = a hosted systemone endpoint (key from the environment)
REFLEX_S1_API_KEY=... uv run rflx gateway --upstream https://api.openai.com/v1 \
  --s1 systemone --s1-url https://openrouter.ai/api/v1/systemone --s1-model typesafe/jev-1.13

# S1 = Strands Decider served locally
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
uv run rflx gateway --upstream https://api.openai.com/v1 --s1 strands-decider

# S1 = any logprobs-capable, non-reasoning OpenAI-compatible instruct model (default base URL: OpenRouter)
REFLEX_S1_API_KEY=... uv run rflx gateway --upstream https://api.openai.com/v1 --s1 llm --s1-model <model-id>
```

`--s1 llm` asks the model to answer with one option letter and reads first-token logprobs renormalised over the
option letters; these are not calibrated by themselves (Reflex's LTT does the calibration). The request sets
`provider.require_parameters`, which is OpenRouter-specific and ignored elsewhere. Strands Decider's server has
no authentication, its upstream docs call it local-experiments only, and it reports no cost (S1 cost logs as 0);
the adapter is tested against mocked responses only, not a live server. Non-decision `stream: true` traffic is
passed through but buffered, and gateway traffic records the LLM cost as 0.

### CLI workflow

Global options go before the subcommand: `rflx --db reflex.db calibrate`.

```bash
uv run rflx calibrate --alpha 0.05 --delta 0.1        # threshold per decision type (--type X for one)
uv run rflx report                                    # writes reflex-report.html (--out to change)
uv run rflx activate --type support-triage            # S1 answers above the threshold
uv run rflx shadow --type support-triage              # back to shadow
uv run rflx mark-destructive --type delete-account    # S1 never serves this type
uv run rflx activate --type X --allow-destructive     # lift the block and clear the mark
uv run rflx replay labelled.jsonl --s1 llm --s1-model <id>   # offline: run S1 over labelled rows
```

Replay lines: `{"decision_type", "question", "options", "context", "label", "session_id"}`. Drift reverts a type to
shadow and withdraws its threshold, so recalibrate before activating again.

## Reproducing the benchmarks

`bench/data/` is gitignored and not shipped; you build it from the public sources below. Scripts:
`bench/convert_tau.py`, `bench/convert_abcd.py` (dataset to decision JSONL), `bench/predict.py` (S1 predictions
via `--s1 jev|laya|llm`), `bench/evaluate.py` (tau LTT evaluation), `bench/experiments/{abcd,coverage,probe}.py`.
Each results file in `bench/results/` states the commands and design it used.

- **tau-bench** ([sierra-research/tau-bench](https://github.com/sierra-research/tau-bench), MIT): clone it to
  `bench/data/tau-bench-src`, then `uv run python bench/convert_tau.py` (add `--digest` for the policy-digest variant;
  digests are in `bench/policy_digest_*.txt`).
- **ABCD** ([asappresearch/abcd](https://github.com/asappresearch/abcd), MIT repo; the data ships inside the repo
  and I found no separate data licence, so check it yourself before redistributing): gunzip `data/abcd_v1.1.json.gz`
  to `bench/data/abcd/abcd.json`, then `uv run python bench/convert_abcd.py` (`--binary` for the
  button-or-reply task).
- **Predictions:** `uv run python bench/predict.py --s1 jev --out bench/data/pred-jev.jsonl` (needs
  `REFLEX_S1_API_KEY`; `--budget-usd` caps spend). Costs recorded in the results: Jev on tau-bench 14,866 decisions
  $1.2374 (with digest $1.4215), ABCD 36,482 decisions $1.693, ABCD binary 5,005 decisions $0.1114, the three-LLM pilot
  $0.01344. Laya and the mmBERT probe ran locally ($0; Laya on an RTX 3050).
- The probe needs torch + transformers, which are not part of the uv environment (see the header of
  `bench/experiments/probe.py`).
- Committed in `bench/results/` are aggregate numbers only; per-row S1 predictions stay in the ignored `bench/data/`.

## Third-party notes

- **Jev** (`typesafe/jev-1.13`) is a hosted S1 service from TypeSafe, reached here via OpenRouter, under its own
  terms. Reflex only calls it; it never trains on Jev's outputs (the probe is trained on dataset labels only), and
  per-row Jev outputs are not distributed in this repo.
- **Laya** is a separate local S1 server (`/v1/systemone` schema) used for the local runs; Reflex ships only the
  adapter, not the model.
- **Strands Decider** (AWS, open weights, Apache-2.0) is supported through its local server's schema, read from its
  source at commit `6039cf8`; see the caveat above.
- **tau-bench** and **ABCD** belong to their authors under the licences stated above. No ABCD rows are shipped;
  example decision rows can be regenerated with `bench/convert_abcd.py` from the ABCD repo.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Fatih Aydost.
