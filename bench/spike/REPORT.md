# Reflex benchmark source spike (read-only; data in this dir, converters in ../convert*.py)

## TL;DR
Primary: tau-bench (sierra-research, MIT) `historical_trajectories`, successful runs only, label = the agent's next action. Counted **14,866 decisions** (retail 12,301, airline 2,565) from 165 distinct tasks, converted already to ../tau_bench_decisions.jsonl.
Backup: BFCL v4 `live_multiple` + `multiple`: **1,253** single-label tool-choice items (+372 "none applies" items), but tiny context and not agent-like.
Estimated Jev cost: tau-bench full ~ $1.1; BFCL ~ $0.01.

## 1. tau-bench / tau2-bench (both MIT, Sierra)
**tau-bench** ships `historical_trajectories/` (4 JSON files, 50 MB): gpt-4o and claude-3.5-sonnet-new runs on retail and airline. Each entry = full message list (system policy, user, assistant, tool calls and results) + `reward` (0/1 from DB end-state check) + task. Counts (assistant steps = next-action decisions):
| file | trajectories | tasks | reward=1 | steps (all) | steps (reward=1) |
|---|---|---|---|---|---|
| gpt-4o-airline | 200 | 50 | 84 | 2,454 | 829 |
| gpt-4o-retail | 460 | 115 | 278 | 6,564 | 3,786 |
| sonnet35new-airline | 400 | 50 | 184 | 4,674 | 1,736 |
| sonnet35new-retail | 920 | 115 | 637 | 12,624 | 8,515 |
Total, reward=1: 829+1736=2,565 airline; 3,786+8,515=12,301 retail (=14,866). Telecom is not in tau-bench.
Repo also has gold action sequences (tasks_test.py: retail 115 tasks/582 `Action(`, airline 50 tasks/158), but those are WRITE actions only; no gold for reads/respond.

**tau2-bench** (cloned, 856 MB) ships `data/tau2/results/final/*.json` (26 files, 8-51 MB each): simulations with full `messages` and `reward_info` for claude-3.7-sonnet, gpt-4.1, gpt-4.1-mini, o4-mini, 4 trials each. Domains: airline 50 tasks, retail 114, telecom 2,285 tasks in tasks.json (114 evaluated in results; modes default/workflow/no-user/op). Assistant steps in reward=1 sims (counted per model, domain): airline 1.1-1.4k each (4 models, ~4.9k), retail 4.0-5.3k each (4 models, ~18k), telecom (plain "telecom" files, all modes pooled; claude 3.8k, gpt-4.1 14.8k, 4.1-mini 4.0k, o4-mini 15.5k) plus telecom-workflow 17.7k (gpt-4.1) and 15.6k (o4-mini). So several tens of thousands in total, but only ~280 distinct tasks (telecom 114 evaluated) and heavy duplication across models/trials. tau2 gold `evaluation_criteria.actions`: airline 142, retail 550, telecom 13,215 (write/user-side actions, tau2 telecom has user-side tools too).

**Decision types**: with one type per domain = 2 (tau-bench) or 3 (tau2), 2.5k to 12k decisions per type. Finer: decision_type = domain x previous-event kind, still thousands each. `respond` is 46% of labels in tau-bench-retail data (top: respond 6,784, get_order_details 2,276, get_user_details 1,022, find_user_id_by_name_zip 758, get_product_details 754). Options per decision: 15-17 tools (+respond) fixed per domain (tau2 telecom ~37-40).
**Context size**: raw (policy + history) median 8.4-10 kB, p90 15-18 kB; policy alone ~6.7 kB. To fit <=6000 chars I put tool descriptions in `options`, drop the policy, keep last 5,000 chars of the conversation (tool results truncated to 600 chars). Resulting avg ctx 2.8 kB (max 5 kB), ctx+options avg 6.0 kB retail / 4.3 kB airline. Dropping the policy makes some labels less derivable (e.g. confirmation rules); a ~1 kB policy digest could be added to `question`.
**Label quality**: NOT a human gold. Label = what a strong LLM agent did in an episode that ended with the correct DB state (reward=1), so it is a verified-good-trajectory teacher label. Reads/respond/think choices within it are not unique (several next actions would be fine), so there is label noise of the "acceptable alternative" kind; only write actions are truly constrained. For Reflex, where the guarantee is S1-vs-reference disagreement, this "teacher on successful runs" label is defensible; say so plainly in the post (do not call it gold). Also: `think` tool and the 4-trial/2-model duplication mean decisions are correlated within a task (165 tasks) -> split calibration/test BY TASK, not by row, or LTT guarantees will be optimistic.
**Conversion effort**: tau-bench done (0.5 h incl. this spike, script ../convert.py). tau2 results: 3-4 h (tool schemas from env code, different message schema, telecom user tools, dedupe).

Example line (options and context shortened here; real file has all 17 options):
```
{"decision_type": "tau_retail_next_action", "question": "Given the conversation so far, what should the customer-service agent do next: call one tool or reply to the user?", "options": {"transfer_to_human_agents": "Transfer the user to a human agent, with a summary of the user's issue. Only tra", "think": "Use the tool to think about something. It will not obtain new information or cha", "return_delivered_order_items": "Return some items of a delivered order. The order status will be changed to 'ret", "modify_user_address": "Modify the default address of a user. The agent needs to explain the modificatio", "...": "(17 total)"}, "context": {"conversation_tail": "[user]  Hi, I'd like to check how many t-shirt options you currently have in your online store.\n[agent called] list_all_product_types{}\n[tool result list_all_product_types] {\"Action Camera\": \"3377618313\", \"Air Purifier\": \"3821016478\", \"Backpack\": \"2524789262\", \"Bicycle\": \"9783735446\", \"Bluetooth Spe ..."}, "label": "get_product_details"}
```

## 2. BFCL (gorilla repo, code Apache-2.0; HF dataset card also apache-2.0)
v4 data (bfcl_eval/data, JSONL; format_sensitivity file is not JSONL). Items with >=2 candidate functions: live_multiple 1,053 (gold file present; every gold is a single call), multiple 200, parallel_multiple 200 and live_parallel_multiple 24 (multi-call gold, unusable for single label), live_irrelevance 372 of 884 (gold = no function applies, no answer file) and live_relevance 11/16 (should call something, no name). Single-function sets (simple_*, parallel, irrelevance, multi-turn, memory, web_search) are not choices. Usable: **1,253** (live_multiple+multiple), up to ~1,625 if irrelevance is added with a `none` option. Candidates per item: avg 3.8, median 4 (live_multiple), max 37. Context: median ~600 chars (live: 3.2 kB incl. function docs); `user_request` is one utterance, no history, some non-English (Vietnamese etc). Decision types: one ("which function"), or split live_multiple/multiple; options differ per item, so there is no fixed per-type option set (does not match "per-decision-type" calibration well: pooled type = 1,053 decisions with varying option sets). Label: human/curated ground-truth function name, highest trust. Hours: 0.5 (done, ../convert_bfcl.py). Cost: avg ~610 chars incl options -> total 0.77M chars -> 0.22M tokens -> $0.01.

Example:
```
{"decision_type": "bfcl_live_multiple", "question": "Which function should be called to satisfy the user's request?", "options": {"uber.ride": "Finds a suitable Uber ride for the customer based on the starting location, the desired ride type, and the maximum wait time the customer is willing to accept.", "api.weather": "Retrieve current weather information for a specified location."}, "context": {"user_request": "T\u00ecm chuy\u1ebfn xe cho t\u00f4i t\u1eeb H\u00e0 N\u1ed9i lo\u1ea1i Plus trong 10 ph\u00fat n\u1eefa. Address: 123 Hanoi Street"}, "label": "uber.ride"}
```

## 3. Alternatives (checked via HF API only, not downloaded; counts for Glaive from HF datasets-server)
- Glaive function-calling v2 (apache-2.0, 112,960 rows): synthetic GPT-4-generated chats with function schema in system prompt; label = which function (or none) in the next assistant turn. Plenty of volume but synthetic, mostly 1-3 functions; label trust medium; multi-function share not verified. 2-3 h.
- xLAM-60k (cc-by-4.0, gated: needs login, 60k, synthetic verified): not checked, gated.
- ToolBench/StableToolBench (apache-2.0 on HF mirror): labels come from ChatGPT trajectories, large API sets (hundreds of options per call), noisy; not recommended.
- Banking77 (cc-by-4.0, 13,083 utterances, 77 labels) and CLINC150 (cc-by-3.0, 23,700 in-scope+oos): high-trust human labels, tens of thousands, but single utterance, intent classification; weakest "agent" story, useful only as an easy-regime contrast (a 77-way fixed option set fits per-type calibration perfectly).
- Nexus function calling: HF card has no license field; not examined.

## Recommendation
Primary: **tau-bench historical_trajectories (reward=1)**, decision_type per domain (retail 12,301, airline 2,565), calibrate/test split by task. Realistic multi-turn context, fixed 17-way option set per type, MIT. Caveat for the post: label = verified-successful agent's choice, not human gold. Add tau2 results (telecom, + more models) only if more distinct tasks are wanted (~3-4 h).
Backup / second panel: **BFCL live_multiple + multiple** (1,253, human gold) for an "unambiguous label" panel, and optionally Banking77 as the easy contrast.

## Jev cost @ $0.042/M input tokens, tokens = chars/3.5
Add ~500 chars instruction overhead per call.
- tau-bench full: 85.3M chars (ctx+options) + 7.4M overhead = 92.7M chars = 26.5M tokens -> **~$1.11**. A 5,000-decision subsample: ~$0.37.
- BFCL 1,253: (0.77M + 0.63M)=1.4M chars = 0.4M tokens -> **~$0.02**.
Not verified: whether `rflx replay` input size/latency limits hold with 17 options x 300-char descriptions; Jev/Laya accuracy on this task (unknown, determines coverage).
