# ABCD feasibility spike for Reflex (2026-10-16)

Sources: `git clone --depth 1 github.com/asappresearch/abcd` (commit 6b8700c, 2022-01-26) into `abcd-spike/repo`; `data/abcd_v1.1.json.gz` gunzipped to `abcd-spike/data/abcd.json` (121 MB); paper PDF arXiv 2104.00783. All counts below come from my own `python3 -I` scripts on the data files (scripts in `abcd-spike/*.py`, none executed from inside the clone). Jev was NOT called; no accuracy of S1 is measured here.

## 1. Size (counted from abcd_v1.1.json)

| split | dialogues (sessions) | all turns (cust+agent+action) | agent button presses ("take_action") |
|---|---|---|---|
| train | 8,034 | 176,434 | 29,190 |
| dev | 1,004 | 22,380 | 3,684 |
| test | 1,004 | 22,169 | 3,608 |
| total | 10,042 | 220,983 | 36,482 |

- Turn breakdown: 89,372 customer, 95,129 agent utterances, 36,482 action events (a button press, logged as its own "action" turn with system output text). Mean 22.0 turns per dialogue (min 6, max 75). Matches paper (10,042; 22.1).
- 10 flows, 55 canonical subflows (ontology.json and `targets[0]`). The `scenario.subflow` field has 96 (flow, variant) strings such as `pricing_3`; use `delexed[*].targets[0]` for the 55-way label (the first-turn target is the canonical subflow).
- Flow sizes: storewide_query 1,094; purchase_dispute 1,076; product_defect 1,070; account_access 1,048; single_item_query 1,045; order_issue 1,040; troubleshoot_site 1,026; shipping_issue 1,020; subscription_inquiry 910; manage_account 713.
- Subflow sizes (all splits): min 30, max 361; 5 subflows under 80 sessions (status_active, manage_change_phone/name/address, manage_payment_method). Test split has as few as 3 per subflow.
- "next_step" targets in the data only take `retrieve_utterance` (95,129) and `take_action` (36,482); `end_conversation` never occurs as a label.

## 2. Licence

- Repo `LICENSE`: MIT License, "Copyright (c) 2021 ASAPP Research", quote: "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software". The data file lives inside this repo (`data/abcd_v1.1.json.gz`).
- No separate data licence, terms of use, non-commercial or research-only clause anywhere in the repo (grep for licen/terms/non-commercial: only LICENSE matches). README has only a citation request. Caveat: MIT text formally covers "Software and associated documentation files"; there is no explicit statement that the dialogue data are under MIT. Reasonable reading is permissive, but a one-line email to dchen@asapp.com (listed in README) would settle it before a public launch claim. I did not contact anyone.
- Paper: data are fictional customers/orders generated from prompts, collected from MTurk crowdworkers who were paid (ethics section). Utterances may contain made-up PII (names, emails, phones) - fictional by construction.
- Paper itself is arXiv CC BY 4.0 (irrelevant to data).

## 3. Label structure

- Collection: "Expert Live Chat". Humans on both sides, real-time, on MTurk. Agents are qualified (20-question quiz, 80% pass, ~20% pass rate, tutorials, bonuses for identifying the right intent); customers get a prompt generated from a hidden scenario (flow/subflow + fictional personal/order/product data). Agents follow the "Agent Guidelines" document (guidelines.json: 10 flows, 55 subflows, ~33 distinct button names with step text) and use a dashboard with buttons (30 distinct in the data) that take slot values.
- Per-turn `targets` = [subflow (55), next_step (3), button (30), slot values (125 value options in ontology), utterance rank (100 candidates)]. Subflow/flow are auto-annotated from the scenario provenance (not human-judged). Button and values are the human agent's actual dashboard clicks (logged, not post-hoc annotated). So the labels are the human agent's behaviour, not a model teacher.
- 30 buttons (counts of presses): pull-up-account 7,070; verify-identity 3,680; search-faq 2,419; validate-purchase 2,418; enter-details 2,095; ask-the-oracle 1,748; select-faq 1,703; record-reason 1,614; update-order 1,578; membership 1,434; shipping-status 940; log-out-in 899; offer-refund 826; send-link 763; update-account 730; try-again 719; subscription-status 634; notify-team 584; promo-code 581; make-purchase 483; search-membership 471; search-pricing 405; instructions 399; search-policy 392; make-password 362; search-timing 337; search-boots 312; search-jeans 303; search-jacket 301; search-shirt 282.
- Subflow to action mapping: kb.json lists the expected button sequence per subflow (e.g. recover_username = pull-up-account, verify-identity). Per-flow the data use 23-30 of the 30 buttons, so flows do not shrink the option set much (single_item_query uses all 30).
- Label noise measured here: the observed button sequence equals the guideline (kb.json) sequence exactly in only 3,465/10,042 (34.5%) sessions; 288 more after collapsing consecutive repeats, 431 contain it as a subsequence; 5,566 (55%) are missing at least one expected button; mean recall of expected buttons 0.78. Humans deviate (skip steps, extra FAQ searches, repeated presses). So "label" is what the human did, not the policy optimum.
- Paper reference numbers (Table 4): button prediction (AST, given context) fine-tuned BERT/RoBERTa 85.9-88.5%; human volunteers 79.0% on button choice (100 samples/person, 55/30-way lists), next-step (act vs respond) models 87.6% vs human 84.0%; intent (55-way) models ~89-90.5% vs human 85.5%. Interpretation: human agreement ceiling is below 95% on the button decision, i.e. irreducible ambiguity of roughly 10-20%.
- My own cheap baselines on test (train-fit, action decisions only, no dialogue text): predict from last button only = 41.9%; oracle subflow + last button = 77.4%. The remaining signal sits in the text, which Jev would read.

## 4. Candidate finite-choice formulations

Context for every formulation: dialogue so far (original text, "Agent:/Customer:" lines, plus `[Agent pressed button: name(values)] -> system output` lines), no scenario/intent leak. Measured on the real data: mean 697 chars, median 637, p95 1,595, max 3,945, 0 of 131,611 decisions over 6,000 chars. Action-only decisions: mean 678 chars.

| formulation | decisions | sessions per decision type | options | context chars (mean) | label quality | conversion effort |
|---|---|---|---|---|---|---|
| (a1) next button given an action is taken, one type | 36,482 | 10,042 (single type) | 30 | 678 | human clicks; paper human row: 84.0% next-step / 79.0% button-slot | ~2 h (done in spike: build.py, conv.py) |
| (a2) same, decision_type = last button pressed (observable at runtime) | 36,482 | 31 types; 27 types >= 80 sessions (none 10,042; pull-up-account 6,861; verify-identity 2,922; search-faq 2,337; validate-purchase 2,031; enter-details 1,734; record-reason 1,442; ask-the-oracle 1,185; membership 1,060; ... send-link 133). Under 80: update-account 91 is OK; offer-refund 77, make-purchase 50, promo-code 25, make-password 23 -> merge into "other" | 30 | 678 | as above; majority baseline within type 16-86% | ~3 h |
| (a3) next step incl. "respond" (31 options), type = last button | 131,611 | 31 types (all >= 269 sessions) | 31 | 697 | label skew: 72.3% are "respond"; trivial always-respond has 27.7% disagreement; several consecutive agent messages count as separate decisions; weaker, inflated claim | ~3 h |
| (b) per-flow subtype (flow as decision_type) | 36,482 | 10 types, 713-1,094 sessions each | 23-30 | 678 | flow is latent (agent must infer); fine for ablation, not a run-time key | ~1 h on top of a1 |
| (c1) subflow classification at conversation start (55-way) | 10,042 (1 per session) | 10,042 (single type) | 55 | first 2-6 turns, ~200-500 | auto-labelled from scenario; human 85.5%, BERT 89-90% on 55-way with full dialogue; first utterance often does not reveal subflow (e.g. status_ vs manage_ siblings) | ~3 h (needs descriptions of 55 subflows from guidelines.json) |
| (c2) subflow given flow (3-8 options) | 10,042 | 10 types, 713-1,094 | 3-8 | same | easier but needs flow known | ~3 h |

Counting rule note: "sessions per decision type" counts distinct dialogues that contain at least one decision of that type. All 10,042 sessions have at least one button press. Calibration does not need a train split (no training), so all splits can be pooled; if you keep the official test split for final evaluation, scale by 0.1.

## 5. Pros and cons vs tau-bench

Pros of ABCD:
- Sessions: 10,042 versus 43/111 per domain in tau-bench. Per-type session counts of 1,000-10,000 satisfy even alpha=0.01 (needs ~ >=300-500 per type) and 27 decision types clear 80.
- Labels are human behaviour (logged clicks of qualified MTurk agents in real-time chat), not a teacher model; the claim "agrees with a human agent" is stronger than "agrees with the S2 teacher".
- Context is short (about 700 chars), so S1 inputs are cheap and well under 6,000 chars; zero decisions need truncation.
- MIT-licensed repo, public, widely cited (NAACL 2021), reproducible; fixed option set with a documented ontology.
- Decisions are of one fixed schema, which also makes cross-type calibration easy.

Cons:
- Domain is a scripted e-commerce support role-play (fictional "AcmeBrands"), crowdworkers, not production traffic; text is templated-ish and models fine-tuned in 2021 already reach ~86-88% button accuracy. No tool-calling or agentic trajectory, no real consequences.
- Only 34.5% of sessions match the guideline button sequence exactly (our count) and the paper's human button-slot accuracy is 79.0%: expect label noise. At alpha=0.05 the served fraction may be low; expect Reflex to serve mainly deterministic steps (first pull-up-account, select-faq after search-*, make-password etc.). Unmeasured: needs a Jev pilot to see coverage.
- Teacher-free: the "disagreement with S2" framing changes to "disagreement with a human" (cannot compare S1 to the S2 model unless you also run S2 on the same rows).
- Action-vs-respond: dataset has no end_conversation label and the respond decision is free text; only (a3) covers it and is skewed.
- Within-session decisions are correlated and unequal (3.6 button decisions per session; 1 decision per session in c1); session-level LTT unit works but variance per session is high with few decisions.
- Licence ambiguity for the data portion (see section 2).
- Official test split is only 1,004 sessions; per-type counts shrink 10x if you hold it out.

## 6. Recommendation

Use ABCD, formulation (a2): decision = next dashboard button given the agent acts, decision_type = "next_button_after_<last button>" (merge 4 rare states into "other"), 30 named options with one-line descriptions, context = dialogue-so-far. Keep (a3) (with "respond") as a secondary robustness check only, and (c1) only if you want a 55-option stress test. Rationale: runtime-observable key, 10,042 sessions overall, >= 80 sessions in 27 of 31 types (full) and 12 types in a 5,000-decision whole-session subsample, short contexts, human labels.

## 7. Jev cost estimate (input only, $0.042 per 1M tokens, chars/3.5)

Prompt = question + 30 name:description options (about 1,980 chars) + context. Output tokens ignored (single choice).

| scenario | decisions | est. input tokens | cost |
|---|---|---|---|
| (a2) full, all 10,042 sessions | 36,482 | 27.7 M | $1.16 |
| (a2) 5,000-decision subsample (whole sessions, 5,003 decisions from 1,403 sessions) | 5,003 | 3.8 M | $0.16 |
| (a3) with respond, full | 131,611 | 103.2 M | $4.33 |
| (a3) 5,000 decisions | 5,000 | 3.9 M | $0.16 |

Even doubling for the full guideline excerpt per prompt (kb.json is only ~5 KB, about 1.5k tokens) cost stays under ~$5 for (a2) full. Subsample caveat: sampling 5,003 decisions by whole random sessions leaves 12 decision types with >= 80 sessions (none 1,403; pull-up-account 949; verify-identity 383; search-faq 321; validate-purchase 275; enter-details 235; record-reason 198; membership 151; ask-the-oracle 151; shipping-status 86; subscription-status 82; try-again 80). For LTT at alpha=0.05 this subsample is borderline per type; the full set is cheap enough to just run.

## 8. Example JSONL rows (options elided here; full file not shipped (regenerate with bench/convert_abcd.py), options = {button: one-line description}, 30 entries)

    {"decision_type": "next_button_after_none", "question": "A customer-service agent follows company guidelines and is about to press a button on the agent dashboard. Which button do they press next?", "options": "<30 name:description pairs>", "context": {"dialogue": "Agent: Thank you for contacting Acme Brands! How can I help you today?\nCustomer: Hi! I'd like to check my shipping because the shipping seems like it was too high!\nCustomer: Sorry the shipping cost seems like it was too high!\nAgent: Okay, I can help you with that. Can I have your name or account ID?\nCustomer: Sure my name is Crystal Minh."}, "label": "pull-up-account", "meta": {"task_id": 9854, "split": "train", "flow": "shipping_issue", "subflow": "cost", "turn_count": 6}}
    {"decision_type": "next_button_after_verify-identity", "question": "A customer-service agent follows company guidelines and is about to press a button on the agent dashboard. Which button do they press next?", "options": "<30 name:description pairs>", "context": {"dialogue": "Agent: Hello! Thank you for shopping with AcmeBrands. How may I assist you?\nCustomer: Can\nCustomer: hi\nAgent: Hello! :)\nCustomer: Can i move hy shipping to next week\nAgent: I'll be glad to help you with that. Would you provide your full name, please?\nCustomer: Sanya Afzal\n[Agent pressed button: pull-up-account(sanya afzal)] -> Account has been pulled up for Sanya Afzal.\nAgent: To continue assisting you, may I please get your Account ID and Order ID?\nCustomer: DYWI9VUKSF\nCustomer: 5284434792\n[Agent pressed button: verify-identity(sanya afzal, dywi9vuksf, 5284434792)] -> Identity verification in progress ...\nAgent: Perfect. What is the shipping status on your order? It may be \"Order Received\", \"In Transit\", \"Out For Delivery\", or \"Delivered\".\nCustomer: In Transit"}, "label": "shipping-status", "meta": {"task_id": 6445, "split": "train", "flow": "order_issue", "subflow": "manage_downgrade", "turn_count": 15}}
    {"decision_type": "next_button_after_ask-the-oracle", "question": "A customer-service agent follows company guidelines and is about to press a button on the agent dashboard. Which button do they press next?", "options": "<30 name:description pairs>", "context": {"dialogue": "Agent: How can I help you?\nCustomer: Hello, I want to check a status of my order - it seems like there was a random extra fee added without an explanation.\nAgent: Sorry about that, can I have your name to pull up your account?\nCustomer: My name is David Williams\n[Agent pressed button: pull-up-account(david williams)] -> Account has been pulled up for David Williams.\nAgent: To verify your identity, can I have your account and order ID?\nCustomer: My account ID is QSIVPO9MFO. My order ID is 8058366065\n[Agent pressed button: verify-identity(david williams, qsivpo9mfo, 8058366065)] -> Identity verification in progress ...\n[Agent pressed button: ask-the-oracle()] -> Querying the system for an answer ...\nAgent: Thanks, I was able to verify your identity. Can I have your membership level?\nCustomer: I am bronze level"}, "label": "membership", "meta": {"task_id": 8985, "split": "train", "flow": "order_issue", "subflow": "status_mystery_fee", "turn_count": 12}}

(Indented rows above; the options object in the real file is e.g. {"pull-up-account": "Look up the customer's account by full name or account ID", "verify-identity": "...", ... "search-shirt": "Search FAQ: shirt"}. Labels: pull-up-account, shipping-status, membership.)

## Not checked
- S1 accuracy, calibration or coverage (no Jev call). Human-vs-guideline agreement is measured only through the kb.json sequence match above, which may under-state agreement if kb.json is a simplified template.
- HF/Kaggle mirrors for a separate data licence; the Google Doc of Agent Guidelines (guidelines.json used instead); Appendix A pay details beyond the abstract.
- Which of the 3.6 button decisions per session cluster tightly (within-session correlation not estimated).
