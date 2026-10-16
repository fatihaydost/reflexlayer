"""Convert ABCD (asappresearch/abcd v1.1) into Reflex decision JSONL.

Decision = the point where the human agent presses a dashboard button (speaker == "action");
label = the button actually pressed (MTurk role-play, not policy-optimal). decision_type =
next_button_after_<last button | none>; types with < min_sessions calibration sessions merge into
"other" (counts calibration sessions only, never labels or test data).
"""
import argparse
import collections
import json
import random

QUESTION = ("A customer-service agent follows company guidelines and is about to press a button on the agent "
            "dashboard. Which button do they press next?")
OPTIONS = {"pull-up-account": "Look up the customer's account by full name or account ID", "verify-identity": "Verify identity with name/account ID/order ID/zip code", "search-faq": "Open the FAQ/policy search for a general question", "select-faq": "Choose the FAQ answer to read out", "validate-purchase": "Check a purchase using username, email and order ID", "enter-details": "Enter extra details (address, username, email, PIN...) in the details form", "ask-the-oracle": "Query the system oracle whether the customer's claim/request is valid", "record-reason": "Record the reason/refund target/payment method etc. from the customer", "update-order": "Change an order (address, date, item, method, price)", "membership": "Check membership level privileges", "shipping-status": "Look up shipping status/options", "log-out-in": "Troubleshooting step: ask customer to log out and in", "offer-refund": "Offer a refund with a dollar amount", "send-link": "Send a reset/security/info link to the customer", "update-account": "Add/remove/renew/extend a service on the account", "try-again": "Troubleshooting step: ask customer to try again", "subscription-status": "Look up subscription status, amount due or due date", "notify-team": "Notify manager, website team or purchasing department", "promo-code": "Issue a promo code", "make-purchase": "Add a product to an order", "search-membership": "Search FAQ: membership questions", "search-pricing": "Search FAQ: pricing questions", "instructions": "Give troubleshooting instructions", "search-policy": "Search FAQ: policy questions", "make-password": "Generate a new password", "search-timing": "Search FAQ: timing questions", "search-boots": "Search FAQ: boots", "search-jeans": "Search FAQ: jeans", "search-jacket": "Search FAQ: jacket", "search-shirt": "Search FAQ: shirt"}
assert len(OPTIONS) == 30
PREFIX = "next_button_after_"


def conversation_rows(split, c):
    lines, last = [], "none"
    for (speaker, text), x in zip(c["original"], c["delexed"]):
        tg = x["targets"]
        if x["speaker"] == "action":
            assert tg[2] in OPTIONS, tg[2]
            yield {
                "decision_type": PREFIX + last,
                "question": QUESTION,
                "options": OPTIONS,
                "context": {"dialogue": "\n".join(lines) or "(conversation just started)"},
                "label": tg[2],
                "meta": {"task_id": str(c["convo_id"]), "split": split, "flow": c["scenario"]["flow"],
                         "subflow": tg[0], "turn_count": x["turn_count"]},
            }  # emitted before the press is appended: the context never holds its own label
            lines.append(f"[Agent pressed button: {tg[2]}({', '.join(tg[3])})] -> {text}")
            last = tg[2]
        else:
            lines.append(f"{x['speaker'].capitalize()}: {text}")


BIN_QUESTION = ("A customer-service agent follows company guidelines. What do they do next: press a dashboard "
                "button, or reply to the customer?")
BIN_OPTIONS = {"button": "Press a dashboard button (look up, verify, search, update, etc.) before replying",
               "reply": "Write a natural-language reply to the customer now"}


def binary_rows(split, c):
    """One row per agent step (reply or button press); label = button | reply; type by the previous speaker."""
    lines, prev = [], "none"
    for (speaker, text), x in zip(c["original"], c["delexed"]):
        if speaker in ("agent", "action"):
            yield {
                "decision_type": f"button_or_reply_after_{prev}",
                "question": BIN_QUESTION,
                "options": BIN_OPTIONS,
                "context": {"dialogue": "\n".join(lines) or "(conversation just started)"},
                "label": "button" if x["speaker"] == "action" else "reply",
                "meta": {"task_id": str(c["convo_id"]), "split": split, "flow": c["scenario"]["flow"],
                         "subflow": x["targets"][0], "turn_count": x["turn_count"]},
            }
        if x["speaker"] == "action":
            tg = x["targets"]
            lines.append(f"[Agent pressed button: {tg[2]}({', '.join(tg[3])})] -> {text}")
        else:
            lines.append(f"{x['speaker'].capitalize()}: {text}")
        prev = x["speaker"]


def binary_decisions(data, target=5000, seed=7):
    """Whole sessions in seeded random order until >= target decisions (never splits a session)."""
    convs = [(s, c) for s, cs in data.items() for c in cs]
    random.Random(seed).shuffle(convs)
    rows = []
    for s, c in convs:
        if len(rows) >= target:
            break
        rows += binary_rows(s, c)
    return rows


def decisions(data):
    for split, convs in data.items():
        for c in convs:
            yield from conversation_rows(split, c)


def merge_rare(rows, min_sessions=80, calib_splits=("train", "dev")):
    sessions = collections.defaultdict(set)
    for r in rows:
        if r["meta"]["split"] in calib_splits:
            sessions[r["decision_type"]].add(r["meta"]["task_id"])
    rare = {t for t in {r["decision_type"] for r in rows} if len(sessions[t]) < min_sessions}
    return [dict(r, decision_type=PREFIX + "other") if r["decision_type"] in rare else r for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bench/data/abcd/abcd.json")
    ap.add_argument("--out", default="bench/data/abcd_decisions.jsonl")
    ap.add_argument("--min-sessions", type=int, default=80)
    ap.add_argument("--binary", action="store_true", help="button-or-reply decisions on a seeded whole-session subsample")
    args = ap.parse_args(argv)
    with open(args.src, encoding="utf-8") as f:
        data = json.load(f)
    if args.binary:
        rows = binary_decisions(data)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sess = collections.defaultdict(set)
        for r in rows:
            sess[r["meta"]["split"]].add(r["meta"]["task_id"])
        print(f"{len(rows)} decisions, labels {dict(collections.Counter(r['label'] for r in rows))}, "
              f"types {dict(collections.Counter(r['decision_type'] for r in rows))}, sessions {({k: len(v) for k, v in sess.items()})}")
        return
    rows = merge_rare(list(decisions(data)), args.min_sessions)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    calib, dec = collections.defaultdict(set), collections.Counter()
    for r in rows:
        dec[r["decision_type"]] += 1
        if r["meta"]["split"] in ("train", "dev"):
            calib[r["decision_type"]].add(r["meta"]["task_id"])
    for t, n in dec.most_common():
        print(f"{t:40s} calib_sessions={len(calib[t]):5d} decisions={n}")
    print(f"total: {len(rows)} decisions, {len({r['meta']['task_id'] for r in rows})} sessions")


if __name__ == "__main__":
    main()
