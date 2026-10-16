"""Offline coverage levers for session-level LTT on cached tau-bench S1 predictions (no S1 calls).

Levers, fixed from domain knowledge before evaluation:
- rule: a write tool chosen by S1 is not served unless the last turn is a user turn with an affirmative.
- types: each domain's decision type split by the kind of the last turn (user / tool result / other).
- wsr: the betting bound (bound="wsr") instead of Hoeffding-Bentkus inside the same LTT loop.
Exploratory, added after seeing the rule's result (so not pre-registered):
- nowrite: S1 never serves a write tool; every write goes to the LLM.
"""
import argparse
import itertools
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import load, pairs, split_by_task  # noqa: E402

from reflexlayer.calibrate import ltt_threshold  # noqa: E402

# Tools that change the DB. Every retail one says "ask for explicit user confirmation (yes/no)" in its
# description; the airline descriptions do not, but these are its book_/cancel_/update_/send_ tools.
WRITE = {
    "tau_retail_next_action": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
                               "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
                               "return_delivered_order_items"},
    "tau_airline_next_action": {"book_reservation", "cancel_reservation", "send_certificate", "update_reservation_baggages",
                                "update_reservation_flights", "update_reservation_passengers"},
}
AFFIRM = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|go ahead|proceed|sure|correct|that's right)\b", re.I)
TURN = re.compile(r"^\[(user|assistant|agent called|tool result [^\]]*)\] ")
LEVERS = ("rule", "types", "wsr")
EXPLORATORY = [("nowrite",), ("nowrite", "types"), ("nowrite", "types", "wsr")]


def last_turn(tail):
    """(kind, text) of the newest turn; user turns may span lines, so walk back to the last turn marker."""
    lines = tail.split("\n")
    for j in range(len(lines) - 1, -1, -1):
        m = TURN.match(lines[j])
        if m:
            kind = {"user": "user", "agent called": "other", "assistant": "other"}.get(m.group(1), "tool")
            return kind, "\n".join(lines[j:])
    return "other", tail


def blocked(row, nowrite=False):
    if row["choice"] not in WRITE.get(row["decision_type"], ()):
        return False
    return nowrite or not (row["kind"] == "user" and AFFIRM.search(row["turn"]))


def join(preds, decisions_path):
    ctx = {}
    with open(decisions_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                ctx[i] = json.loads(line)["context"]["conversation_tail"]
    out = []
    for r in preds:
        kind, turn = last_turn(ctx[r["i"]])
        out.append({**r, "kind": kind, "turn": turn})
    return out


def run_split(rows, seed, alpha, delta, cfg, calib_frac=0.5):
    calib, test = split_by_task(rows, seed, calib_frac)
    gate = "rule" in cfg or "nowrite" in cfg
    eff = (lambda r: 0.0 if blocked(r, "nowrite" in cfg) else r["p_max"]) if gate else (lambda r: r["p_max"])
    group = (lambda r: r["kind"]) if "types" in cfg else (lambda r: "all")
    bound = "wsr" if "wsr" in cfg else "hb"
    groups = sorted({group(r) for r in rows})
    lams = {g: ltt_threshold(pairs([{**r, "p_max": eff(r)} for r in calib if group(r) == g]), alpha, delta, bound=bound)
            for g in groups}
    srv = [r for r in test if lams[group(r)] is not None and eff(r) >= lams[group(r)]]
    by_task = defaultdict(list)
    for r in srv:
        by_task[r["task_id"]].append(r["choice"] == r["label"])
    sub = {g: (sum(group(r) == g for r in srv), sum(group(r) == g for r in test), lams[g]) for g in groups}
    return {"n": len(test), "served": len(srv), "wrong": sum(r["choice"] != r["label"] for r in srv),
            "task_avg": statistics.mean(1 - sum(a) / len(a) for a in by_task.values()) if by_task else None, "sub": sub}


def aggregate(splits, alpha):
    serving = [s for s in splits if s["served"]]
    tot_srv = sum(s["served"] for s in serving)
    sub = defaultdict(lambda: [0, 0, 0])  # served rows, test rows, splits with a threshold
    for s in splits:
        for g, (a, b, lam) in s["sub"].items():
            sub[g][0] += a
            sub[g][1] += b
            sub[g][2] += lam is not None
    return {"coverage": statistics.mean(s["served"] / s["n"] for s in splits),
            "pooled_disagreement": sum(s["wrong"] for s in serving) / tot_srv if tot_srv else None,
            "held": sum(s["wrong"] / s["served"] <= alpha for s in serving), "held_task_avg": sum(s["task_avg"] <= alpha for s in serving),
            "serving": len(serving), "splits": len(splits),
            "sub": {g: {"coverage": a / b if b else 0.0, "threshold_splits": t} for g, (a, b, t) in sub.items()}}


def rule_stats(rows, nowrite=False):
    b = [r for r in rows if blocked(r, nowrite)]
    hi = [r for r in rows if r["p_max"] >= 0.9 and r["choice"] != r["label"]]
    return {"rows": len(rows), "blocked": len(b), "blocked_agreeing": sum(r["choice"] == r["label"] for r in b),
            "hi_conf_disagreements": len(hi), "hi_conf_disagreements_blocked": sum(blocked(r, nowrite) for r in hi)}


def configs():
    return [c for k in range(len(LEVERS) + 1) for c in itertools.combinations(LEVERS, k)] + EXPLORATORY


def agreement(rows):
    hi = [r for r in rows if r["p_max"] >= 0.9]
    return {"top1": sum(r["choice"] == r["label"] for r in rows) / len(rows), "share_ge_0.9": len(hi) / len(rows),
            "disagreement_ge_0.9": sum(r["choice"] != r["label"] for r in hi) / len(hi) if hi else None}


def experiment(rows, alphas, delta, n_splits, cfgs=None, calib_frac=0.5):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["decision_type"]].append(r)
    out = {}
    for t, rs in sorted(by_type.items()):
        kinds = defaultdict(set)
        for r in rs:
            kinds[r["kind"]].add(r["task_id"])
        res = {"agreement": agreement(rs), "rule": rule_stats(rs), "nowrite": rule_stats(rs, nowrite=True), "tasks_per_kind": {k: len(v) for k, v in sorted(kinds.items())},
               "rows_per_kind": {k: sum(r["kind"] == k for r in rs) for k in sorted(kinds)}, "results": {}}
        for cfg in cfgs or configs():
            for a in alphas:
                res["results"][f"{'+'.join(cfg) or 'baseline'} a={a}"] = aggregate(
                    [run_split(rs, seed, a, delta, cfg, calib_frac) for seed in range(n_splits)], a)
        out[t] = res
    return out


def _p(x):
    return "–" if x is None else f"{x:.1%}"


def render(out):
    lines = ["| config | domain | α | coverage | pooled disagreement | held when serving | held, task-avg | served in | per sub-type coverage (splits with threshold) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t, res in out.items():
        dom = t.split("_")[1]
        for name, m in res["results"].items():
            cfg, a = name.rsplit(" a=", 1)
            sub = ", ".join(f"{g} {v['coverage']:.1%} ({v['threshold_splits']})" for g, v in m["sub"].items()) if "types" in cfg else ""
            lines.append(f"| {cfg} | {dom} | {a} | {_p(m['coverage'])} | {_p(m['pooled_disagreement'])} | {m['held']}/{m['serving']} | "
                         f"{m['held_task_avg']}/{m['serving']} | {m['serving']}/{m['splits']} | {sub} |")
    lines.append("")
    for t, res in out.items():
        g = res["agreement"]
        lines.append(f"- {t}: top-1 agreement {g['top1']:.1%}; p_max >= 0.9 on {g['share_ge_0.9']:.1%} of rows with "
                     f"{_p(g['disagreement_ge_0.9'])} disagreement.")
        r, w = res["rule"], res["nowrite"]
        lines.append(f"- {t}: nowrite blocks {w['blocked']} rows ({w['blocked_agreeing']} agreed), removes "
                     f"{w['hi_conf_disagreements_blocked']} of {w['hi_conf_disagreements']} disagreements at p_max >= 0.9.")
        lines.append(f"- {t}: rule blocks {r['blocked']} of {r['rows']} rows ({r['blocked_agreeing']} of them agreed with the teacher); "
                     f"removes {r['hi_conf_disagreements_blocked']} of {r['hi_conf_disagreements']} disagreements at p_max >= 0.9. "
                     f"Tasks per last-turn kind: {res['tasks_per_kind']}; rows: {res['rows_per_kind']}.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="bench/data/pred-jev.jsonl")
    ap.add_argument("--decisions", default="bench/data/tau_decisions.jsonl")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--configs", nargs="+", help="e.g. baseline nowrite wsr nowrite+wsr; default: all")
    ap.add_argument("--calib-frac", type=float, default=0.5, help="share of tasks in the calibration half")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    cfgs = [() if c == "baseline" else tuple(c.split("+")) for c in args.configs] if args.configs else None
    preds, _ = load(args.pred)
    out = experiment(join(preds, args.decisions), args.alphas, args.delta, args.splits, cfgs, args.calib_frac)
    print(render(out))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": out, "alphas": args.alphas, "delta": args.delta, "splits": args.splits,
                       "calib_frac": args.calib_frac, "pred": args.pred}, f, indent=2)


if __name__ == "__main__":
    main()
