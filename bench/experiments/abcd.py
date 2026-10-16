"""Offline evaluation of cached S1 predictions on ABCD next-button decisions (no S1 calls).

Views: "official" (calibrate on train+dev sessions, test on test sessions, one run) and "random" (N seeded
50/50 session splits over all sessions). Methods: fixed thresholds and session-level LTT (hb and wsr).
Each decision type is calibrated and evaluated on its own; "total" pools rows of all types.
Baselines learn from calibration rows only: majority = the most frequent button over all calibration rows,
last-button = the most frequent next button for the same decision type (type = last button pressed).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluate import load, pairs, split_by_task  # noqa: E402

from reflexlayer.calibrate import ltt_threshold  # noqa: E402

FIXED = [0.8, 0.9, 0.95]
HI = 0.9


def join(preds, decisions_path):
    """Attach meta.split to each prediction; row i of the decisions file is prediction i."""
    split = []
    with open(decisions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                split.append(json.loads(line)["meta"]["split"])
    bad = [r["i"] for r in preds if not 0 <= r["i"] < len(split)]
    if bad:
        raise ValueError(f"prediction index {bad[0]} is outside the decisions file ({len(split)} rows); wrong --decisions?")
    return [{**r, "split": split[r["i"]]} for r in preds]


def official_split(rows):
    calib = [r for r in rows if r["split"] in ("train", "dev")]
    test = [r for r in rows if r["split"] == "test"]
    leak = {r["task_id"] for r in calib} & {r["task_id"] for r in test}
    if leak:
        raise ValueError(f"{len(leak)} sessions are in both calibration and test, e.g. {sorted(leak)[:3]}")
    return calib, test


def methods(alphas, delta):
    m = {f"fixed {f}": (None, lambda p, f=f: f) for f in FIXED}
    for bound in ("hb", "wsr"):
        for a in alphas:
            m[f"LTT-{bound} a={a}"] = (a, lambda p, a=a, bound=bound: ltt_threshold(p, a, delta, bound=bound))
    return m


def baselines(calib):
    """(domain majority label, {decision_type: most frequent label}) from calibration rows only."""
    top = lambda rs: Counter(r["label"] for r in rs).most_common(1)[0][0] if rs else None  # noqa: E731
    by_type = defaultdict(list)
    for r in calib:
        by_type[r["decision_type"]].append(r)
    return top(calib), {t: top(rs) for t, rs in by_type.items()}


def run_split(calib, test, meths):
    """{"ref": {type: counts}, method: {type: {n, served, wrong, threshold}}} for one calibration/test split."""
    major, last = baselines(calib)
    types = sorted({r["decision_type"] for r in test})
    out = {"ref": {}}
    for t in types:
        c = [r for r in calib if r["decision_type"] == t]
        te = [r for r in test if r["decision_type"] == t]
        hi = [r for r in te if r["p_max"] >= HI]
        out["ref"][t] = {"n": len(te), "s1_correct": sum(r["choice"] == r["label"] for r in te),
                         "hi_n": len(hi), "hi_wrong": sum(r["choice"] != r["label"] for r in hi),
                         "majority_correct": sum(r["label"] == major for r in te),
                         "last_correct": sum(r["label"] == last.get(t, major) for r in te)}
        for name, (_, pick) in meths.items():
            lam = pick(pairs(c))
            srv = [r for r in te if lam is not None and r["p_max"] >= lam]
            sess = defaultdict(lambda: [0, 0])  # session -> [served rows, wrong rows]
            for r in srv:
                sess[r["task_id"]][0] += 1
                sess[r["task_id"]][1] += r["choice"] != r["label"]
            out.setdefault(name, {})[t] = {"n": len(te), "served": len(srv), "threshold": lam, "sess": dict(sess),
                                           "wrong": sum(r["choice"] != r["label"] for r in srv)}
    return out


def _session_avg(sess):
    return sum(w / n for n, w in sess.values()) / len(sess)


def _merge_sess(dicts):
    out = defaultdict(lambda: [0, 0])
    for d in dicts:
        for sid, (n, w) in d.items():
            out[sid][0] += n
            out[sid][1] += w
    return dict(out)


def aggregate(stats, alpha=None):
    """stats: per-split {n, served, wrong, ...} (types already summed). Coverage is the mean of per-split
    served/n (row-weighted inside a split), disagreement is pooled over serving splits."""
    stats = [s for s in stats if s["n"]]
    serving = [s for s in stats if s["served"]]
    srv = sum(s["served"] for s in serving)
    return {"coverage": sum(s["served"] / s["n"] for s in stats) / len(stats) if stats else None,
            "pooled_disagreement": sum(s["wrong"] for s in serving) / srv if srv else None,
            "served_in": len(serving), "splits": len(stats),
            "held_rows": sum(s["wrong"] / s["served"] <= alpha for s in serving) if alpha is not None else None,
            # LTT bounds the mean over sessions of each session's disagreement rate, not the row-pooled rate
            "held": sum(_session_avg(s["sess"]) <= alpha for s in serving) if alpha is not None else None}


def _sum(dicts, keys):
    return {k: sum(d[k] for d in dicts) for k in keys}


def summarize(splits, meths):
    """Aggregate run_split outputs over splits, per type and for the domain total."""
    types = sorted({t for s in splits for t in s["ref"]})
    ref_keys = ("n", "s1_correct", "hi_n", "hi_wrong", "majority_correct", "last_correct")
    res = {"reference": {}, "methods": {m: {} for m in meths}}
    for t in types + [None]:
        sel = lambda s, m: [v for k, v in s[m].items() if t in (None, k)]  # noqa: E731
        ref = _sum([v for s in splits for v in sel(s, "ref")], ref_keys)
        n = ref["n"] or 1
        res["reference"][t or "total"] = {
            **ref, "s1_top1": ref["s1_correct"] / n, "majority": ref["majority_correct"] / n,
            "last_button": ref["last_correct"] / n,
            "hi_share": ref["hi_n"] / n, "hi_disagreement": ref["hi_wrong"] / ref["hi_n"] if ref["hi_n"] else None}
        for name, (alpha, _) in meths.items():
            per = [{**_sum(sel(s, name), ("n", "served", "wrong")), "sess": _merge_sess(v["sess"] for v in sel(s, name))}
                   for s in splits]
            lams = [v["threshold"] for s in splits for v in sel(s, name) if v["threshold"] is not None]
            res["methods"][name][t or "total"] = {**aggregate(per, alpha), "threshold_mean": sum(lams) / len(lams) if lams and t else None}  # no meaningful mean across types
    return res


def evaluate(rows, alphas, delta, n_splits):
    meths = methods(alphas, delta)
    views = {"official": [run_split(*official_split(rows), meths)],
             "random": [run_split(*split_by_task(rows, seed), meths) for seed in range(n_splits)]}
    return {v: summarize(s, meths) | {"splits": len(s)} for v, s in views.items()}


def _p(x):
    return "–" if x is None else f"{x:.1%}"


def _k(m, key):
    return "–" if m[key] is None else f"{m[key]}/{m['served_in']}"


def _cell(m):
    held = "" if m["held"] is None else f" / held {m['held']}/{m['served_in']}"
    return f"{_p(m['coverage'])} / {_p(m['pooled_disagreement'])}{held} / in {m['served_in']}/{m['splits']}"


def render(out):
    lines = []
    for view, res in out.items():
        lines += [f"### View: {view} ({res['splits']} split{'s' if res['splits'] != 1 else ''})", "",
                  "| method | coverage | pooled disagreement | held when serving (session avg) | held (rows, not the guaranteed quantity) | served in |", "| --- | --- | --- | --- | --- | --- |"]
        for name, by in res["methods"].items():
            m = by["total"]
            lines.append(f"| {name} | {_p(m['coverage'])} | {_p(m['pooled_disagreement'])} | {_k(m, 'held')} | {_k(m, 'held_rows')} | {m['served_in']}/{m['splits']} |")
        names = list(res["methods"])
        lines += ["", "Per decision type (test rows pooled over splits; method cells are coverage / pooled disagreement / held (session avg) k of splits served / served in k of splits):", "",
                  "| type | n | S1 top-1 | majority | last-button | dis. at p_max>=0.9 | " + " | ".join(names) + " |",
                  "| " + " | ".join(["---"] * (6 + len(names))) + " |"]
        for t, r in res["reference"].items():
            hi = f"{_p(r['hi_disagreement'])} ({_p(r['hi_share'])} of rows)"
            cells = " | ".join(_cell(res['methods'][m][t]) for m in names)
            lines.append(f"| {t} | {r['n']} | {_p(r['s1_top1'])} | {_p(r['majority'])} | {_p(r['last_button'])} | {hi} | {cells} |")
        lines.append("")
    lines += ["Notes", "",
              "- Held when serving (session avg): test disagreement measured as the mean over served sessions of each session's rate, the quantity LTT bounds; the rows column pools rows instead.",
              "- The total row pools types: LTT guarantees each type separately, so a per-type guarantee does not imply the total holds.",
              "- Label = the human agent's logged button (MTurk role-play), not policy-optimal. Data distributed in the MIT-licensed asappresearch/abcd repo.",
              "- Official view: calibrate on train+dev sessions, test on test sessions, one run; held when serving is 0/1 or 1/1.",
              "- Random view: seeded 50/50 session splits over one dataset, so the splits are not independent trials.",
              "- Each type is calibrated on its own; total pools all types (coverage per split = served / test rows, then averaged).",
              "- Majority = most frequent button over all calibration rows; last-button = most frequent button for the type in calibration. Both learn from calibration rows only.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="bench/data/pred-abcd-jev.jsonl")
    ap.add_argument("--decisions", default="bench/data/abcd_decisions.jsonl")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    preds, failed = load(args.pred)
    out = evaluate(join(preds, args.decisions), args.alphas, args.delta, args.splits)
    print(render(out))
    print(f"S1 errors (never succeeded): {failed}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": out, "failed": failed, "alphas": args.alphas, "delta": args.delta, "splits": args.splits,
                       "pred": args.pred}, f, indent=2)


if __name__ == "__main__":
    main()
