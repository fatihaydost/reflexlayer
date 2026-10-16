"""Task-split comparison of hand-picked thresholds vs Learn-then-Test on cached S1 predictions."""
import argparse
import json
import random
import statistics
from collections import Counter, defaultdict

from reflexlayer.calibrate import MIN_SESSIONS, ltt_threshold

FIXED = [0.8, 0.9, 0.95]
GRID_FLOOR = 0.50  # lowest threshold in reflexlayer.calibrate.GRID


def load(path):
    """Last successful row per decision index; also the number of indices that never succeeded."""
    ok, seen = {}, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:  # torn by a kill mid-write; the decision is re-run on resume
                continue
            if not isinstance(r, dict) or r.get("i") is None:
                continue
            seen.add(r["i"])
            if r.get("error") is None:
                ok[r["i"]] = r
    return [ok[i] for i in sorted(ok)], len(seen - set(ok))


def split_by_task(rows, seed, calib_frac=0.5):
    tasks = sorted({r["task_id"] for r in rows})
    random.Random(seed).shuffle(tasks)
    calib = set(tasks[: int(len(tasks) * calib_frac)])
    return [r for r in rows if r["task_id"] in calib], [r for r in rows if r["task_id"] not in calib]


def pairs(rows, unit="task_id"):
    """(p_max, agree, session); unit="i" makes every row its own session, i.e. the old row-level test."""
    return [(r["p_max"], r["choice"] == r["label"], r[unit]) for r in rows]


def served(test_pairs, lam):
    if lam is None or not test_pairs:
        return 0.0, None
    sel = [agree for p, agree, *_ in test_pairs if p >= lam]
    return len(sel) / len(test_pairs), ((len(sel) - sum(sel)) / len(sel) if sel else None)


def served_task_avg(test_pairs, lam):
    """Mean over tasks with a served row of each task's disagreement rate: the quantity LTT session bounds."""
    if lam is None:
        return None
    by_task = defaultdict(list)
    for p, agree, task in test_pairs:
        if p >= lam:
            by_task[task].append(agree)
    return sum(1 - sum(a) / len(a) for a in by_task.values()) / len(by_task) if by_task else None


def _quantile(vals, q):
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _methods(alphas, delta):
    m = {f"fixed {f}": (None, lambda calib, f=f: f) for f in FIXED}
    for a in alphas:  # tasks act as sessions; "row" is the pre-session test, kept for comparison
        m[f"LTT session a={a}"] = (a, lambda calib, a=a: ltt_threshold(pairs(calib), a, delta))
        m[f"LTT row a={a}"] = (a, lambda calib, a=a: ltt_threshold(pairs(calib, "i"), a, delta))
    return m


def evaluate(rows, alphas, delta, splits):
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["decision_type"]].append(r)
    out = {}
    for t, rs in sorted(by_type.items()):
        p = [r["p_max"] for r in rs]
        top, top_n = Counter(r["label"] for r in rs).most_common(1)[0]
        res = {"n": len(rs), "majority_label": top, "majority_share": top_n / len(rs),
               "tasks": len({r["task_id"] for r in rs}), "splits": splits, "delta": delta,
               "accuracy": sum(r["choice"] == r["label"] for r in rs) / len(rs),
               "p_max_quantiles": {f"p{q}": _quantile(p, q / 100) for q in (10, 50, 90)},
               "share_ge_grid_floor": sum(x >= GRID_FLOOR for x in p) / len(p), "methods": {}}
        for name, (alpha, pick) in _methods(alphas, delta).items():
            covs, errs, held, lams, served_in = [], [], [], [], []
            tot_err = tot_srv = 0
            held_srv, held_task = [], []
            for seed in range(splits):
                calib, test = split_by_task(rs, seed)
                lam = pick(calib)
                cov, err = served(pairs(test), lam)
                covs.append(cov)
                lams.append(lam)
                served_in.append(lam is not None and cov > 0)
                if err is not None:
                    errs.append(err)
                    n_srv = round(cov * len(test))
                    tot_srv += n_srv
                    tot_err += round(err * n_srv)
                if alpha is not None:
                    held.append(err is None or err <= alpha)
                    if err is not None:
                        held_srv.append(err <= alpha)
                        held_task.append(served_task_avg(pairs(test), lam) <= alpha)
            found = [l for l in lams if l is not None]
            res["methods"][name] = {
                "threshold_mean": statistics.mean(found) if found else None,
                "no_threshold_share": lams.count(None) / splits,
                "served_in_share": sum(served_in) / splits,
                "coverage_mean": statistics.mean(covs),
                "pooled_disagreement": tot_err / tot_srv if tot_srv else None,
                "disagreement_p10": _quantile(errs, 0.1) if errs else None,
                "disagreement_p90": _quantile(errs, 0.9) if errs else None,
                "disagreement_mean_unweighted": statistics.mean(errs) if errs else None,
                "held_when_serving": sum(held_srv) / len(held_srv) if held_srv else None,
                "held_count": sum(held_srv) if alpha is not None else None,
                "serving_count": len(held_srv) if alpha is not None else None,
                "held_task_avg_count": sum(held_task) if alpha is not None else None,
                # abstaining counts as held, so this is vacuous for methods that mostly abstain
                "held_or_abstained_share": statistics.mean(held) if held else None,
            }
        out[t] = res
    return out


def _f(x, pct=True):
    return "–" if x is None else (f"{x:.1%}" if pct else f"{x:.2f}")


def _spread(m):
    return "–" if m["disagreement_p10"] is None else f"{m['disagreement_p10']:.1%}–{m['disagreement_p90']:.1%}"


def _held(m):
    if m["serving_count"] is None:
        return "– | –"
    return f"{m['held_count']}/{m['serving_count']} | {m['held_task_avg_count']}/{m['serving_count']}"


def render(results):
    lines, notes_src = [], None
    for model, by_type in results.items():
        for t, res in by_type.items():
            q = res["p_max_quantiles"]
            notes_src = notes_src or res
            lines += [f"### {model} · {t}", "",
                      f"{res['n']} decisions, {res['tasks']} tasks, top-1 agreement with teacher label {_f(res['accuracy'])}, "
                      f"majority baseline ('{res['majority_label']}') {_f(res['majority_share'])}, "
                      f"p_max p10/p50/p90 {q['p10']:.2f}/{q['p50']:.2f}/{q['p90']:.2f}, "
                      f"≥{GRID_FLOOR:.2f}: {res['share_ge_grid_floor']:.0%}", "",
                      "| method | threshold | no threshold | served in | coverage | pooled disagreement | p10–p90 | held when serving | held, task-avg |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for name, m in res["methods"].items():
                lines.append(f"| {name} | {_f(m['threshold_mean'], pct=False)} | {_f(m['no_threshold_share'])} | "
                             f"{_f(m['served_in_share'])} | {_f(m['coverage_mean'])} | {_f(m['pooled_disagreement'])} | "
                             f"{_spread(m)} | {_held(m)} |")
            lines.append("")
    if notes_src:
        lines += ["Notes", "",
                  "- LTT bounds the population disagreement with probability >= 1-δ over calibration draws "
                  f"(δ = {notes_src['delta']}). It does not promise every test half is <= α: a threshold whose true risk sits "
                  "just under α will exceed α on a finite test half fairly often.",
                  "- LTT session treats each task as one session: it bounds the task-averaged disagreement with a "
                  f"Hoeffding-Bentkus test over >= {MIN_SESSIONS} tasks. LTT row treats every decision as its own session "
                  "(the previous row-level test); rows within a task are correlated, so its guarantee does not hold here.",
                  "- Threshold columns average splits where a threshold was found; disagreement figures (pooled, p10–p90, "
                  "held when serving) use serving splits only; coverage averages all splits (no threshold counts as 0).",
                  "- held when serving = held/serving: among splits with coverage > 0, how many had test disagreement <= α "
                  "(fixed thresholds have no α). Pooled disagreement = sum of disagreements / sum of served rows over splits.",
                  "- held, task-avg = the same count with test disagreement measured as the mean over served tasks of each "
                  "task's disagreement rate, the quantity LTT session bounds.",
                  "- Labels are the teacher's next action on verified-successful runs, not gold.",
                  f"- Splits are by task: {notes_src['splits']} overlapping splits over one dataset, so shares are not independent trials.",
                  f"- The threshold grid is 1.00..{GRID_FLOOR:.2f}.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path, repeatable")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    results, errors = {}, {}
    for spec in args.pred:
        name, path = spec.split("=", 1)
        rows, errors[name] = load(path)
        results[name] = evaluate(rows, args.alphas, args.delta, args.splits)
    print(render(results))
    print("S1 errors (never succeeded): " + ", ".join(f"{k} {v}" for k, v in errors.items()))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "errors": errors, "alphas": args.alphas, "delta": args.delta,
                       "splits": args.splits}, f, indent=2)


if __name__ == "__main__":
    main()
