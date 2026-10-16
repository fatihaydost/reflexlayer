"""Locally trained probe as Reflex's System 1 on tau-bench, vs cached Jev(digest), under session-level LTT.

Probe: frozen mmBERT-base, mean-pooled last hidden state, softmax regression per domain trained on tau-bench teacher
labels only (never on Jev outputs; the hosted service's terms restrict training on them). Jev predictions are read from the cache as a comparison S1.

Two steps, both in a separate venv with torch + transformers (not in the uv test path; $PY below):
  HF_HUB_OFFLINE=1 $PY bench/experiments/probe.py embed
  HF_HUB_OFFLINE=1 $PY bench/experiments/probe.py run --json bench/results/....json
`embed` caches features under bench/data/ (gitignored) once; `run` only fits linear heads per seed.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from evaluate import load, pairs  # noqa: E402
from experiments.coverage import last_turn  # noqa: E402

from reflexlayer.calibrate import ltt_threshold  # noqa: E402

# Pre-registered before any probe prediction was looked at (see the results header).
PREREG = {
    "encoder": "jhu-clsp/mmBERT-base (frozen, fp16 autocast)",
    "text": "context.conversation_tail exactly as the S1 saw it (digest left out: constant within a domain)",
    "max_length": 2048, "truncation_side": "left (keeps the newest turns; 0.4% of tails exceed 2048 tokens)",
    "pooling": "mean over attention mask of last_hidden_state",
    "head": "softmax regression over the domain's options, features standardised on train rows, AdamW lr 1e-2 wd 1e-2, "
            "300 full-batch steps, seed-fixed init",
    "variants": {"tail": "mean-pool of the full tail (768-d)",
                 "tail+last": "tail ⊕ mean-pool of the newest turn alone (max 512 tokens), 1536-d"},
    "selection": "per domain, the variant with the lower mean NLL on calibrate rows over the primary split's seeds",
    "splits": {"primary": [0.4, 0.3, 0.3], "secondary": [0.5, 0.25, 0.25], "calib-heavy": [0.2, 0.5, 0.3]},
    "split_rule": "per domain, tasks sorted then shuffled with random.Random(seed); train = first floor(n*f_train), "
                  "calibrate = next floor(n*f_cal), test = rest; seeds 0..49",
    "ltt": "reflexlayer.calibrate.ltt_threshold, sessions = task_id, bound hb and wsr, delta 0.1, alpha 0.05 and 0.10",
}
EMB = {"tail": "bench/data/emb-mmbert-tail.npy", "last": "bench/data/emb-mmbert-last.npy"}


def split3(tasks, seed, fracs):
    t = sorted(tasks)
    random.Random(seed).shuffle(t)
    a, b = int(len(t) * fracs[0]), int(len(t) * fracs[0]) + int(len(t) * fracs[1])
    return set(t[:a]), set(t[a:b]), set(t[b:])


def read_decisions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- embedding (torch) ----------

def embed_texts(texts, max_length, budget=8192):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")
    tok.truncation_side = "left"
    enc = AutoModel.from_pretrained("jhu-clsp/mmBERT-base").to("cuda").eval()
    lens = [len(x) for x in tok(texts, truncation=True, max_length=max_length)["input_ids"]]
    order = sorted(range(len(texts)), key=lambda i: -lens[i])
    out = np.zeros((len(texts), enc.config.hidden_size), dtype=np.float32)
    i, t0, last = 0, time.time(), 0
    with torch.no_grad():
        while i < len(order):
            bs = max(1, budget // lens[order[i]])  # longest first: token budget per batch
            idx = order[i:i + bs]
            b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out[idx] = ((h.float() * m).sum(1) / m.sum(1)).cpu().numpy()
            i += bs
            if i - last >= 1000:
                last = i
                print(f"  {i}/{len(order)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)
    print(f"done {len(texts)} in {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MB", flush=True)
    assert np.isfinite(out).all()
    return out


def cmd_embed(args):
    import numpy as np
    dec = read_decisions(args.decisions)
    tails = [d["context"]["conversation_tail"] for d in dec]
    if args.what in ("tail", "both"):
        np.save(EMB["tail"], embed_texts(tails, 2048))
    if args.what in ("last", "both"):
        np.save(EMB["last"], embed_texts([last_turn(t)[1] for t in tails], 512))


# ---------- head (torch) ----------

def fit_predict(Xtr, ytr, Xev_list, n_cls, seed, steps=300):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Xtr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lin = torch.nn.Linear(Xtr.size(1), n_cls)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    yt, Z = torch.tensor(ytr), (Xtr - mu) / sd
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(Z), yt).backward()
        opt.step()
    with torch.no_grad():
        return [F.softmax(lin((torch.from_numpy(X) - mu) / sd), -1).numpy() for X in Xev_list]


# ---------- evaluation (pure python) ----------

def ece(conf, correct):
    edges = [0, .5, .6, .7, .8, .9, .95, 1.0001]
    n, e = len(conf), 0.0
    for lo, hi in zip(edges, edges[1:]):
        m = [k for k, c in enumerate(conf) if lo <= c < hi]
        if m:
            e += len(m) / n * abs(sum(conf[k] for k in m) / len(m) - sum(correct[k] for k in m) / len(m))
    return e


def serve(cal, test, alpha, delta, bound):
    lam = ltt_threshold(pairs(cal), alpha, delta, bound=bound)
    srv = [r for r in test if lam is not None and r["p_max"] >= lam]
    return {"n": len(test), "served": len(srv), "wrong": sum(r["choice"] != r["label"] for r in srv), "lam": lam,
            "cal_sessions": len({r["task_id"] for r in cal})}


def aggregate(splits, alpha):
    srv = [s for s in splits if s["served"]]
    tot = sum(s["served"] for s in srv)
    return {"coverage": sum(s["served"] for s in splits) / sum(s["n"] for s in splits),
            "pooled_disagreement": sum(s["wrong"] for s in srv) / tot if tot else None,
            "held": sum(s["wrong"] / s["served"] <= alpha for s in srv), "serving": len(srv), "splits": len(splits),
            "cal_sessions": statistics.mean(s["cal_sessions"] for s in splits),
            "threshold_mean": statistics.mean(s["lam"] for s in srv) if srv else None}


def quality(rows):
    hi = [r for r in rows if r["p_max"] >= 0.9]
    return {"rows": len(rows), "top1": sum(r["choice"] == r["label"] for r in rows) / len(rows),
            "share_ge_0.9": len(hi) / len(rows),
            "disagreement_ge_0.9": sum(r["choice"] != r["label"] for r in hi) / len(hi) if hi else None,
            "ece": ece([r["p_max"] for r in rows], [r["choice"] == r["label"] for r in rows])}


def cmd_run(args):
    import numpy as np
    dec = read_decisions(args.decisions)
    jev = {r["i"]: r for r in load(args.jev)[0]}
    feats = {"tail": np.load(EMB["tail"])}
    feats["tail+last"] = np.concatenate([feats["tail"], np.load(EMB["last"])], 1)
    by_dom = defaultdict(list)
    for i, d in enumerate(dec):
        by_dom[d["decision_type"]].append(i)
    out = {"prereg": PREREG, "delta": args.delta, "seeds": args.seeds, "domains": {}}
    for dt, idx in sorted(by_dom.items()):
        opts = list(dec[idx[0]]["options"])
        y = np.array([opts.index(dec[i]["label"]) for i in idx])
        task = [dec[i]["meta"]["task_id"] for i in idx]
        tasks = set(task)
        dom = {"rows": len(idx), "tasks": len(tasks), "options": len(opts), "variants": {}, "views": {}}
        # 1) variant selection on calibrate rows of the primary split (test never touched)
        nll = {v: [] for v in feats}
        preds = {}  # (view, seed) -> probe rows on cal + test for the selected variant, filled below
        cache = {}
        for name, fr in PREREG["splits"].items():
            for seed in range(args.seeds):
                tr, ca, te = split3(tasks, seed, fr)
                m_tr = np.array([t in tr for t in task]); m_ca = np.array([t in ca for t in task]); m_te = np.array([t in te for t in task])
                for v, X in feats.items():
                    Pc, Pt = fit_predict(X[idx][m_tr], y[m_tr], [X[idx][m_ca], X[idx][m_te]], len(opts), seed)
                    cache[(name, seed, v)] = (np.flatnonzero(m_ca), Pc, np.flatnonzero(m_te), Pt)
                    if name == "primary":
                        nll[v].append(float(-np.log(Pc[np.arange(len(Pc)), y[m_ca]] + 1e-12).mean()))
            print(f"{dt} {name}: heads fitted", flush=True)
        dom["variants"] = {v: {"cal_nll_mean": statistics.mean(x)} for v, x in nll.items()}
        best = min(nll, key=lambda v: statistics.mean(nll[v]))
        dom["selected"] = best

        def rows_of(pos, P):
            return [{"task_id": task[k], "label": dec[idx[k]]["label"], "choice": opts[int(P[j].argmax())],
                     "p_max": round(float(P[j].max()), 6)} for j, k in enumerate(pos)]

        def jrows(pos):
            return [{**jev[idx[k]], "task_id": task[k]} for k in pos]

        for name, fr in PREREG["splits"].items():
            view = {"fracs": fr, "configs": {}}
            q = defaultdict(list)
            res = defaultdict(list)
            for seed in range(args.seeds):
                tr, ca, te = split3(tasks, seed, fr)
                for v in feats:  # every variant reported, the selected one is marked
                    pc, Pc, pt, Pt = cache[(name, seed, v)]
                    cal_r, test_r = rows_of(pc, Pc), rows_of(pt, Pt)
                    q[f"probe[{v}]"] += test_r
                    for a in args.alphas:
                        for bnd in ("hb", "wsr"):
                            res[(f"probe[{v}]", a, bnd)].append(serve(cal_r, test_r, a, args.delta, bnd))
                pc, _, pt, _ = cache[(name, seed, best)]
                pos_tr = [k for k, t in enumerate(task) if t in tr]
                jc, jt, jtr = jrows(pc), jrows(pt), jrows(pos_tr)
                q["jev-digest"] += jt
                for a in args.alphas:
                    for bnd in ("hb", "wsr"):
                        res[("jev-digest", a, bnd)].append(serve(jc, jt, a, args.delta, bnd))
                        res[("jev-digest (calib on train+cal)", a, bnd)].append(serve(jc + jtr, jt, a, args.delta, bnd))
            view["quality_test_pooled"] = {k: quality(v) for k, v in q.items()}
            view["configs"] = {f"{s1} a={a} {bnd}": aggregate(sp, a) for (s1, a, bnd), sp in res.items()}
            dom["views"][name] = view
            print(f"{dt} {name}: evaluated", flush=True)
        out["domains"][dt] = dom
    print(render(out))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


def _p(x):
    return "–" if x is None else f"{x:.1%}"


def _f2(x):
    return "–" if x is None else f"{x:.2f}"


def render(out):
    lines = []
    for dt, dom in out["domains"].items():
        d = dt.split("_")[1]
        lines += [f"### {d}: {dom['rows']} rows, {dom['tasks']} tasks, {dom['options']} options; selected variant "
                  f"`{dom['selected']}` (calibrate NLL " + ", ".join(f"{v} {x['cal_nll_mean']:.3f}" for v, x in dom["variants"].items()) + ")", ""]
        for name, view in dom["views"].items():
            lines += [f"#### {d}, {name} split {'/'.join(str(round(f * 100)) for f in view['fracs'])}", "",
                      "| S1 | rows | top-1 | share p≥0.9 | disagreement p≥0.9 | ECE |", "| --- | --- | --- | --- | --- | --- |"]
            for s1, qq in view["quality_test_pooled"].items():
                lines.append(f"| {s1} | {qq['rows']} | {_p(qq['top1'])} | {_p(qq['share_ge_0.9'])} | {_p(qq['disagreement_ge_0.9'])} | {qq['ece']:.3f} |")
            lines += ["", "| S1 | α | bound | calib sessions | coverage | pooled disagreement | held when serving (rows) | served in | mean threshold |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for k, m in view["configs"].items():
                s1, rest = k.split(" a=")
                a, bnd = rest.split()
                lines.append(f"| {s1} | {a} | {bnd} | {m['cal_sessions']:.0f} | {_p(m['coverage'])} | {_p(m['pooled_disagreement'])} | "
                             f"{m['held']}/{m['serving']} | {m['serving']}/{m['splits']} | "
                             f"{_f2(m['threshold_mean'])} |")
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--what", choices=["tail", "last", "both"], default="both")
    r = sub.add_parser("run")
    r.add_argument("--jev", default="bench/data/pred-jev-digest.jsonl")
    r.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    r.add_argument("--delta", type=float, default=0.1)
    r.add_argument("--seeds", type=int, default=50)
    r.add_argument("--json")
    for p in (e, r):
        p.add_argument("--decisions", default="bench/data/tau_decisions_digest.jsonl")
    args = ap.parse_args(argv)
    (cmd_embed if args.cmd == "embed" else cmd_run)(args)


if __name__ == "__main__":
    main()
