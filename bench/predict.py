"""Run a System 1 model once per decision and cache raw predictions as JSONL (resumable, budgeted)."""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from reflexlayer.s1 import LogprobLLMAdapter, S1Error, SystemOneAdapter

S1 = {
    "jev": ("https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.13"),
    "laya": ("http://127.0.0.1:8790/v1/systemone", "multilingual"),
}
OPT_CHARS, CTX_CHARS = 60, 1800  # compact mode: fits laya's 1,024-token window
MAX_CONSECUTIVE_ERRORS = 50  # circuit breaker: failed calls may still be billed


def compact(item):
    """Keep each option's first sentence (<= OPT_CHARS) and the last CTX_CHARS of every context value."""
    opts = item["options"]
    if isinstance(opts, dict):
        opts = {k: v.split(". ")[0][:OPT_CHARS] for k, v in opts.items()}
    ctx = {k: str(v)[-CTX_CHARS:] for k, v in item["context"].items()}
    return {**item, "options": opts, "context": ctx}


def _rows(path):
    """Parsed rows of the cache file; lines that fail to parse (torn by a kill mid-write) are skipped."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def done_indices(path):
    return {r["i"] for r in _rows(path) if r.get("error") is None}


def spent_so_far(path):
    return sum(r.get("cost") or 0.0 for r in _rows(path))  # failed calls may be billed too


def _ensure_newline(path):
    """Stop a torn last line from gluing onto the next appended row."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")


def run(items, adapter, out_path, workers, budget, do_compact, require_cost=False):
    """Returns the dollars spent by this call. `budget` is a total cap across resumes (seeded from the file)."""
    lock, stop = threading.Lock(), threading.Event()
    total = [spent_so_far(out_path)]  # includes earlier runs
    this_run = [0.0]
    consecutive = [0]
    no_cost = [False]
    if total[0] >= budget:
        stop.set()
    _ensure_newline(out_path)
    with open(out_path, "a", encoding="utf-8") as out:
        def one(pair):
            i, item = pair
            if stop.is_set():
                return
            row = {"i": i}
            try:
                row.update(decision_type=item["decision_type"], task_id=item["meta"]["task_id"], label=item["label"])
                x = compact(item) if do_compact else item
                p = adapter.predict(x["question"], x["options"], x["context"])
                row.update(choice=p.choice, probs=p.probs, p_max=p.p_max, cost=p.cost, ms=p.latency_ms, error=None)
            except S1Error as e:  # message is the S1Error reason only, never a response body or key
                row["error"] = f"S1Error: {e}"[:120]
                if e.cost:
                    row["cost"] = e.cost
            except Exception as e:  # other types: record the class name only, never its message
                row["error"] = type(e).__name__
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                cost = row.get("cost") or 0.0
                total[0] += cost
                this_run[0] += cost
                if require_cost and row["error"] is None and not cost and not stop.is_set():
                    stop.set()
                    no_cost[0] = True
                consecutive[0] = 0 if row["error"] is None else consecutive[0] + 1
                if total[0] >= budget:
                    stop.set()
                if consecutive[0] >= MAX_CONSECUTIVE_ERRORS and not stop.is_set():
                    stop.set()
                    print(f"circuit breaker: {MAX_CONSECUTIVE_ERRORS} consecutive errors, no new requests", file=sys.stderr)

        # Soft cap: up to workers-1 calls already in flight can finish past the budget.
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, items))
    if no_cost[0]:
        sys.exit("S1 reports no cost (no usage.cost): the budget cap cannot be enforced, refusing to continue")
    return this_run[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bench/data/tau_decisions.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1", choices=sorted([*S1, "llm"]), required=True)
    ap.add_argument("--s1-model", help="OpenRouter model id, required with --s1 llm")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    key = os.environ.get("REFLEX_S1_API_KEY") if args.s1 in ("jev", "llm") else None
    if args.s1 in ("jev", "llm") and not key:
        sys.exit("REFLEX_S1_API_KEY is not set")
    if args.s1 == "llm":
        if not args.s1_model:
            sys.exit("--s1-model is required with --s1 llm")
        adapter = LogprobLLMAdapter("https://openrouter.ai/api/v1", args.s1_model, api_key=key, timeout=args.timeout, max_state_chars=20_000)
    else:
        url, model = S1[args.s1]
        adapter = SystemOneAdapter(url, model, api_key=key, timeout=args.timeout, max_state_chars=20_000)  # per key: tail (5,000) + digest + last message fit whole
    done = done_indices(args.out)
    with open(args.data, encoding="utf-8") as f:
        items = [(i, json.loads(l)) for i, l in enumerate(f) if l.strip() and i not in done]
    items = items[: args.limit] if args.limit else items
    spent = run(items, adapter, args.out, args.workers, args.budget_usd, args.compact, require_cost=args.s1 == "llm")
    total = spent_so_far(args.out)
    print(f"requested {len(items)} decisions, spent this run ${spent:.4f}, total ${total:.4f} of ${args.budget_usd:.2f} budget, already done before: {len(done)}")


if __name__ == "__main__":
    main()
