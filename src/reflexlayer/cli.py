import argparse
import json
import os
import sys

from .calibrate import calibrate
from .log import DecisionLog
from .report import write_report
from .s1 import LogprobLLMAdapter, StrandsDeciderAdapter, SystemOneAdapter
from .types import LLMResult


def _s1_args(p):
    p.add_argument("--s1", choices=["systemone", "strands-decider", "llm"], default="systemone",
                   help="systemone: Jev or laya-serve; strands-decider: a local `strands-decider serve`; "
                        "llm: logprob adapter over any OpenAI-compatible /chat/completions (needs --s1-model)")
    p.add_argument("--s1-url", help="systemone endpoint, e.g. http://127.0.0.1:8790/v1/systemone "
                                    f"(strands-decider default {StrandsDeciderAdapter.DEFAULT_URL})")
    p.add_argument("--s1-model", help="e.g. multilingual (laya) or typesafe/jev-1.13 (Jev); unused by strands-decider")
    p.add_argument("--s1-timeout", type=float, default=None, help="seconds; default 2 (10 for --s1 llm)")
    p.add_argument("--max-state-chars", type=int, default=6000)


def _make_s1(args):
    if args.s1_timeout is None:
        args.s1_timeout = 10.0 if args.s1 == "llm" else 2.0
    if args.s1 == "strands-decider":
        return StrandsDeciderAdapter(args.s1_url or StrandsDeciderAdapter.DEFAULT_URL,
                                     timeout=args.s1_timeout, max_state_chars=args.max_state_chars)
    if args.s1 == "llm":
        return LogprobLLMAdapter(args.s1_url or "https://openrouter.ai/api/v1", args.s1_model, api_key=os.environ.get("REFLEX_S1_API_KEY"),
                                 timeout=args.s1_timeout, max_state_chars=args.max_state_chars)
    return SystemOneAdapter(args.s1_url, args.s1_model, api_key=os.environ.get("REFLEX_S1_API_KEY"),
                            timeout=args.s1_timeout, max_state_chars=args.max_state_chars)


def main(argv=None, s1=None) -> int:
    ap = argparse.ArgumentParser(prog="rflx", description="Calibrated System-1 routing for agent decisions")
    ap.add_argument("--db", default="reflex.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gateway", help="run the OpenAI-compatible gateway")
    g.add_argument("--upstream", required=True, help="customer's LLM base URL, e.g. https://api.openai.com/v1")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8788)
    g.add_argument("--store-content", action="store_true")
    _s1_args(g)

    c = sub.add_parser("calibrate", help="compute thresholds from shadow data")
    c.add_argument("--type")
    c.add_argument("--alpha", type=float)
    c.add_argument("--delta", type=float)

    for name in ("activate", "shadow", "mark-destructive"):
        p = sub.add_parser(name)
        p.add_argument("--type", required=True)
        if name == "activate":
            p.add_argument("--allow-destructive", action="store_true")

    r = sub.add_parser("report")
    r.add_argument("--out", default="reflex-report.html")

    rp = sub.add_parser("replay", help="run S1 over labelled JSONL and log labels as LLM answers")
    rp.add_argument("jsonl")
    _s1_args(rp)

    args = ap.parse_args(argv)
    if getattr(args, "s1", None) == "systemone" and not (args.s1_url and args.s1_model):
        ap.error("--s1-url and --s1-model are required with --s1 systemone")
    if getattr(args, "s1", None) == "llm" and not args.s1_model:
        ap.error("--s1-model is required with --s1 llm")
    if getattr(args, "s1", None) == "llm" and not os.environ.get("REFLEX_S1_API_KEY"):
        ap.error("REFLEX_S1_API_KEY is not set (required with --s1 llm)")
    log = DecisionLog(args.db, store_content=getattr(args, "store_content", False))

    if args.cmd == "gateway":
        import uvicorn

        from .gateway import create_app
        uvicorn.run(create_app(s1 or _make_s1(args), log, args.upstream), host=args.host, port=args.port)
        return 0

    if args.cmd == "calibrate":
        for t in [args.type] if args.type else log.decision_types():
            try:
                pol = calibrate(log, t, args.alpha, args.delta)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
            print(f"{t}: threshold={pol.threshold} mode={pol.mode} {pol.note}".rstrip())
        return 0

    if args.cmd in ("activate", "shadow", "mark-destructive"):
        known = log.decision_types()
        if args.type not in known:
            print(f"{args.type}: unknown decision type; known types: {', '.join(known) or '(none)'}", file=sys.stderr)
            return 2
        pol = log.get_policy(args.type)
        if args.cmd == "shadow":
            pol.mode = "shadow"
            print(f"{args.type}: shadow")
        elif args.cmd == "mark-destructive":
            pol.destructive = True
            print(f"{args.type}: marked destructive; S1 will not serve it")
        else:
            if pol.threshold is None:
                print(f"{args.type}: no threshold yet, run `rflx calibrate` after collecting shadow data", file=sys.stderr)
                return 2
            if pol.destructive and not args.allow_destructive:
                print(f"{args.type}: marked destructive; pass --allow-destructive to let S1 serve it", file=sys.stderr)
                return 2
            pol.mode, pol.destructive, pol.note = "active", False, ""
            print(f"{args.type}: active, threshold {pol.threshold}, budget <= {pol.alpha:.1%} disagreement "
                  f"at {1 - pol.delta:.0%} confidence, {pol.sample_rate:.0%} audited")
        log.set_policy(pol)
        return 0

    if args.cmd == "report":
        write_report(log, args.out)
        print(args.out)
        return 0

    if args.cmd == "replay":
        s1 = s1 or _make_s1(args)
        ok = failed = 0
        with open(args.jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                try:
                    pred = s1.predict(item["question"], item["options"], item["context"])
                except Exception:
                    pred, failed = None, failed + 1
                log.record(item["decision_type"], item["context"], pred, LLMResult(item["label"], 0.0, 0.0), "llm", item["label"], False,
                           item.get("session_id"))
                ok += 1
        print(f"replayed {ok} decisions, {failed} S1 failures")
        return 1 if ok > 0 and failed == ok else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
