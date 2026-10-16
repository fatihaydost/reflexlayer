"""Convert tau-bench historical trajectories (reward=1 only) into Reflex decision JSONL.

Label = the next action a strong agent took in a run that ended in the correct DB state
(a teacher label on verified-successful runs, not a human gold label).
"""
import argparse
import collections
import glob
import json
import os
import re

QUESTION = "Given the conversation so far, what should the customer-service agent do next: call one tool or reply to the user?"
CTX_CAP, TOOL_RESULT_CAP, DESC_CAP, FIRST_USER_CAP, LAST_AGENT_CAP = 5000, 600, 300, 600, 1000
MARKER = "[...]"


def build_context(turns, cap=CTX_CAP):
    """Join whole turns into <= cap chars: keep the first user turn, then the newest turns that fit."""
    text = "\n".join(turns)
    if len(text) <= cap:
        return text
    first = next((i for i, t in enumerate(turns) if t.startswith("[user] ")), None)
    head = turns[first][:FIRST_USER_CAP] if first is not None else None
    used = (len(head) + 1 if head is not None else 0) + len(MARKER) + 1
    start = first + 1 if first is not None else 0
    tail = []
    for t in reversed(turns[start:]):
        if used + len(t) + 1 > cap:
            break
        tail.insert(0, t)
        used += len(t) + 1
    if not tail and len(turns) - 1 >= start and cap - used > 1:
        tail = [turns[-1][-(cap - used - 1):]]  # newest turn alone too big: keep its final chars
    parts = ([head] if head is not None else []) + [MARKER] + tail
    return "\n".join(parts)


def tool_descriptions(root, domain):
    d = {}
    for f in glob.glob(os.path.join(root, "tau_bench/envs", domain, "tools/*.py")):
        s = open(f, encoding="utf-8").read()
        n = re.search(r'"name":\s*"(\w+)"', s)
        m = re.search(r'"description":\s*(\(?\s*(?:"[^"\n]*"\s*)+)', s)
        if n and m:
            d[n.group(1)] = "".join(re.findall(r'"([^"\n]*)"', m.group(1)))[:DESC_CAP]
    d["respond"] = "Send a message to the user (ask a question, answer, confirm) instead of calling a tool."
    return d


def with_digest(context, digest, last_agent):
    """Add the policy digest, and the agent's last message to the user when the capped tail lost it."""
    out = dict(context)
    if last_agent is not None and last_agent not in context["conversation_tail"]:
        out["last_agent_message"] = last_agent[-LAST_AGENT_CAP:]
    out["policy_digest"] = digest
    return out


def decisions(root, domain, digest=None):
    options = tool_descriptions(root, domain)
    for path in sorted(glob.glob(os.path.join(root, "historical_trajectories", f"*{domain}.json"))):
        for t, x in enumerate(json.load(open(path, encoding="utf-8"))):
            if x["reward"] != 1:
                continue
            traj = x["traj"]
            for i, m in enumerate(traj):
                if m["role"] != "assistant":
                    continue
                calls = m.get("tool_calls")
                label = calls[0]["function"]["name"] if calls else "respond"
                turns, last_agent = [], None
                for p in traj[1:i]:  # traj[0] is the policy system prompt
                    if p["role"] == "tool":
                        turns.append(f"[tool result {p['name']}] {p['content'][:TOOL_RESULT_CAP]}")
                    elif p["role"] == "assistant" and p.get("tool_calls"):
                        turns.append("[agent called] " + ", ".join(
                            c["function"]["name"] + str(c["function"]["arguments"]) for c in p["tool_calls"]))
                    else:
                        turns.append(f"[{p['role']}] {p['content']}")
                        if p["role"] == "assistant":
                            last_agent = turns[-1]
                ctx = {"conversation_tail": build_context(turns)}
                if digest is not None:
                    ctx = with_digest(ctx, digest, last_agent)
                yield {
                    "decision_type": f"tau_{domain}_next_action",
                    "question": QUESTION,
                    "options": options,
                    "context": ctx,
                    "label": label,
                    "meta": {"task_id": f"{domain}:{x['task_id']}", "source": os.path.basename(path), "trial": x.get("trial"), "traj": t, "step": i},
                }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="bench/data/tau-bench-src")
    ap.add_argument("--out", default="bench/data/tau_decisions.jsonl")
    ap.add_argument("--digest", action="store_true", help="add bench/policy_digest_<domain>.txt to each context")
    args = ap.parse_args(argv)
    counts, tasks = collections.Counter(), collections.defaultdict(set)
    with open(args.out, "w", encoding="utf-8") as out:
        for domain in ("retail", "airline"):
            digest = None
            if args.digest:
                with open(os.path.join(os.path.dirname(__file__), f"policy_digest_{domain}.txt"), encoding="utf-8") as f:
                    digest = f.read().strip()
            for row in decisions(args.root, domain, digest):
                out.write(json.dumps(row) + "\n")
                counts[domain] += 1
                tasks[domain].add(row["meta"]["task_id"])
    for d in counts:
        print(f"{d}: {counts[d]} decisions, {len(tasks[d])} tasks")


if __name__ == "__main__":
    main()
