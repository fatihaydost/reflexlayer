import json
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedDecision:
    kind: str  # "schema_enum" | "tool_select"
    options: dict[str, str]
    field: str | None
    question: str
    context: dict
    decision_type: str


def _transcript(messages, last=6):
    lines = []
    for m in messages[-last:]:
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        lines.append(f"{m.get('role', '?')}: {c or ''}")
    return "\n".join(lines)


def detect(req, header_type):
    try:  # malformed request shapes are not decisions; the upstream LLM answers them
        return _detect(req, header_type)
    except (AttributeError, TypeError, ValueError):
        return None


def _detect(req, header_type):
    if req.get("stream") or req.get("n", 1) != 1:
        return None
    ctx = {"conversation": _transcript(req.get("messages") or [])}

    rf = req.get("response_format") or {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema") or {}
        props = schema.get("properties") or {}
        if schema.get("type") == "object" and len(props) == 1:
            (field, spec), = props.items()
            enum = spec.get("enum")
            if spec.get("type") == "string" and isinstance(enum, list) and len(enum) >= 2 and all(isinstance(e, str) for e in enum):
                question = spec.get("description") or js.get("description") or f"Which `{field}` fits the latest turn of `conversation`?"
                return DetectedDecision("schema_enum", {e: e for e in enum}, field, question, ctx,
                                        header_type or f"schema:{js.get('name', 'unnamed')}")

    tools = req.get("tools") or []
    if req.get("tool_choice") == "required" and len(tools) >= 2:
        fns = [t.get("function") or {} for t in tools if t.get("type") == "function"]
        if len(fns) == len(tools) and all(f.get("name") and not any((f.get("parameters") or {}).get(k) for k in ("properties", "required")) for f in fns):
            options = {f["name"]: f.get("description") or f["name"] for f in fns}
            return DetectedDecision("tool_select", options, None, "Which tool should be called next for `conversation`?",
                                    ctx, header_type or "tools:" + ",".join(sorted(options)))
    return None


def build_response(det, choice, model):
    if det.kind == "schema_enum":
        msg, finish = {"role": "assistant", "content": json.dumps({det.field: choice})}, "stop"
    else:
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": choice, "arguments": "{}"}}
        msg, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
    return {
        "id": "chatcmpl-reflex-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def parse_choice(det, resp):
    try:
        msg = resp["choices"][0]["message"]
        choice = json.loads(msg["content"])[det.field] if det.kind == "schema_enum" else msg["tool_calls"][0]["function"]["name"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return choice if isinstance(choice, str) and choice in det.options else None
