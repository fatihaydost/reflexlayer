import json
import warnings

import httpx
from fakes import FakeS1
# starlette's TestClient warns about httpx (prefers httpx2); a new dependency is out of scope
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with")
    from starlette.testclient import TestClient

from reflexlayer.gateway import create_app
from reflexlayer.log import DecisionLog
from reflexlayer.types import Policy

ENUM_REQ = {
    "model": "gpt-x",
    "messages": [{"role": "user", "content": "I want my money back"}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "triage", "schema": {
        "type": "object", "properties": {"intent": {"type": "string", "enum": ["refund", "other"]}}, "required": ["intent"]}}},
}


def upstream(content='{"intent": "refund"}', status=200, headers=None):
    calls = []

    def handler(req):
        calls.append({"url": str(req.url), "headers": dict(req.headers), "body": req.content})
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "rate limited"}}, headers=headers)
        return httpx.Response(200, json={"id": "up", "object": "chat.completion", "model": "gpt-x",
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}, headers=headers)
    return handler, calls


def make(tmp_path, handler, s1=None, policy=None):
    log = DecisionLog(tmp_path / "g.db")
    if policy:
        log.set_policy(policy)
    app = create_app(s1 or FakeS1("refund"), log, "http://up/v1", client=httpx.Client(transport=httpx.MockTransport(handler)))
    return TestClient(app), log


def test_plain_request_passes_through_untouched(tmp_path):
    handler, calls = upstream(content="hello")
    c, log = make(tmp_path, handler)
    r = c.post("/v1/chat/completions", json={"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]},
               headers={"authorization": "Bearer sk-cust"})
    assert r.status_code == 200 and r.json()["id"] == "up"
    assert calls[0]["url"] == "http://up/v1/chat/completions" and calls[0]["headers"]["authorization"] == "Bearer sk-cust"
    assert log.decision_types() == []


def test_stream_request_passes_through(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    c.post("/v1/chat/completions", json={**ENUM_REQ, "stream": True})
    assert len(calls) == 1 and log.decision_types() == []


def test_shadow_returns_upstream_and_logs_pair(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    r = c.post("/v1/chat/completions", json=ENUM_REQ)
    assert r.json()["id"] == "up" and len(calls) == 1
    assert log.pairs("schema:triage") == [(0.95, True, None)]


def test_session_header_is_logged_not_forwarded(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    c.post("/v1/chat/completions", json=ENUM_REQ, headers={"x-reflex-session-id": "conv-9"})
    assert log.pairs("schema:triage") == [(0.95, True, "conv-9")]
    assert "x-reflex-session-id" not in calls[0]["headers"]


def test_active_serves_s1_without_calling_upstream(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler, policy=Policy("schema:triage", mode="active", threshold=0.9, sample_rate=0.0))
    r = c.post("/v1/chat/completions", json=ENUM_REQ)
    assert calls == []
    assert r.headers["x-reflex-tier"] == "s1"
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"intent": "refund"}


def test_upstream_error_returned_verbatim(tmp_path):
    handler, calls = upstream(status=429)
    c, log = make(tmp_path, handler)
    r = c.post("/v1/chat/completions", json=ENUM_REQ)
    assert r.status_code == 429 and r.json() == {"error": {"message": "rate limited"}}


def test_unparseable_upstream_answer_returned_verbatim(tmp_path):
    handler, calls = upstream(content="not json")
    c, log = make(tmp_path, handler)
    r = c.post("/v1/chat/completions", json=ENUM_REQ)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "not json"


def test_decision_type_header_used_and_not_forwarded(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    c.post("/v1/chat/completions", json=ENUM_REQ, headers={"x-reflex-decision-type": "support-triage"})
    assert log.decision_types() == ["support-triage"]
    assert "x-reflex-decision-type" not in calls[0]["headers"]


def test_invalid_json_body_passes_through(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    c.post("/v1/chat/completions", content=b"{oops", headers={"content-type": "application/json"})
    assert calls[0]["body"] == b"{oops"


def test_other_endpoints_are_proxied(tmp_path):
    handler, calls = upstream()
    c, log = make(tmp_path, handler)
    c.get("/v1/models?limit=2")
    assert calls[0]["url"] == "http://up/v1/models?limit=2"


def test_upstream_error_headers_preserved(tmp_path):
    handler, calls = upstream(status=429, headers={"retry-after": "7", "x-request-id": "abc", "content-encoding": "identity"})
    c, log = make(tmp_path, handler)
    r = c.post("/v1/chat/completions", json=ENUM_REQ)
    assert r.status_code == 429 and r.headers["retry-after"] == "7" and r.headers["x-request-id"] == "abc"
    assert "content-encoding" not in r.headers and r.json() == {"error": {"message": "rate limited"}}
