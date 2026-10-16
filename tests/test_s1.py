import json
from datetime import datetime

import httpx
import pytest

from reflexlayer.s1 import S1Error, StrandsDeciderAdapter, SystemOneAdapter


def _adapter(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return SystemOneAdapter("http://s1/v1/systemone", "multilingual", api_key="sk-secret", client=client, **kw)


def _ok(probs, cost=0.00002):
    def handler(req):
        answer = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs, "confidence": 0.9}
        return httpx.Response(200, json={"answers": {"d": answer}, "usage": {"cost": cost}})
    return handler


def test_predict_returns_argmax_pmax_and_cost():
    p = _adapter(_ok({"billing": 0.9, "tech": 0.1})).predict("Which team?", ["billing", "tech"], {"message": "refund"})
    assert (p.choice, p.p_max, p.cost) == ("billing", 0.9, 0.00002)
    assert p.latency_ms >= 0


def test_request_shape():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        seen["auth"] = req.headers["authorization"]
        return _ok({"a": 0.5, "b": 0.5})(req)

    _adapter(handler).predict("Q?", {"a": "first", "b": "second"}, {"x": 1})
    assert seen["model"] == "multilingual"
    assert seen["questions"]["d"] == {"type": "choice", "instructions": "Q?", "criteria": {"a": "first", "b": "second"}}
    assert seen["state"] == {"x": "1"}
    assert seen["auth"] == "Bearer sk-secret"


def test_no_auth_header_without_key():
    seen = {}

    def handler(req):
        seen["has_auth"] = "authorization" in req.headers
        return _ok({"a": 0.6, "b": 0.4})(req)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    SystemOneAdapter("http://s1/v1/systemone", "multilingual", client=client).predict("Q", ["a", "b"], {})
    assert seen["has_auth"] is False


def test_two_decimal_rounding_is_tolerated():
    p = _adapter(_ok({"a": 0.34, "b": 0.34, "c": 0.33})).predict("Q", ["a", "b", "c"], {})
    assert p.p_max == 0.34


@pytest.mark.parametrize("probs", [{"a": 0.5, "zzz": 0.5}, {"a": 1.0}, {"a": 0.7, "b": 0.5}])
def test_invalid_probabilities_raise(probs):
    with pytest.raises(S1Error):
        _adapter(_ok(probs)).predict("Q", ["a", "b"], {})


def test_http_error_raises_without_leaking_secret():
    a = _adapter(lambda req: httpx.Response(401, text="bad key sk-secret"))
    with pytest.raises(S1Error) as e:
        a.predict("Q", ["a", "b"], {})
    assert "sk-secret" not in str(e.value)


def test_timeout_raises_s1error():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(S1Error):
        _adapter(handler).predict("Q", ["a", "b"], {})


def test_state_is_stringified_and_truncated():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return _ok({"a": 0.6, "b": 0.4})(req)

    _adapter(handler, max_state_chars=40).predict("Q", ["a", "b"], {"long": "x" * 100, "when": datetime(2026, 1, 1)})
    assert all(isinstance(v, str) and len(v) <= 20 for v in seen["state"].values())


@pytest.mark.parametrize("probs", [{"a": -0.5, "b": 1.5}, {"a": float("nan"), "b": 0.5}])
def test_out_of_range_or_nan_probabilities_raise(probs):
    with pytest.raises(S1Error):
        _adapter(_ok(probs)).predict("Q", ["a", "b"], {})


def _strands(handler):
    return StrandsDeciderAdapter(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_strands_decider_request_and_answer():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        seen["has_auth"] = "authorization" in req.headers
        # response shape of strands_decider.schema.SystemOneResponse plus server latency_ms; no cost
        return httpx.Response(200, json={
            "model": "strands-decider-2B-qwen3.5-v1-2610",
            "answers": {"d": {"type": "choice", "choice": "billing", "confidence": 0.659,
                              "probabilities": {"billing": 0.7731, "sales": 0.1211, "retail": 0.1057}}},
            "usage": {"input_tokens": 86, "output_tokens": 1}, "latency_ms": 115.2})

    p = _strands(handler).predict("Which team?", ["billing", "sales", "retail"], {"message": "payouts failing"})
    assert (p.choice, p.p_max, p.cost) == ("billing", 0.7731, 0.0)
    assert seen["url"] == "http://127.0.0.1:8000/v1/systemone"
    assert seen["body"]["questions"]["d"]["criteria"] == {"billing": "billing", "sales": "sales", "retail": "retail"}
    assert seen["has_auth"] is False


def test_strands_decider_validation_error_is_s1error():
    with pytest.raises(S1Error):
        _strands(lambda req: httpx.Response(422, json={"detail": "choice requires at least 2 options"})).predict(
            "Q", ["a", "b"], {})


def test_strands_decider_skips_single_option_call():
    calls = []
    with pytest.raises(S1Error):
        _strands(lambda req: calls.append(req)).predict("Q", ["a"], {})
    assert calls == []
