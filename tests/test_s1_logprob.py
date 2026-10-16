import json
import logging
import math

import httpx
import pytest

from reflexlayer.s1 import LogprobLLMAdapter, S1Error


def _resp(top, usage=None, message=None, first=(".", 0)):
    body = {
        "choices": [{"message": message or {"content": "A"},
                     "logprobs": {"content": [{"token": first[0], "logprob": first[1], "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top]}]}}],
        "usage": usage or {},
    }
    return httpx.Response(200, json=body)


def _adapter(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LogprobLLMAdapter("http://llm/v1", "m", api_key="sk-secret", client=client, **kw)


def test_request_shape():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        seen["auth"], seen["url"] = req.headers["authorization"], str(req.url)
        return _resp([("A", -0.1)])

    _adapter(handler).predict("Which team?", {"billing": "money", "tech": "bugs"}, {"msg": "refund"})
    assert seen["url"] == "http://llm/v1/chat/completions"
    assert (seen["max_tokens"], seen["temperature"], seen["logprobs"], seen["top_logprobs"]) == (1, 0, True, 20)
    assert seen["provider"] == {"require_parameters": True}
    assert seen["auth"] == "Bearer sk-secret"
    prompt = " ".join(m["content"] for m in seen["messages"])
    assert "Which team?" in prompt and "A. billing: money" in prompt and "B. tech: bugs" in prompt and "refund" in prompt
    assert "single letter" in prompt


def test_probabilities_keyed_by_names_and_missing_gets_zero():
    p = _adapter(lambda r: _resp([(" A", -0.1), ("B", -2.5), ("C", -5)])).predict("Q", ["x", "y", "z"], {})
    assert set(p.probs) == {"x", "y", "z"} and p.choice == "x"
    assert sum(p.probs.values()) == pytest.approx(1)
    assert p.p_max == p.probs["x"] > 0.9
    q = _adapter(lambda r: _resp([("A", -0.5), ("B", -1.0)])).predict("Q", ["x", "y", "z"], {})
    assert q.probs["z"] == 0.0 and sum(q.probs.values()) == pytest.approx(1)


def test_normalisation_and_logsumexp_aggregation():
    p = _adapter(lambda r: _resp([(" A", -1.0), ("A", -1.0), ("(B", -1.0), ("B.", -3.0), ("Apple", -0.1), ("the", -0.2)])).predict("Q", ["x", "y"], {})
    a, b = 2 * math.exp(-1.0), math.exp(-1.0) + math.exp(-3.0)
    assert p.probs["x"] == pytest.approx(a / (a + b))


def test_no_label_and_no_logprobs():
    with pytest.raises(S1Error, match="no_option_logprobs"):
        _adapter(lambda r: _resp([("the", -0.1), ("Z", -1)])).predict("Q", ["x", "y"], {})
    with pytest.raises(S1Error, match="no_logprobs"):
        _adapter(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "A"}, "logprobs": None}]})).predict("Q", ["x", "y"], {})


def test_reasoning_model_rejected():
    with pytest.raises(S1Error, match="not_a_label"):
        _adapter(lambda r: _resp([("Okay", -0.1)], message={"content": "", "reasoning": "thinking..."})).predict("Q", ["x", "y"], {})


def test_http_error_does_not_leak_body_or_key():
    with pytest.raises(S1Error) as e:
        _adapter(lambda r: httpx.Response(500, text="boom sk-secret")).predict("Q", ["x", "y"], {})
    assert "sk-secret" not in str(e.value) and "boom" not in str(e.value)
    with pytest.raises(S1Error):
        _adapter(lambda r: httpx.Response(200, text="not json")).predict("Q", ["x", "y"], {})
    with pytest.raises(S1Error):
        _adapter(lambda r: httpx.Response(200, json={"choices": []})).predict("Q", ["x", "y"], {})


def test_cost_from_usage_then_price_in_then_zero():
    top = [("A", -0.1)]
    assert _adapter(lambda r: _resp(top, {"cost": 0.0003, "prompt_tokens": 100}), price_in=1.0).predict("Q", ["x", "y"], {}).cost == 0.0003
    assert _adapter(lambda r: _resp(top, {"prompt_tokens": 1000}), price_in=2.0).predict("Q", ["x", "y"], {}).cost == pytest.approx(0.002)
    assert _adapter(lambda r: _resp(top, {"prompt_tokens": 1000})).predict("Q", ["x", "y"], {}).cost == 0.0


def test_too_many_options():
    with pytest.raises(S1Error, match="too_many_options"):
        _adapter(lambda r: _resp([("A", -0.1)])).predict("Q", [str(i) for i in range(27)], {})


def test_extreme_logprobs_are_skipped_not_crashing():
    def h(r):
        body = _resp([("A", -9999), ("B", 0), ("C", -0.2)]).content.replace(b'"token":"B","logprob":0', b'"token":"B","logprob":-Infinity')
        return httpx.Response(200, content=body)

    p = _adapter(h).predict("Q", ["x", "y", "z"], {})
    assert p.choice == "z" and p.probs["z"] == pytest.approx(1)


def test_lowercase_ignored_and_first_token_is_candidate():
    p = _adapter(lambda r: _resp([("a", -0.1), ("i", -0.1), ("B", -1.0)], first=("A", 0))).predict("Q", ["x", "y"], {})
    # lowercase "a"/"i" ignored; content[0] token "A" (logprob 0) counts as A
    assert p.probs["x"] == pytest.approx(1 / (1 + math.exp(-1)))


def test_failed_call_carries_cost():
    with pytest.raises(S1Error) as e:
        _adapter(lambda r: _resp([("the", -0.1)], {"cost": 0.004})).predict("Q", ["x", "y"], {})
    assert e.value.cost == 0.004 and str(e.value) == "no_option_logprobs"


def test_too_few_options_no_http():
    def boom(r):
        raise AssertionError("no call expected")
    with pytest.raises(S1Error, match="too_few_options"):
        _adapter(boom).predict("Q", ["x"], {})


def test_unknown_cost_logs_warning_once(caplog):
    caplog.set_level(logging.WARNING)
    LogprobLLMAdapter._warned = False
    a = _adapter(lambda r: _resp([("A", -0.1)]))
    a.predict("Q", ["x", "y"], {})
    a.predict("Q", ["x", "y"], {})
    assert [r.message for r in caplog.records].count("S1 cost unknown; budget cannot be enforced") == 1


def test_sampled_token_in_top_logprobs_not_double_counted():
    ln = math.log(0.5)
    body = {"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{"token": "A", "logprob": ln, "top_logprobs": [
        {"token": "A", "logprob": ln}, {"token": "B", "logprob": ln}]}]}}]}
    p = _adapter(lambda r: httpx.Response(200, json=body)).predict("Q", ["x", "y"], {})
    assert p.probs["x"] == pytest.approx(0.5) and p.probs["y"] == pytest.approx(0.5)


def test_non_numeric_cost_becomes_zero_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    LogprobLLMAdapter._warned = False
    p = _adapter(lambda r: _resp([("A", -0.1)], {"cost": "abc"})).predict("Q", ["x", "y"], {})
    assert p.cost == 0.0 and "cost unknown" in caplog.text
