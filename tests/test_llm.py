import json

import httpx
import pytest

from reflexlayer.llm import LLMError, OpenAICompatLLM


def _llm(handler, **kw):
    return OpenAICompatLLM("http://llm/v1/", "gpt-x", "sk-llm", client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def _resp(content, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 5},
    })


def test_returns_choice_and_priced_cost():
    r = _llm(lambda req: _resp('{"choice": "b"}'), price_in=1e-6, price_out=2e-6).decide("Q?", ["a", "b"], {"m": "hi"})
    assert r.choice == "b"
    assert r.cost == pytest.approx(100 * 1e-6 + 5 * 2e-6)


def test_request_uses_strict_enum_schema():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen.update(json.loads(req.content))
        return _resp('{"choice": "a"}')

    _llm(handler).decide("Q?", {"a": "first", "b": "second"}, {"m": "hi"})
    assert seen["url"] == "http://llm/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-llm"
    schema = seen["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["choice"]["enum"] == ["a", "b"]
    assert "first" in seen["messages"][-1]["content"]


def test_choice_outside_options_raises():
    with pytest.raises(LLMError):
        _llm(lambda req: _resp('{"choice": "zzz"}')).decide("Q", ["a", "b"], {})


def test_http_error_propagates():
    with pytest.raises(httpx.HTTPStatusError):
        _llm(lambda req: httpx.Response(500, json={"error": "down"})).decide("Q", ["a", "b"], {})
