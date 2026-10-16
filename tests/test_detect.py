import json

import pytest

from reflexlayer.detect import build_response, detect, parse_choice


def enum_req(**over):
    req = {
        "model": "gpt-x",
        "messages": [{"role": "system", "content": "Triage."}, {"role": "user", "content": [{"type": "text", "text": "money back pls"}]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "triage", "schema": {
            "type": "object", "properties": {"intent": {"type": "string", "enum": ["refund", "other"]}}, "required": ["intent"]}}},
    }
    req.update(over)
    return req


def tool_req(required_params=None, tool_choice="required"):
    params = {"type": "object", "properties": {}, **({"required": required_params} if required_params else {})}
    return {
        "model": "gpt-x",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": tool_choice,
        "tools": [
            {"type": "function", "function": {"name": "handoff_billing", "description": "Billing team", "parameters": params}},
            {"type": "function", "function": {"name": "handoff_tech", "description": "Tech team", "parameters": {"type": "object", "properties": {}}}},
        ],
    }


def test_schema_enum_detected():
    d = detect(enum_req(), None)
    assert (d.kind, d.field, d.decision_type) == ("schema_enum", "intent", "schema:triage")
    assert d.options == {"refund": "refund", "other": "other"}
    assert "user: money back pls" in d.context["conversation"]


def test_header_overrides_decision_type():
    assert detect(enum_req(), "support-triage").decision_type == "support-triage"


@pytest.mark.parametrize("over", [{"stream": True}, {"n": 2}])
def test_stream_and_multi_choice_are_not_decisions(over):
    assert detect(enum_req(**over), None) is None


@pytest.mark.parametrize("prop", [
    {"type": "string"},
    {"type": "string", "enum": ["only"]},
    {"type": "integer", "enum": [1, 2]},
])
def test_non_finite_schemas_are_not_decisions(prop):
    req = enum_req()
    req["response_format"]["json_schema"]["schema"]["properties"] = {"intent": prop}
    assert detect(req, None) is None


def test_two_property_schema_is_not_a_decision():
    req = enum_req()
    req["response_format"]["json_schema"]["schema"]["properties"]["reason"] = {"type": "string"}
    assert detect(req, None) is None


def test_tool_select_detected():
    d = detect(tool_req(), None)
    assert d.kind == "tool_select"
    assert d.options == {"handoff_billing": "Billing team", "handoff_tech": "Tech team"}
    assert d.decision_type == "tools:handoff_billing,handoff_tech"


@pytest.mark.parametrize("req", [tool_req(required_params=["amount"]), tool_req(tool_choice="auto")])
def test_tools_with_args_or_optional_choice_are_not_decisions(req):
    assert detect(req, None) is None


def test_plain_chat_is_not_a_decision():
    assert detect({"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]}, None) is None


def test_build_and_parse_schema_response():
    d = detect(enum_req(), None)
    resp = build_response(d, "refund", "gpt-x")
    assert json.loads(resp["choices"][0]["message"]["content"]) == {"intent": "refund"}
    assert resp["model"] == "gpt-x" and resp["object"] == "chat.completion"
    assert parse_choice(d, resp) == "refund"


def test_build_and_parse_tool_response():
    d = detect(tool_req(), None)
    resp = build_response(d, "handoff_tech", "gpt-x")
    call = resp["choices"][0]["message"]["tool_calls"][0]
    assert call["function"] == {"name": "handoff_tech", "arguments": "{}"}
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert parse_choice(d, resp) == "handoff_tech"


@pytest.mark.parametrize("content", ["not json", '{"intent": "zzz"}', '{"intent": ["refund"]}', '{"other": "refund"}'])
def test_parse_choice_rejects_garbage(content):
    d = detect(enum_req(), None)
    assert parse_choice(d, {"choices": [{"message": {"content": content}}]}) is None


@pytest.mark.parametrize("body", [
    [],
    {"messages": "hi"},
    {"messages": [1, 2]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": None}]}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string", "enum": ["x", "y"]}}}}}},
    {"response_format": {"type": "json_schema", "json_schema": "oops"}},
    {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": ["a"]}}}},
    {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"a": "string"}}}}},
    {"tool_choice": "required", "tools": ["x", "y"]},
    {"tool_choice": "required", "tools": [{"type": "function", "function": {"name": "a", "parameters": "oops"}}, {"type": "function", "function": {"name": "b"}}]},
])
def test_malformed_bodies_are_not_decisions(body):
    assert detect(body, None) is None


def test_tool_with_only_optional_properties_is_not_a_decision():
    req = tool_req()
    req["tools"][0]["function"]["parameters"]["properties"] = {"note": {"type": "string"}}
    assert detect(req, None) is None
