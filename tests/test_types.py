from reflexlayer.types import Decision, LLMResult, Policy, S1Prediction


def test_policy_defaults_are_safe():
    p = Policy("triage")
    assert (p.mode, p.threshold, p.destructive) == ("shadow", None, False)
    assert (p.alpha, p.delta, p.sample_rate) == (0.02, 0.05, 0.05)


def test_value_types():
    s = S1Prediction("a", {"a": 0.9, "b": 0.1}, 0.9, 0.0, 3.0)
    assert s.p_max == 0.9
    assert LLMResult("a", 0.01, 400.0).choice == "a"
    assert Decision("a", None, "llm", 0.01, 400.0).tier == "llm"
