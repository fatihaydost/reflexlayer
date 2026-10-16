from convert_tau import CTX_CAP, MARKER, build_context


def test_short_history_unchanged():
    turns = ["[user] hi", "[agent called] get_user_details{}", "[tool result x] ok"]
    assert build_context(turns) == "\n".join(turns)


def test_long_history_keeps_first_user_turn_and_whole_tail():
    first = "[user] I want to cancel order #W1"
    middle = [f"[tool result t] {'x' * 900}" for _ in range(10)]
    turns = [first] + middle + ["[assistant] Sure, let me check."]
    ctx = build_context(turns)
    assert len(ctx) <= CTX_CAP
    assert ctx.startswith(first)
    assert f"\n{MARKER}\n" in ctx
    tail = ctx.split(f"\n{MARKER}\n", 1)[1].split("\n")
    assert tail[-1] == "[assistant] Sure, let me check."
    assert all(line in turns for line in tail)  # only whole turns, no mid-turn cut


def test_oversized_last_turn_keeps_its_end():
    turns = ["[user] goal", "[tool result t] " + "y" * (CTX_CAP * 2) + "END"]
    ctx = build_context(turns)
    assert len(ctx) <= CTX_CAP
    assert ctx.startswith("[user] goal")
    assert ctx.endswith("END")


def test_digest_adds_policy_and_lost_last_agent_message():
    from convert_tau import with_digest
    ctx = {"conversation_tail": "[user] hi\n[assistant] Shall I cancel it?"}
    assert with_digest(ctx, "D", "[assistant] Shall I cancel it?") == {**ctx, "policy_digest": "D"}
    out = with_digest({"conversation_tail": "[user] hi"}, "D", "[assistant] old summary")
    assert out["last_agent_message"] == "[assistant] old summary" and out["policy_digest"] == "D"
    assert with_digest({"conversation_tail": "x"}, "D", None) == {"conversation_tail": "x", "policy_digest": "D"}
