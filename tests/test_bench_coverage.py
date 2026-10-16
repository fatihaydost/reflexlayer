from experiments import coverage as cv


def row(choice="get_order_details", label=None, kind="user", turn="[user] hi", p=0.99, task="retail:0"):
    return {"decision_type": "tau_retail_next_action", "choice": choice, "label": label or choice, "kind": kind,
            "turn": turn, "p_max": p, "task_id": task}


def test_last_turn_walks_back_over_multiline_user_turn():
    tail = "[user] first\n[agent called] get_user_details{}\n[tool result get_user_details] {}\n[user] Yes,\nplease go ahead"
    assert cv.last_turn(tail) == ("user", "[user] Yes,\nplease go ahead")
    assert cv.last_turn("[user] a\n[tool result get_order_details] {...}")[0] == "tool"
    assert cv.last_turn("no marker")[0] == "other"


def test_rule_blocks_writes_without_affirmative_user_turn():
    w = "cancel_pending_order"
    assert not cv.blocked(row())  # reads are never blocked
    assert not cv.blocked(row(w, turn="[user] Yes, cancel it."))
    assert cv.blocked(row(w, turn="[user] I want to cancel my order."))
    assert cv.blocked(row(w, kind="tool", turn="[tool result get_order_details] yes"))
    assert cv.blocked(row(w, turn="[user] Yes, cancel it."), nowrite=True)
    assert not cv.AFFIRM.search("yesterday I ordered")


def test_types_calibrate_each_kind_separately():
    # user rows clean, tool rows 50% wrong: one shared type would mix them; split, only user rows serve
    rows = []
    for t in range(200):
        rows.append(row(task=f"retail:{t}"))
        rows.append(row(kind="tool", label="respond", task=f"retail:{t}") if t % 2 else row(kind="tool", task=f"retail:{t}"))
    s = cv.run_split(rows, 0, 0.05, 0.1, ("types",))
    assert s["sub"]["user"][0] == s["sub"]["user"][1] > 0 and s["sub"]["tool"][2] is None
    assert s["served"] == s["sub"]["user"][0] and s["wrong"] == 0
    assert cv.run_split(rows, 0, 0.05, 0.1, ())["served"] == 0


def test_configs_cover_every_lever_combination():
    names = ["+".join(c) for c in cv.configs()]
    assert {"", "rule", "types", "wsr", "rule+types", "rule+types+wsr"} <= set(names)


def test_calib_frac_moves_tasks_into_calibration():
    from evaluate import split_by_task
    rows = [row(task=f"retail:{t}") for t in range(100)]
    assert split_by_task(rows, 0) == split_by_task(rows, 0, 0.5)
    calib, test = split_by_task(rows, 0, 0.7)
    assert (len(calib), len(test)) == (70, 30)


def test_agreement_stats():
    rows = [row(), row(label="respond"), row(p=0.5, label="respond")]
    g = cv.agreement(rows)
    assert g["top1"] == 1 / 3 and g["share_ge_0.9"] == 2 / 3 and g["disagreement_ge_0.9"] == 0.5
