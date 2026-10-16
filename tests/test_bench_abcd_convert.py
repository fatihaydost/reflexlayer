import convert_abcd as ca


def conv(cid, steps, flow="storewide_query"):
    """steps: (speaker, text, button|None) -> an ABCD-shaped conversation."""
    orig, dex = [], []
    for i, (sp, text, btn) in enumerate(steps, 1):
        orig.append([sp, text])
        tg = ["sub", "take_action" if btn else None, btn, ["v"] if btn else [], -1]
        dex.append({"speaker": sp, "text": text.lower(), "turn_count": i, "targets": tg})
    return {"convo_id": cid, "scenario": {"flow": flow}, "original": orig, "delexed": dex}


STEPS = [
    ("agent", "Hello", None),
    ("customer", "Hi, need help", None),
    ("action", "Account pulled up.", "pull-up-account"),
    ("agent", "Verifying", None),
    ("action", "Verified.", "verify-identity"),
]


def test_two_presses_two_rows_with_last_button_type():
    rows = list(ca.decisions({"train": [conv(7, STEPS)]}))
    assert [r["decision_type"] for r in rows] == ["next_button_after_none", "next_button_after_pull-up-account"]
    assert [r["label"] for r in rows] == ["pull-up-account", "verify-identity"]
    assert rows[0]["meta"] == {"task_id": "7", "split": "train", "flow": "storewide_query", "subflow": "sub", "turn_count": 3}


def test_context_has_earlier_press_but_not_own_button():
    r1, r2 = ca.decisions({"dev": [conv(1, STEPS)]})
    assert "[Agent pressed button: pull-up-account(v)] -> Account pulled up." in r2["context"]["dialogue"]
    assert "verify-identity" not in r2["context"]["dialogue"]
    assert "pull-up-account" not in r1["context"]["dialogue"]


def test_utterances_make_no_rows():
    assert list(ca.decisions({"train": [conv(1, STEPS[:2] + STEPS[3:4])]})) == []


def test_30_options_and_label_in_options():
    for r in ca.decisions({"train": [conv(1, STEPS)]}):
        assert len(r["options"]) == 30 and r["label"] in r["options"]


def test_merge_rare_counts_only_calibration_sessions():
    def row(cid, split, last):
        return {"decision_type": f"next_button_after_{last}", "meta": {"task_id": str(cid), "split": split}}
    rows = [row(i, "test", "x") for i in range(100)] + [row(i, "train", "x") for i in range(5)]
    rows += [row(i, "train", "y") for i in range(100, 190)] + [row(1000, "train", "y")] * 3  # 91 sessions, duplicates count once
    out = ca.merge_rare(rows, min_sessions=80, calib_splits={"train", "dev"})
    types = [r["decision_type"] for r in out]
    assert types.count("next_button_after_other") == 105
    assert types.count("next_button_after_y") == 93


def test_binary_labels_types_and_context():
    rows = list(ca.binary_rows("train", conv(3, STEPS)))
    assert [r["label"] for r in rows] == ["reply", "button", "reply", "button"]
    assert [r["decision_type"] for r in rows] == ["button_or_reply_after_none", "button_or_reply_after_customer",
                                                   "button_or_reply_after_action", "button_or_reply_after_agent"]
    assert set(rows[0]["options"]) == {"button", "reply"}
    assert "Hello" not in rows[0]["context"]["dialogue"]  # own step excluded
    d = rows[3]["context"]["dialogue"]
    assert "Verifying" in d and "Verified" not in d and "Agent pressed button: pull-up-account" in d


def test_binary_subsample_whole_sessions_deterministic():
    data = {"train": [conv(i, STEPS) for i in range(50)], "test": [conv(100 + i, STEPS) for i in range(50)]}
    a = ca.binary_decisions(data, target=30, seed=7)
    assert a == ca.binary_decisions(data, target=30, seed=7)
    assert len(a) == 32 and len(a) % 4 == 0  # 8 whole sessions of 4 steps
    ids = {r["meta"]["task_id"] for r in a}
    assert all(sum(r["meta"]["task_id"] == i for r in a) == 4 for i in ids)
