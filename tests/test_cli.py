import argparse
import json

import pytest
from fakes import FakeS1

from reflexlayer.cli import main
from reflexlayer.log import DecisionLog
from reflexlayer.types import LLMResult, Policy, S1Prediction


def seed(db, n=200):
    log = DecisionLog(db)
    for i in range(n):
        log.record("t", {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0, 1.0), LLMResult("a", 0.01, 1.0), "llm", "a", False, f"s{i}")
    return log


def test_calibrate_then_activate(tmp_path, capsys):
    db = tmp_path / "r.db"
    seed(db)
    assert main(["--db", str(db), "calibrate"]) == 0
    assert main(["--db", str(db), "activate", "--type", "t"]) == 0
    pol = DecisionLog(db).get_policy("t")
    assert pol.mode == "active" and pol.threshold is not None
    assert "2.0%" in capsys.readouterr().out


def test_activate_refuses_without_threshold(tmp_path):
    db = tmp_path / "r.db"
    seed(db, n=5)
    assert main(["--db", str(db), "activate", "--type", "t"]) == 2
    assert DecisionLog(db).get_policy("t").mode == "shadow"


def test_activate_refuses_destructive_without_flag(tmp_path):
    db = tmp_path / "r.db"
    log = seed(db)
    log.set_policy(Policy("t", threshold=0.9, destructive=True))
    assert main(["--db", str(db), "activate", "--type", "t"]) == 2
    assert main(["--db", str(db), "activate", "--type", "t", "--allow-destructive"]) == 0
    pol = DecisionLog(db).get_policy("t")
    assert pol.mode == "active" and pol.destructive is False


def test_shadow_and_mark_destructive(tmp_path):
    db = tmp_path / "r.db"
    log = seed(db)
    log.set_policy(Policy("t", mode="active", threshold=0.9))
    assert main(["--db", str(db), "shadow", "--type", "t"]) == 0
    assert main(["--db", str(db), "mark-destructive", "--type", "t"]) == 0
    pol = DecisionLog(db).get_policy("t")
    assert pol.mode == "shadow" and pol.destructive is True


def test_replay_logs_labels_as_llm_answers(tmp_path):
    db, data = tmp_path / "r.db", tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps({"decision_type": "t", "question": "Q", "options": ["a", "b"],
                                          "context": {"m": i}, "label": "a" if i % 2 else "b", "session_id": f"s{i // 5}"}) for i in range(10)))
    assert main(["--db", str(db), "replay", str(data), "--s1-url", "x", "--s1-model", "x"], s1=FakeS1("a")) == 0
    pairs = DecisionLog(db).pairs("t")
    assert sorted(a for _, a, _ in pairs) == [False] * 5 + [True] * 5 and {s for *_, s in pairs} == {"s0", "s1"}


def test_report(tmp_path):
    db = tmp_path / "r.db"
    seed(db, n=3)
    out = tmp_path / "rep.html"
    assert main(["--db", str(db), "report", "--out", str(out)]) == 0
    assert out.exists()


def test_invalid_alpha_exits_2(tmp_path, capsys):
    db = tmp_path / "r.db"
    seed(db)
    assert main(["--db", str(db), "calibrate", "--alpha", "2"]) == 2
    assert "alpha" in capsys.readouterr().err


def test_unknown_type_is_refused_and_writes_nothing(tmp_path, capsys):
    db = tmp_path / "r.db"
    seed(db)
    for cmd in ("activate", "shadow", "mark-destructive"):
        assert main(["--db", str(db), cmd, "--type", "typo"]) == 2
        assert "known types: t" in capsys.readouterr().err
    assert DecisionLog(db)._q("SELECT * FROM policies") == []


def test_shadow_and_mark_destructive_confirm(tmp_path, capsys):
    db = tmp_path / "r.db"
    seed(db, n=1)
    main(["--db", str(db), "shadow", "--type", "t"])
    main(["--db", str(db), "mark-destructive", "--type", "t"])
    out = capsys.readouterr().out
    assert "t: shadow" in out and "marked destructive" in out


def test_replay_exits_1_when_every_s1_call_fails(tmp_path):
    db, data = tmp_path / "r.db", tmp_path / "d.jsonl"
    data.write_text(json.dumps({"decision_type": "t", "question": "Q", "options": ["a", "b"], "context": {}, "label": "a"}))
    assert main(["--db", str(db), "replay", str(data), "--s1-url", "x", "--s1-model", "x"], s1=FakeS1(fail=RuntimeError())) == 1


def test_s1_backend_selection():
    from reflexlayer.cli import _make_s1
    from reflexlayer.s1 import StrandsDeciderAdapter

    ns = argparse.Namespace(s1="strands-decider", s1_url=None, s1_model=None, s1_timeout=2.0, max_state_chars=6000)
    assert isinstance(_make_s1(ns), StrandsDeciderAdapter)
    assert _make_s1(ns).url == StrandsDeciderAdapter.DEFAULT_URL


def test_systemone_requires_url_and_model(tmp_path):
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "r.db"), "replay", "x.jsonl", "--s1-url", "x"])
