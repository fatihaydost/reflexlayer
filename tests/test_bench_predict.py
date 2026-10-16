import json

from fakes import FakeS1
from predict import compact, done_indices, run

from reflexlayer.s1 import S1Error


def item(i, task="retail:1"):
    return {"decision_type": "t", "question": "Q?", "options": {"a": "First sentence. Second sentence.", "b": "x" * 200},
            "context": {"conversation_tail": "y" * 5000}, "label": "a", "meta": {"task_id": task}}


def test_compact_shrinks_options_and_context():
    c = compact(item(0))
    assert c["options"] == {"a": "First sentence", "b": "x" * 60}
    assert len(c["context"]["conversation_tail"]) == 1800


def test_run_writes_rows_and_resume_skips_done(tmp_path):
    out = tmp_path / "p.jsonl"
    s1 = FakeS1("a", 0.9)
    run([(0, item(0)), (1, item(1))], s1, out, workers=2, budget=10.0, do_compact=False)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["i"] for r in rows) == [0, 1]
    assert rows[0]["error"] is None and rows[0]["task_id"] == "retail:1" and rows[0]["p_max"] == 0.9
    assert done_indices(out) == {0, 1}


def test_errors_are_recorded_without_secret_and_not_counted_done(tmp_path):
    out = tmp_path / "p.jsonl"
    run([(0, item(0))], FakeS1(fail=S1Error("HTTPStatusError")), out, workers=1, budget=10.0, do_compact=False)
    row = json.loads(out.read_text())
    assert row["error"] == "S1Error: HTTPStatusError" and "p_max" not in row
    assert done_indices(out) == set()


def test_budget_stops_new_requests(tmp_path):
    out = tmp_path / "p.jsonl"
    s1 = FakeS1("a", 0.9)  # FakeS1 cost is 0.0001 per call
    spent = run([(i, item(i)) for i in range(50)], s1, out, workers=1, budget=0.00025, do_compact=False)
    assert s1.calls == 3 and spent >= 0.00025


def test_resume_respects_total_budget_from_existing_file(tmp_path):
    out = tmp_path / "p.jsonl"
    out.write_text(json.dumps({"i": 0, "error": None, "cost": 0.5}) + "\n")
    s1 = FakeS1("a", 0.9)
    spent = run([(1, item(1))], s1, out, workers=1, budget=0.4, do_compact=False)
    assert s1.calls == 0 and spent == 0.0


def test_torn_last_line_is_skipped_and_next_row_parses(tmp_path):
    out = tmp_path / "p.jsonl"
    out.write_text(json.dumps({"i": 0, "error": None, "cost": 0.0}) + "\n" + '{"i": 1, "err')
    assert done_indices(out) == {0}
    run([(2, item(2))], FakeS1("a", 0.9), out, workers=1, budget=10.0, do_compact=False)
    assert done_indices(out) == {0, 2}


def test_circuit_breaker_stops_after_fifty_consecutive_errors(tmp_path):
    out = tmp_path / "p.jsonl"
    s1 = FakeS1(fail=S1Error("x"))
    run([(i, item(i)) for i in range(60)], s1, out, workers=1, budget=10.0, do_compact=False)
    assert s1.calls == 50


def test_malformed_item_becomes_error_row_and_run_continues(tmp_path):
    out = tmp_path / "p.jsonl"
    bad = {"decision_type": "t", "question": "Q?", "options": {"a": "x"}, "context": {}, "label": "a"}
    s1 = FakeS1("a", 0.9)
    run([(0, bad), (1, item(1))], s1, out, workers=1, budget=10.0, do_compact=False)
    rows = {r["i"]: r for r in (json.loads(l) for l in out.read_text().splitlines())}
    assert rows[0]["error"] == "KeyError" and rows[1]["error"] is None
    assert s1.calls == 1


def test_non_s1_exception_records_only_class_name(tmp_path):
    out = tmp_path / "p.jsonl"
    run([(0, item(0))], FakeS1(fail=RuntimeError("secret-ish text")), out, workers=1, budget=10.0, do_compact=False)
    assert json.loads(out.read_text())["error"] == "RuntimeError"


def test_error_row_cost_is_recorded_and_counted(tmp_path):
    from predict import spent_so_far
    out = tmp_path / "o.jsonl"
    run([(0, item(0))], FakeS1(fail=S1Error("not_a_label", 0.25)), str(out), 1, 10.0, False)
    row = json.loads(out.read_text().splitlines()[0])
    assert row["error"] == "S1Error: not_a_label" and row["cost"] == 0.25
    assert spent_so_far(str(out)) == 0.25


def test_require_cost_refuses_when_cost_is_zero(tmp_path):
    import pytest
    from reflexlayer.types import S1Prediction

    class Free:
        def predict(self, q, o, c):
            return S1Prediction("a", {"a": 1.0}, 1.0, 0.0, 1.0)

    with pytest.raises(SystemExit, match="cannot be enforced"):
        run([(0, item(0))], Free(), str(tmp_path / "o.jsonl"), 1, 10.0, False, require_cost=True)
