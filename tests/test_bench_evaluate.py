import json

import pytest

import evaluate as ev
from evaluate import evaluate, load, render, served, split_by_task


def rows_synthetic(n_tasks=120, per_task=30):  # calibration half must clear MIN_SESSIONS
    out, i = [], 0
    for t in range(n_tasks):
        for k in range(per_task):
            confident = k % 3 != 0  # two thirds confident and right, one third unsure and 50% wrong
            out.append({"i": i, "decision_type": "t", "task_id": f"d:{t}", "label": "a",
                        "choice": "a" if confident or k % 2 else "b", "p_max": 0.99 if confident else 0.6, "error": None})
            i += 1
    return out


def test_split_has_no_task_leak_and_keeps_all_rows():
    rows = rows_synthetic()
    calib, test = split_by_task(rows, seed=3)
    assert {r["task_id"] for r in calib}.isdisjoint({r["task_id"] for r in test})
    assert len(calib) + len(test) == len(rows)
    assert split_by_task(rows, seed=3) == (calib, test)


def test_served():
    pairs = [(0.99, True, "s"), (0.99, False, "s"), (0.5, True, "s"), (0.5, True, "s")]
    assert served(pairs, None) == (0.0, None)
    assert served(pairs, 0.9) == (0.5, 0.5)
    assert served(pairs, 1.0) == (0.0, None)


def test_load_dedupes_by_i_and_counts_errors(tmp_path):
    p = tmp_path / "p.jsonl"
    lines = [{"i": 0, "error": "S1Error: x"}, {"i": 0, "decision_type": "t", "task_id": "d:0", "label": "a", "choice": "a", "p_max": 0.9, "error": None},
             {"i": 1, "error": "S1Error: x"}]
    p.write_text("\n".join(json.dumps(l) for l in lines))
    rows, n_errors = load(p)
    assert [r["i"] for r in rows] == [0] and n_errors == 1


def test_load_skips_torn_lines(tmp_path):
    p = tmp_path / "p.jsonl"
    good = {"i": 0, "decision_type": "t", "task_id": "d:0", "label": "a", "choice": "a", "p_max": 0.9, "error": None}
    p.write_text(json.dumps(good) + '\n{"i": 1, "decisi\n')  # second line torn by a kill mid-write
    rows, n_errors = load(p)
    assert [r["i"] for r in rows] == [0] and n_errors == 0


def test_evaluate_fixed_and_ltt():
    res = evaluate(rows_synthetic(), alphas=[0.05], delta=0.1, splits=10)["t"]
    fixed = res["methods"]["fixed 0.9"]
    assert abs(fixed["coverage_mean"] - 2 / 3) < 0.05 and fixed["pooled_disagreement"] == 0.0
    for name in ("LTT session a=0.05", "LTT row a=0.05"):
        ltt = res["methods"][name]
        assert ltt["held_when_serving"] == 1.0 and ltt["held_count"] == ltt["serving_count"] > 0 and ltt["coverage_mean"] > 0.6
    assert fixed["served_in_share"] == 1.0
    assert res["tasks"] == 120 and res["n"] == 3600


def test_session_ltt_rejects_correlated_errors_that_row_ltt_accepts():
    # 100 tasks x 30 confident rows; 3 tasks are wrong on every row. Row level: 90/3000 = 3% passes alpha=0.05;
    # session level: 3 of 100 sessions fully wrong is not enough evidence. Without the bad tasks both pass.
    from reflexlayer.calibrate import ltt_threshold
    clean = [dict(r, choice="a", p_max=0.99) for r in rows_synthetic(n_tasks=100)]
    bad = [dict(r, choice="b") if int(r["task_id"][2:]) < 3 else r for r in clean]
    assert ltt_threshold(ev.pairs(clean), 0.05, 0.1) is not None
    assert ltt_threshold(ev.pairs(bad, "i"), 0.05, 0.1) is not None
    assert ltt_threshold(ev.pairs(bad), 0.05, 0.1) is None


def test_no_threshold_counts_as_zero_coverage():
    rows = [dict(r, choice="b") for r in rows_synthetic()]  # everything wrong
    ltt = evaluate(rows, alphas=[0.05], delta=0.1, splits=5)["t"]["methods"]["LTT session a=0.05"]
    assert ltt["no_threshold_share"] == 1.0 and ltt["coverage_mean"] == 0.0 and ltt["held_or_abstained_share"] == 1.0
    assert ltt["held_when_serving"] is None and ltt["serving_count"] == 0 and ltt["pooled_disagreement"] is None
    assert ltt["served_in_share"] == 0.0


def test_render_contains_methods():
    md = render({"jev": evaluate(rows_synthetic(), alphas=[0.05], delta=0.1, splits=3)})
    assert "fixed 0.9" in md and "LTT session a=0.05" in md and "LTT row a=0.05" in md and "jev" in md
    assert "served in" in md and "pooled disagreement" in md and "p10–p90" in md and "held when serving" in md and "held, task-avg" in md
    assert "test disagreement ≤ α" not in md and "majority baseline ('a')" in md and "1.00..0.50" in md
    assert "p_max p10/p50/p90" in md and "≥0.50:" in md and "\nNotes\n" in md


def test_load_skips_rows_without_i(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"error": null}\n{"i": 0, "decision_type": "t", "task_id": "d:0", "label": "a", "choice": "a", "p_max": 0.9, "error": null}\n')
    rows, n_errors = load(p)
    assert [r["i"] for r in rows] == [0] and n_errors == 0


def test_quantiles_and_grid_floor_share():
    p = [0.1, 0.2, 0.3, 0.4, 0.6]
    rows = [{"i": k, "decision_type": "t", "task_id": f"d:{k}", "label": "a", "choice": "a", "p_max": x, "error": None}
            for k, x in enumerate(p)]
    res = evaluate(rows, alphas=[0.05], delta=0.1, splits=1)["t"]
    q = res["p_max_quantiles"]
    assert q["p10"] == pytest.approx(0.14) and q["p50"] == pytest.approx(0.3) and q["p90"] == pytest.approx(0.52)
    assert res["share_ge_grid_floor"] == pytest.approx(0.2)
    assert res["delta"] == 0.1 and res["splits"] == 1
    assert "Notes" in render({"m": {"t": res}})


def test_held_when_serving_none_hold(monkeypatch):
    monkeypatch.setattr(ev, "ltt_threshold", lambda pairs, a, d: 0.9)  # force serving with 1/3 wrong
    rows = [dict(r, choice="b" if r["i"] % 3 == 0 else "a", p_max=0.99) for r in rows_synthetic()]
    m = evaluate(rows, alphas=[0.05], delta=0.1, splits=6)["t"]["methods"]["LTT session a=0.05"]
    assert (m["held_count"], m["serving_count"]) == (0, 6) and m["held_when_serving"] == 0.0
    assert m["held_or_abstained_share"] == 0.0


def test_pooled_differs_from_mean():
    # task A: 1 served row, wrong; task B: 9 served rows, right. Test half is A or B depending on the split.
    rows = [{"i": 0, "decision_type": "t", "task_id": "A", "label": "a", "choice": "b", "p_max": 0.99, "error": None}]
    rows += [{"i": k, "decision_type": "t", "task_id": "B", "label": "a", "choice": "a", "p_max": 0.99, "error": None}
             for k in range(1, 10)]
    m = evaluate(rows, alphas=[], delta=0.1, splits=20)["t"]["methods"]["fixed 0.9"]
    assert m["disagreement_p10"] == 0.0 and m["disagreement_p90"] == 1.0
    assert abs(m["pooled_disagreement"] - m["disagreement_mean_unweighted"]) > 0.2


def test_held_task_avg_differs_from_pooled(monkeypatch):
    # one test task with 29 right rows and 1 wrong vs many one-row tasks: pooled and task-avg disagreement differ
    monkeypatch.setattr(ev, "ltt_threshold", lambda pairs, a, d: 0.9)
    test = [(0.99, k != 0, "big") for k in range(30)] + [(0.99, False, "small")]
    assert ev.served_task_avg(test, 0.9) == pytest.approx((1 / 30 + 1) / 2)
    assert ev.served_task_avg(test, None) is None and ev.served_task_avg(test, 1.0) is None
    rows = [dict(r, choice="b" if r["i"] % 3 == 0 else "a", p_max=0.99) for r in rows_synthetic()]
    m = evaluate(rows, alphas=[0.05], delta=0.1, splits=2)["t"]["methods"]["LTT session a=0.05"]
    assert (m["held_task_avg_count"], m["serving_count"]) == (0, 2)  # every task is 1/3 wrong


def test_majority_baseline():
    res = evaluate(rows_synthetic(), alphas=[0.05], delta=0.1, splits=1)["t"]
    assert res["majority_label"] == "a" and res["majority_share"] == 1.0


def test_load_skips_non_dict_lines(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('[1, 2]\n"x"\n42\n{"i": 0, "decision_type": "t", "task_id": "d:0", "label": "a", "choice": "a", "p_max": 0.9, "error": null}\n')
    rows, n_errors = load(p)
    assert [r["i"] for r in rows] == [0] and n_errors == 0
