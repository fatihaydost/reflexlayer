import json

import pytest
from experiments import abcd


def row(i, task, split, label="a", choice="a", p=0.99, t="next_button_after_x"):
    return {"i": i, "task_id": task, "split": split, "label": label, "choice": choice, "p_max": p, "decision_type": t}


def corpus(n_train=60, n_test=60, wrong_every=0):
    rows = []
    for k in range(n_train + n_test):
        wrong = wrong_every and k % wrong_every == 0
        rows.append(row(k, f"s{k}", "train" if k < n_train else "test", choice="b" if wrong else "a"))
    return rows


def test_official_split_has_no_session_leak():
    calib, test = abcd.official_split(corpus())
    assert not {r["task_id"] for r in calib} & {r["task_id"] for r in test}
    assert {r["split"] for r in calib} <= {"train", "dev"} and {r["split"] for r in test} == {"test"}
    with pytest.raises(ValueError, match="both"):
        abcd.official_split([row(0, "s", "train"), row(1, "s", "test")])


def test_random_split_deterministic_and_by_session():
    rows = [row(i, f"s{i // 3}", "train") for i in range(90)]
    a, b = abcd.split_by_task(rows, 3), abcd.split_by_task(rows, 3)
    assert a == b
    assert not {r["task_id"] for r in a[0]} & {r["task_id"] for r in a[1]}
    assert abcd.split_by_task(rows, 4) != a


def test_last_button_baseline_uses_calibration_rows_only():
    calib = [row(0, "c0", "train", label="x", t="T"), row(1, "c1", "train", label="x", t="T"), row(2, "c2", "train", label="y", t="U")]
    test = [row(3, "t0", "test", label="z", t="T"), row(4, "t1", "test", label="z", t="T"), row(5, "t2", "test", label="x", t="T")]
    major, last = abcd.baselines(calib)
    assert (major, last) == ("x", {"T": "x", "U": "y"})
    s = abcd.run_split(calib, test, {})
    assert s["ref"]["T"]["last_correct"] == 1  # test labels "z" never leak into the baseline
    assert s["ref"]["T"]["majority_correct"] == 1


def test_held_when_serving_counts_split_with_error_budget():
    meths = abcd.methods([0.05], 0.1)
    clean = abcd.summarize([abcd.run_split(*abcd.official_split(corpus()), meths)], meths)
    m = clean["methods"]["LTT-hb a=0.05"]["total"]
    assert (m["held"], m["served_in"]) == (1, 1) and m["coverage"] == 1.0
    # fixed thresholds carry no alpha, so no held count
    assert clean["methods"]["fixed 0.9"]["total"]["held"] is None
    dirty = corpus(wrong_every=2)  # 50% wrong everywhere: LTT abstains, fixed 0.9 serves and misses nothing it can hold
    s = abcd.summarize([abcd.run_split(*abcd.official_split(dirty), meths)], meths)
    assert s["methods"]["LTT-hb a=0.05"]["total"]["served_in"] == 0
    f = s["methods"]["fixed 0.9"]["total"]
    assert f["served_in"] == 1 and f["pooled_disagreement"] == 0.5


def test_total_coverage_is_row_weighted_across_types():
    calib = [row(i, f"c{i}", "train", t="A") for i in range(60)] + [row(100 + i, f"d{i}", "train", t="B", p=0.1) for i in range(60)]
    test = [row(200 + i, f"x{i}", "test", t="A") for i in range(30)] + [row(300 + i, f"y{i}", "test", t="B", p=0.1) for i in range(10)]
    meths = abcd.methods([0.05], 0.1)
    s = abcd.summarize([abcd.run_split(calib, test, meths)], meths)["methods"]["fixed 0.9"]
    assert s["A"]["coverage"] == 1.0 and s["B"]["coverage"] == 0.0
    assert s["total"]["coverage"] == pytest.approx(30 / 40)  # not the 0.5 mean of the two types


def test_reference_metrics_and_cli(tmp_path, capsys):
    rows = [row(i, f"s{i}", "train" if i < 60 else "test", choice="a" if i % 4 else "b", p=0.95 if i % 2 else 0.5) for i in range(120)]
    pred, dec = tmp_path / "p.jsonl", tmp_path / "d.jsonl"
    pred.write_text("".join(json.dumps({k: r[k] for k in ("i", "decision_type", "task_id", "label", "choice", "p_max")} | {"error": None}) + "\n" for r in rows))
    dec.write_text("".join(json.dumps({"meta": {"split": r["split"]}}) + "\n" for r in rows))
    out_json = tmp_path / "o.json"
    abcd.main(["--pred", str(pred), "--decisions", str(dec), "--splits", "3", "--json", str(out_json)])
    text = capsys.readouterr().out
    assert "View: official" in text and "View: random (3 splits)" in text and "LTT-wsr a=0.02" in text
    res = json.loads(out_json.read_text())["results"]
    ref = res["official"]["reference"]["total"]
    assert ref["n"] == 60 and ref["s1_top1"] == pytest.approx(45 / 60)
    assert ref["hi_n"] == 30 and ref["hi_disagreement"] == pytest.approx(ref["hi_wrong"] / 30)


def test_join_rejects_index_beyond_decisions(tmp_path):
    dec = tmp_path / "d.jsonl"
    dec.write_text(json.dumps({"meta": {"split": "train"}}) + "\n")
    with pytest.raises(ValueError, match="outside the decisions"):
        abcd.join([{"i": 5}], dec)


def test_held_session_avg_differs_from_row_pooled():
    # one 10-row session with 1 wrong plus 9 clean single-row sessions: rows 1/19 = 5.3%, session mean 1/10/10 = 1%
    calib = [row(i, f"c{i}", "train") for i in range(60)]
    test = [row(100 + i, "big", "test", choice="b" if i == 0 else "a") for i in range(10)] + [row(200 + i, f"t{i}", "test") for i in range(9)]
    m = {"fixed 0.9": (0.05, lambda p: 0.9)}
    a = abcd.summarize([abcd.run_split(calib, test, m)], m)["methods"]["fixed 0.9"]["total"]
    assert (a["held_rows"], a["held"]) == (0, 1)


def test_bound_is_passed_through():
    pairs = [(0.99, i >= 2, f"s{i}") for i in range(100)]  # 2 wrong sessions of 100: hb abstains, wsr serves
    m = abcd.methods([0.05], 0.1)
    assert m["LTT-hb a=0.05"][1](pairs) is None and m["LTT-wsr a=0.05"][1](pairs) is not None


def test_test_only_type_falls_back_to_majority_baseline():
    calib = [row(i, f"c{i}", "train", label="x", t="A") for i in range(3)]
    test = [row(10, "t0", "test", label="x", t="NEW"), row(11, "t1", "test", label="y", t="NEW")]
    s = abcd.run_split(calib, test, {})
    assert s["ref"]["NEW"]["last_correct"] == 1 == s["ref"]["NEW"]["majority_correct"]


def test_random_view_counts_held_and_served_in():
    rows = corpus(60, 40)  # all clean; random view splits 50/50 over 100 sessions
    out = abcd.evaluate(rows, [0.10], 0.1, 4)["random"]["methods"]["LTT-hb a=0.1"]["total"]
    assert out["splits"] == 4 and out["served_in"] == 4 and out["held"] == 4 and out["held_rows"] == 4
    assert out["coverage"] == 1.0
