from reflexlayer.log import DecisionLog
from reflexlayer.report import render_report, write_report
from reflexlayer.types import LLMResult, Policy, S1Prediction


def seed(log, t, n=40):
    for i in range(n):
        log.record(t, {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0001, 5.0), LLMResult("a" if i % 10 else "b", 0.01, 400.0), "llm", "a", False)
    log.record(t, {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0001, 5.0), None, "s1", "a", False)


def test_report_shows_type_policy_and_curve(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    seed(log, "triage")
    log.set_policy(Policy("triage", threshold=0.93))
    html = render_report(log)
    assert "triage" in html and "shadow" in html and "0.93" in html
    assert "<svg" in html and "90.0%" in html  # agreement 36/40


def test_report_escapes_decision_type(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    seed(log, "<script>x</script>", n=2)
    html = render_report(log)
    assert "<script>x" not in html and "&lt;script&gt;x" in html


def test_empty_log(tmp_path):
    assert "No decisions yet" in render_report(DecisionLog(tmp_path / "r.db"))


def test_write_report(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    seed(log, "t", n=3)
    write_report(log, tmp_path / "out.html")
    assert (tmp_path / "out.html").read_text().startswith("<!doctype html>")


def test_served_share_uses_traffic_not_pairs(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    s1 = S1Prediction("a", {"a": 0.95}, 0.95, 0.0001, 5.0)
    llm = LLMResult("a", 0.01, 400.0)
    for _ in range(10):
        log.record("t", {}, s1, None, "s1", "a", False)
    log.record("t", {}, s1, llm, "llm", "a", True)
    for _ in range(9):
        log.record("t", {}, s1, llm, "llm", "a", False)
    # S1 trusted share = (10 served + 1 audited) / 20 decisions with an S1 score
    assert "55.0%" in render_report(log)


def test_row_is_labelled_trusted_share(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    seed(log, "t", n=2)
    assert "S1 trusted share" in render_report(log) and "S1 share of traffic" not in render_report(log)
