import threading
from datetime import datetime

from reflexlayer.log import DecisionLog
from reflexlayer.types import LLMResult, Policy, S1Prediction

S1 = S1Prediction("a", {"a": 0.9, "b": 0.1}, 0.9, 0.0001, 5.0)
LLM_A = LLMResult("a", 0.01, 400.0)
LLM_B = LLMResult("b", 0.01, 400.0)


def test_content_is_hashed_not_stored_by_default(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    log.record("t", {"msg": "secret text"}, S1, LLM_A, "llm", "a", False)
    row = log.rows("t")[0]
    assert row["content"] is None and len(row["context_hash"]) == 64
    assert b"secret text" not in (tmp_path / "r.db").read_bytes()


def test_content_stored_when_opted_in(tmp_path):
    log = DecisionLog(tmp_path / "r.db", store_content=True)
    log.record("t", {"msg": "hello"}, None, LLM_A, "llm", "a", False)
    assert "hello" in log.rows("t")[0]["content"]


def test_non_json_context_does_not_crash(tmp_path):
    log = DecisionLog(tmp_path / "r.db", store_content=True)
    log.record("t", {"when": datetime(2026, 1, 1)}, None, LLM_A, "llm", "a", False)
    assert "2026-01-01" in log.rows("t")[0]["content"]


def test_policy_default_and_roundtrip(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    assert log.get_policy("t") == Policy("t")
    p = Policy("t", mode="active", threshold=0.91, alpha=0.01, destructive=True, note="x")
    log.set_policy(p)
    assert log.get_policy("t") == p


def test_pairs_only_rows_with_both_tiers(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    log.record("t", {}, S1, LLM_A, "llm", "a", False, "s1")
    log.record("t", {}, S1, LLM_B, "llm", "b", False)
    log.record("t", {}, S1, None, "s1", "a", False)
    log.record("t", {}, None, LLM_A, "llm", "a", False)
    assert log.pairs("t") == [(0.9, True, "s1"), (0.9, False, None)]


def test_old_log_gains_session_column(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, ts REAL, decision_type TEXT, context_hash TEXT,"
               " content TEXT, s1_choice TEXT, s1_p REAL, s1_cost REAL, s1_ms REAL, llm_choice TEXT, llm_cost REAL,"
               " llm_ms REAL, tier TEXT, final_choice TEXT, audited INTEGER)")
    db.execute("INSERT INTO decisions (decision_type, s1_choice, s1_p, llm_choice) VALUES ('t', 'a', 0.9, 'a')")
    db.commit()
    db.close()
    log = DecisionLog(tmp_path / "old.db")
    log.record("t", {}, S1, LLM_B, "llm", "b", False, "s1")
    assert log.pairs("t") == [(0.9, True, None), (0.9, False, "s1")]
    DecisionLog(tmp_path / "old.db")  # reopening a migrated log is a no-op


def test_recent_audits_newest_first(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    log.record("t", {}, S1, LLM_A, "llm", "a", True)
    log.record("t", {}, S1, LLM_B, "llm", "b", False)
    log.record("t", {}, S1, LLM_B, "llm", "b", True)
    assert log.recent_audits("t", 10) == [False, True]
    assert log.decision_types() == ["t"]


def test_clear_audits_keeps_pairs(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    log.record("t", {}, S1, LLM_A, "llm", "a", True)
    log.record("t", {}, S1, LLM_B, "llm", "b", True)
    log.clear_audits("t")
    assert log.recent_audits("t", 10) == []
    assert log.pairs("t") == [(0.9, True, None), (0.9, False, None)]


def test_concurrent_writes(tmp_path):
    log = DecisionLog(tmp_path / "r.db")

    def work():
        for _ in range(25):
            log.record("t", {}, S1, LLM_A, "llm", "a", False)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(log.rows("t")) == 200
