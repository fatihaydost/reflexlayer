import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .types import LLMResult, Policy, S1Prediction

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, ts REAL, decision_type TEXT, context_hash TEXT, content TEXT,
  s1_choice TEXT, s1_p REAL, s1_cost REAL, s1_ms REAL,
  llm_choice TEXT, llm_cost REAL, llm_ms REAL,
  tier TEXT, final_choice TEXT, audited INTEGER, session_id TEXT
);
CREATE INDEX IF NOT EXISTS decisions_type ON decisions(decision_type, id);
CREATE TABLE IF NOT EXISTS policies (
  decision_type TEXT PRIMARY KEY, mode TEXT, threshold REAL, alpha REAL, delta REAL,
  sample_rate REAL, destructive INTEGER, note TEXT
);
"""


class DecisionLog:
    def __init__(self, path: str | Path, store_content: bool = False):
        self.store_content = store_content
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        if "session_id" not in {r["name"] for r in self._db.execute("PRAGMA table_info(decisions)")}:
            self._db.execute("ALTER TABLE decisions ADD COLUMN session_id TEXT")  # logs written before sessions existed

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def record(
        self,
        decision_type: str,
        context: dict,
        s1: S1Prediction | None,
        llm: LLMResult | None,
        tier: str,
        final_choice: str,
        audited: bool,
        session_id: str | None = None,
    ) -> None:
        # default=str: non-JSON context values (datetime, etc.) must not crash the hot path
        blob = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        row = (
            time.time(), decision_type, hashlib.sha256(blob.encode()).hexdigest(),
            blob if self.store_content else None,
            s1 and s1.choice, s1 and s1.p_max, s1 and s1.cost, s1 and s1.latency_ms,
            llm and llm.choice, llm and llm.cost, llm and llm.latency_ms,
            tier, final_choice, int(audited), session_id,
        )
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO decisions (ts, decision_type, context_hash, content, s1_choice, s1_p, s1_cost, s1_ms,"
                " llm_choice, llm_cost, llm_ms, tier, final_choice, audited, session_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )

    def get_policy(self, decision_type: str) -> Policy:
        rows = self._q("SELECT * FROM policies WHERE decision_type = ?", (decision_type,))
        if not rows:
            return Policy(decision_type)
        r = rows[0]
        return Policy(r["decision_type"], r["mode"], r["threshold"], r["alpha"], r["delta"],
                      r["sample_rate"], bool(r["destructive"]), r["note"])

    def set_policy(self, p: Policy) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?,?)",
                (p.decision_type, p.mode, p.threshold, p.alpha, p.delta, p.sample_rate, int(p.destructive), p.note),
            )

    def pairs(self, decision_type: str) -> list[tuple[float, bool, str | None]]:
        rows = self._q(
            "SELECT s1_p, s1_choice = llm_choice AS agree, session_id FROM decisions"
            " WHERE decision_type = ? AND s1_p IS NOT NULL AND llm_choice IS NOT NULL ORDER BY id",
            (decision_type,),
        )
        return [(r["s1_p"], bool(r["agree"]), r["session_id"]) for r in rows]

    def recent_audits(self, decision_type: str, n: int) -> list[bool]:
        rows = self._q(
            "SELECT s1_choice = llm_choice AS agree FROM decisions"
            " WHERE decision_type = ? AND audited = 1 ORDER BY id DESC LIMIT ?",
            (decision_type, n),
        )
        return [bool(r["agree"]) for r in rows]

    def clear_audits(self, decision_type: str) -> None:
        # drift revert: forget the bad audit window so re-activation starts fresh; pairs() is unaffected
        with self._lock, self._db:
            self._db.execute("UPDATE decisions SET audited = 0 WHERE decision_type = ?", (decision_type,))

    def decision_types(self) -> list[str]:
        return [r[0] for r in self._q("SELECT DISTINCT decision_type FROM decisions ORDER BY 1")]

    def rows(self, decision_type: str) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM decisions WHERE decision_type = ? ORDER BY id", (decision_type,))
