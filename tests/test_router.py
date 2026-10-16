import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fakes import FakeLLM, FakeS1, FixedRng

import reflexlayer.router as router_mod
from reflexlayer.log import DecisionLog
from reflexlayer.router import Router
from reflexlayer.s1 import S1Error
from reflexlayer.types import Policy


def make(tmp_path, s1=None, llm=None, rng=0.99, rules=None, **policy):
    log = DecisionLog(tmp_path / "r.db")
    if policy:
        log.set_policy(Policy("t", **policy))
    s1, llm = s1 or FakeS1(), llm or FakeLLM()
    return Router(s1, llm, log, rules=rules, rng=FixedRng(rng)), s1, llm, log


def test_shadow_mode_asks_both_and_returns_llm(tmp_path):
    r, s1, llm, log = make(tmp_path)
    d = r.decide("Q", ["a", "b"], {"m": 1}, "t")
    assert (d.tier, d.choice, s1.calls, llm.calls) == ("llm", "a", 1, 1)
    assert log.pairs("t") == [(0.95, True, None)]


def test_session_id_is_logged_on_every_tier(tmp_path):
    r, _, _, log = make(tmp_path, rules={"r": lambda ctx: "b"})
    r.decide("Q", ["a", "b"], {}, "t", session_id="conv-1")
    r.decide("Q", ["a", "b"], {}, "r", session_id="conv-1")
    log.set_policy(Policy("t", mode="active", threshold=0.9, sample_rate=0.0))
    r.decide("Q", ["a", "b"], {}, "t", session_id="conv-2")
    assert [row["session_id"] for row in log.rows("t") + log.rows("r")] == ["conv-1", "conv-2", "conv-1"]


def test_active_serves_s1_above_threshold(tmp_path):
    r, s1, llm, log = make(tmp_path, mode="active", threshold=0.9)
    d = r.decide("Q", ["a", "b"], {}, "t")
    assert (d.tier, d.choice, d.p, llm.calls) == ("s1", "a", 0.95, 0)
    assert log.rows("t")[0]["tier"] == "s1"


def test_active_below_threshold_goes_to_llm(tmp_path):
    r, s1, llm, log = make(tmp_path, mode="active", threshold=0.97)
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"


def test_trusted_call_sampled_for_audit(tmp_path):
    r, s1, llm, log = make(tmp_path, rng=0.0, mode="active", threshold=0.9, sample_rate=0.05)
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"
    assert log.recent_audits("t", 10) == [True]


def test_destructive_type_never_served_by_s1(tmp_path):
    r, s1, llm, log = make(tmp_path, mode="active", threshold=0.5, destructive=True)
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"


@pytest.mark.parametrize("err", [S1Error("timeout"), RuntimeError("bug in adapter")])
def test_s1_failure_falls_back_to_llm(tmp_path, err):
    r, s1, llm, log = make(tmp_path, s1=FakeS1(fail=err), mode="active", threshold=0.5)
    d = r.decide("Q", ["a", "b"], {}, "t")
    assert (d.tier, d.choice) == ("llm", "a")
    assert log.rows("t")[0]["s1_p"] is None


def test_llm_error_propagates(tmp_path):
    r, *_ = make(tmp_path, llm=FakeLLM(fail=RuntimeError("upstream 500")))
    with pytest.raises(RuntimeError):
        r.decide("Q", ["a", "b"], {}, "t")


def test_rule_short_circuits(tmp_path):
    r, s1, llm, log = make(tmp_path, rules={"t": lambda ctx: "b" if ctx.get("vip") else None})
    d = r.decide("Q", ["a", "b"], {"vip": True}, "t")
    assert (d.tier, d.choice, s1.calls, llm.calls) == ("rule", "b", 0, 0)
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"


def test_rule_returning_unknown_option_raises(tmp_path):
    r, *_ = make(tmp_path, rules={"t": lambda ctx: "zzz"})
    with pytest.raises(ValueError):
        r.decide("Q", ["a", "b"], {}, "t")


def test_drift_reverts_policy_to_shadow(tmp_path):
    r, s1, llm, log = make(tmp_path, s1=FakeS1("a"), llm=FakeLLM("b"), rng=0.0,
                           mode="active", threshold=0.9, sample_rate=1.0)
    for _ in range(20):
        r.decide("Q", ["a", "b"], {}, "t")
    pol = log.get_policy("t")
    assert pol.mode == "shadow" and pol.note.startswith("drift")


def test_no_drift_trip_below_min_audits(tmp_path):
    r, s1, llm, log = make(tmp_path, s1=FakeS1("a"), llm=FakeLLM("b"), rng=0.0,
                           mode="active", threshold=0.9, sample_rate=1.0)
    for _ in range(19):
        r.decide("Q", ["a", "b"], {}, "t")
    assert log.get_policy("t").mode == "active"


def test_reactivation_after_drift_starts_fresh(tmp_path):
    r, s1, llm, log = make(tmp_path, s1=FakeS1("a"), llm=FakeLLM("b"), rng=0.0,
                           mode="active", threshold=0.9, sample_rate=1.0)
    for _ in range(20):
        r.decide("Q", ["a", "b"], {}, "t")
    assert log.get_policy("t").mode == "shadow"
    log.set_policy(Policy("t", mode="active", threshold=0.9, sample_rate=1.0))
    r2 = Router(s1, FakeLLM("a"), log, rng=FixedRng(0.0))
    r2.decide("Q", ["a", "b"], {}, "t")
    assert log.get_policy("t").mode == "active"


def test_drift_withdraws_threshold_and_logs(tmp_path, caplog):
    r, s1, llm, log = make(tmp_path, s1=FakeS1("a"), llm=FakeLLM("b"), rng=0.0,
                           mode="active", threshold=0.9, sample_rate=1.0)
    with caplog.at_level("WARNING", logger="reflexlayer"):
        for _ in range(20):
            r.decide("Q", ["a", "b"], {}, "t")
    assert log.get_policy("t").threshold is None
    assert any("drift" in m and "'t'" in m for m in caplog.messages)


def test_single_disagreement_in_20_audits_does_not_trip(tmp_path):
    r, s1, llm, log = make(tmp_path, rng=0.0, mode="active", threshold=0.9, sample_rate=1.0)
    for i in range(20):
        llm.choice = "b" if i == 5 else "a"
        r.decide("Q", ["a", "b"], {}, "t")
    assert log.get_policy("t").mode == "active"


def test_drift_revert_keeps_concurrent_destructive_flag(tmp_path):
    class FlaggingLLM(FakeLLM):
        def decide(self, *a):
            if self.calls == 19:  # the audit that trips the wire
                p = log.get_policy("t")
                p.destructive = True  # e.g. `rflx mark-destructive` running meanwhile
                log.set_policy(p)
            return super().decide(*a)

    r, s1, llm, log = make(tmp_path, s1=FakeS1("a"), llm=FlaggingLLM("b"), rng=0.0,
                           mode="active", threshold=0.9, sample_rate=1.0)
    for _ in range(20):
        r.decide("Q", ["a", "b"], {}, "t")
    pol = log.get_policy("t")
    assert pol.mode == "shadow" and pol.destructive is True


class OverlapS1(FakeS1):
    def __init__(self, mine, other, **kw):
        super().__init__(**kw)
        self.mine, self.other, self.overlapped = mine, other, None

    def predict(self, question, options, context):
        self.mine.set()
        self.overlapped = self.other.wait(2)
        return super().predict(question, options, context)


class OverlapLLM(FakeLLM):
    def __init__(self, mine, other, **kw):
        super().__init__(**kw)
        self.mine, self.other, self.overlapped = mine, other, None

    def decide(self, question, options, context):
        self.mine.set()
        self.overlapped = self.other.wait(2)  # sequential calls would time out here
        return super().decide(question, options, context)


@pytest.mark.parametrize("policy", [{}, {"mode": "active", "threshold": 0.5, "destructive": True},
                                    {"mode": "active", "threshold": None}])
def test_unservable_call_runs_s1_alongside_llm(tmp_path, policy):
    s1_started, llm_started = threading.Event(), threading.Event()
    s1, llm = OverlapS1(s1_started, llm_started), OverlapLLM(llm_started, s1_started)
    r, _, _, log = make(tmp_path, s1=s1, llm=llm, **policy)
    d = r.decide("Q", ["a", "b"], {}, "t")
    assert s1.overlapped and llm.overlapped
    assert (d.tier, s1.calls, llm.calls) == ("llm", 1, 1)
    assert log.pairs("t") == [(0.95, True, None)]


def test_shadow_llm_error_propagates_and_abandoned_s1_is_harmless(tmp_path):
    err = RuntimeError("upstream 500")
    r, s1, llm, log = make(tmp_path, llm=FakeLLM(fail=err))
    with pytest.raises(RuntimeError) as e:
        r.decide("Q", ["a", "b"], {}, "t")
    assert e.value is err
    deadline = time.monotonic() + 2
    while s1.calls == 0 and time.monotonic() < deadline:  # the abandoned S1 call still runs to completion
        time.sleep(0.01)
    assert s1.calls == 1 and log.rows("t") == []


def test_hung_s1_with_saturated_pool_does_not_block_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(router_mod, "_S1_POOL", ThreadPoolExecutor(max_workers=1))
    release = threading.Event()

    class HungS1(FakeS1):
        timeout = 0.1

        def predict(self, *a):
            release.wait(5)
            return super().predict(*a)

    r, s1, llm, log = make(tmp_path, s1=HungS1())
    try:
        for _ in range(2):  # the second call's S1 task only queues behind the hung one
            t = time.perf_counter()
            assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"
            assert time.perf_counter() - t < 2
        assert [row["s1_p"] for row in log.rows("t")] == [None, None]
    finally:
        release.set()
        router_mod._S1_POOL.shutdown(wait=True)


def test_active_asks_s1_before_llm(tmp_path):
    order = []

    class S1(FakeS1):
        def predict(self, *a):
            order.append("s1")
            return super().predict(*a)

    class LLM(FakeLLM):
        def decide(self, *a):
            order.append("llm")
            return super().decide(*a)

    r, *_ = make(tmp_path, s1=S1(), llm=LLM(), mode="active", threshold=0.97)
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"
    assert order == ["s1", "llm"]


def test_shadow_s1_failure_is_silent(tmp_path):
    r, s1, llm, log = make(tmp_path, s1=FakeS1(fail=RuntimeError("boom")))
    assert r.decide("Q", ["a", "b"], {}, "t").tier == "llm"
    assert log.rows("t")[0]["s1_p"] is None
