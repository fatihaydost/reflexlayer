import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from .calibrate import binom_cdf
from .types import Decision

DRIFT_MIN = 20
# shadow-path S1 calls run here; sized like starlette's default threadpool, which runs decide()
_S1_POOL = ThreadPoolExecutor(max_workers=40, thread_name_prefix="reflex-s1")


class Router:
    def __init__(self, s1, llm, log, rules=None, rng=None, drift_window=50):
        self.s1, self.llm, self.log = s1, llm, log
        self.rules = rules or {}
        self.rng = rng or random.Random()
        self.drift_window = drift_window

    def decide(self, question, options, context, decision_type, session_id=None) -> Decision:
        t0 = time.perf_counter()

        def ms():
            return (time.perf_counter() - t0) * 1000

        rule = self.rules.get(decision_type)
        if rule is not None:
            choice = rule(context)
            if choice is not None:
                if choice not in options:
                    raise ValueError(f"rule for {decision_type!r} returned unknown option {choice!r}")
                self.log.record(decision_type, context, None, None, "rule", choice, False, session_id)
                return Decision(choice, 1.0, "rule", 0.0, ms())

        pol = self.log.get_policy(decision_type)
        if pol.mode == "active" and not pol.destructive and pol.threshold is not None:
            # S1 must answer first: its confidence decides whether the LLM is called at all
            s1 = self._predict(question, options, context)
            trusted = s1 is not None and s1.p_max >= pol.threshold
            audited = trusted and self.rng.random() < pol.sample_rate
            if trusted and not audited:
                self.log.record(decision_type, context, s1, None, "s1", s1.choice, False, session_id)
                return Decision(s1.choice, s1.p_max, "s1", s1.cost, ms())
            llm = self.llm.decide(question, options, context)
        else:
            # S1 cannot serve this call, so shadow it alongside the LLM instead of in front of it
            fut = _S1_POOL.submit(self._predict, question, options, context)
            llm = self.llm.decide(question, options, context)
            audited = False
            try:
                # bounded even when the pool is saturated by hung S1 calls: then we lose the shadow pair, not latency
                s1 = fut.result(timeout=getattr(self.s1, "timeout", 2.0) + 0.5)
            except FutureTimeout:
                fut.cancel()  # drop it if still queued
                s1 = None

        self.log.record(decision_type, context, s1, llm, "llm", llm.choice, audited, session_id)
        if audited:
            self._check_drift(decision_type)
        return Decision(llm.choice, None, "llm", llm.cost + (s1.cost if s1 else 0.0), ms())

    def _predict(self, question, options, context):
        try:
            return self.s1.predict(question, options, context)
        except Exception:  # any S1 failure only costs the saving, never the customer's call
            return None

    def _check_drift(self, decision_type):
        pol = self.log.get_policy(decision_type)  # fresh copy: don't overwrite concurrent CLI changes
        audits = self.log.recent_audits(decision_type, self.drift_window)
        n, k = len(audits), audits.count(False)
        # one-sided binomial test at delta/10: is the disagreement rate above alpha?
        if n >= DRIFT_MIN and k > 0 and 1 - binom_cdf(k - 1, n, pol.alpha) < pol.delta / 10:
            rate = k / n
            pol.mode, pol.threshold = "shadow", None
            pol.note = f"drift: {rate:.1%} disagreement on last {n} audits"
            self.log.set_policy(pol)
            self.log.clear_audits(decision_type)
            logging.getLogger("reflexlayer").warning("drift revert for %r: %.1f%% disagreement on last %d audits",
                                                     decision_type, rate * 100, n)
