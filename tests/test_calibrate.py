import math
import random

import pytest

from reflexlayer.calibrate import WSR_C, GRID, binom_cdf, calibrate, coverage_curve, hb_pvalue, ltt_threshold, wsr_pvalue
from reflexlayer.log import DecisionLog
from reflexlayer.types import LLMResult, Policy, S1Prediction


def test_binom_cdf_known_values():
    assert binom_cdf(0, 10, 0.1) == pytest.approx(0.9 ** 10)
    assert binom_cdf(10, 10, 0.3) == pytest.approx(1.0)
    assert binom_cdf(2, 20, 0.05) == pytest.approx(sum(math.comb(20, i) * 0.05 ** i * 0.95 ** (20 - i) for i in range(3)))


def sess(pairs):
    """Every row its own session."""
    return [(p, a, f"s{i}") for i, (p, a) in enumerate(pairs)]


def test_hb_pvalue_known_values():
    # zero observed risk: the Hoeffding arm is (1 - alpha)^n; Bentkus is e times that
    assert hb_pvalue(0.0, 45, 0.05) == pytest.approx(0.95 ** 45)
    assert hb_pvalue(0.0, 44, 0.05) > 0.1 >= hb_pvalue(0.0, 45, 0.05)  # the ~45-session activation floor
    assert hb_pvalue(0.05, 100, 0.05) == 1.0  # risk at the budget is never evidence
    kl = 0.02 * math.log(0.02 / 0.05) + 0.98 * math.log(0.98 / 0.95)
    assert hb_pvalue(0.02, 400, 0.05) == pytest.approx(min(math.exp(-400 * kl), math.e * binom_cdf(8, 400, 0.05)))


def test_grid_is_fixed_and_descending():
    assert GRID[0] == 1.0 and GRID[1] == 0.99 and GRID[-1] == 0.5 and GRID == sorted(GRID, reverse=True)


def test_threshold_found_when_confident_predictions_agree():
    pairs = [(0.95, True)] * 400 + [(0.6, False)] * 100 + [(0.6, True)] * 100
    lam = ltt_threshold(sess(pairs), alpha=0.02, delta=0.05)
    assert lam is not None and 0.6 < lam <= 0.95


def test_none_when_too_few_samples():
    assert ltt_threshold(sess([(0.9, True)] * 20), alpha=0.02, delta=0.05) is None


def test_rows_count_once_per_session():
    many_rows_few_sessions = [(0.9, True, f"s{i % 10}") for i in range(1000)]
    assert ltt_threshold(many_rows_few_sessions, alpha=0.1, delta=0.1) is None
    assert ltt_threshold(sess([(0.9, True)] * 1000), alpha=0.1, delta=0.1) is not None


def test_rows_without_session_are_ignored():
    assert ltt_threshold([(0.9, True, None)] * 1000, alpha=0.1, delta=0.1) is None


def test_session_risk_is_averaged_per_session():
    # 60 clean one-row sessions + 1 session with 500 wrong rows: row rate ~89%, session average 1/61
    pairs = [(0.9, True, f"s{i}") for i in range(200)] + [(0.9, False, "bad")] * 500
    assert ltt_threshold(pairs, alpha=0.05, delta=0.1) is not None


def test_one_rung_serves_only_certain_predictions():
    pairs = sess([(1.0, True)] * 100 + [(0.99, False)] * 100)
    assert ltt_threshold(pairs, alpha=0.05, delta=0.1) == 1.0


def test_none_when_budget_not_met():
    assert ltt_threshold(sess([(0.9, True)] * 200 + [(0.9, False)] * 20), alpha=0.02, delta=0.05) is None


def test_guarantee_holds_on_calibrated_synthetic_data():
    # p ~ U(0.5, 1), P(agree) = p, scores stored rounded to 2 decimals (like Jev):
    # rounded >= lam means true p >= lam - 0.005, so the true disagreement above lam is (1 - lam + 0.005) / 2
    alpha, delta, trials, violations, found = 0.1, 0.1, 100, 0, 0
    rng = random.Random(7)
    for _ in range(trials):
        pairs = []
        for _ in range(1000):
            p = rng.uniform(0.5, 1.0)
            pairs.append((round(p, 2), rng.random() < p, len(pairs)))
        lam = ltt_threshold(pairs, alpha, delta)
        found += lam is not None
        if lam is not None and (1 - lam + 0.005) / 2 > alpha:
            violations += 1
    assert violations <= delta * trials + 5
    assert found >= trials / 2  # an always-None stub must not pass


def test_wsr_pvalue_known_values():
    assert wsr_pvalue([], 0.05, 0.1) == 1.0
    # zero losses: bets are capped at c / (1 - alpha), so capital is at most (1 + c * alpha / (1 - alpha))^n
    assert wsr_pvalue([0.0] * 50, 0.05, 0.1) >= (1 + WSR_C * 0.05 / 0.95) ** -50
    assert wsr_pvalue([0.0] * 80, 0.05, 0.1) < 0.1 < wsr_pvalue([0.0] * 50, 0.05, 0.1)
    assert wsr_pvalue([0.05] * 100, 0.05, 0.1) == 1.0  # losses at the budget never grow capital
    assert wsr_pvalue([1.0] * 10, 0.05, 0.1) == 1.0


def test_wsr_beats_hb_on_low_variance_losses():
    rng = random.Random(1)
    xs = [min(1.0, max(0.0, rng.gauss(0.04, 0.02))) for _ in range(55)]
    assert wsr_pvalue(xs, 0.1, 0.1) < 0.1 < hb_pvalue(sum(xs) / 55, 55, 0.1)


def test_wsr_id_correlated_losses_not_rejected_after_hashing():
    # mean loss 0.08 > alpha 0.05, all losses at the end of id order: id order would give p ~ 0.03
    pairs = [(0.9, i < 92, f"s{i:03d}") for i in range(100)]
    assert ltt_threshold(pairs, 0.05, 0.1, bound="wsr") is None
    assert wsr_pvalue([0.0] * 92 + [1.0] * 8, 0.05, 0.1) < 0.1  # the id-ordered input is what hashing guards against


@pytest.mark.parametrize("alpha,delta", [(0, 0.1), (1, 0.1), (0.05, 0), (0.05, 1), (-1, 0.1), (0.05, 2)])
def test_ltt_threshold_validates_alpha_delta(alpha, delta):
    with pytest.raises(ValueError):
        ltt_threshold(sess([(0.9, True)] * 100), alpha, delta)


def test_unknown_bound_rejected():
    with pytest.raises(ValueError):
        ltt_threshold(sess([(0.9, True)] * 100), 0.1, 0.1, bound="x")


def test_wsr_threshold_found_and_guarantee_holds_on_synthetic_sessions():
    # 60 sessions of 20 rows; P(agree) = p per row, so session rates have low variance (WSR's regime)
    alpha, delta, trials, violations, found = 0.1, 0.1, 100, 0, 0
    rng = random.Random(11)
    for _ in range(trials):
        pairs = []
        for s in range(60):
            for _ in range(20):
                p = rng.uniform(0.5, 1.0)
                pairs.append((round(p, 2), rng.random() < p, f"s{s}"))
        lam = ltt_threshold(pairs, alpha, delta, bound="wsr")
        found += lam is not None
        if lam is not None and (1 - lam + 0.005) / 2 > alpha:
            violations += 1
    assert violations <= delta * trials + 5
    assert found >= trials / 2


def test_coverage_curve():
    curve = coverage_curve([(0.95, True), (0.95, False), (0.6, True), (0.6, True)], [0.9, 0.5, 0.99])
    assert curve == [(0.9, 0.5, 0.5), (0.5, 1.0, 0.25), (0.99, 0.0, None)]


def test_calibrate_writes_threshold_and_keeps_mode(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    for i in range(200):
        log.record("t", {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0, 1.0), LLMResult("a", 0.01, 1.0), "llm", "a", False, f"s{i}")
    pol = calibrate(log, "t", alpha=0.03)
    assert pol.threshold is not None and pol.mode == "shadow" and pol.alpha == 0.03
    assert log.get_policy("t") == pol


def test_calibrate_failure_demotes_active_policy(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    log.set_policy(Policy("t", mode="active", threshold=0.9))
    pol = calibrate(log, "t")
    assert (pol.mode, pol.threshold) == ("shadow", None) and pol.note


def test_calibrate_without_session_ids_says_so(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    for _ in range(200):
        log.record("t", {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0, 1.0), LLMResult("a", 0.01, 1.0), "llm", "a", False)
    pol = calibrate(log, "t")
    assert pol.threshold is None and "session_id" in pol.note


@pytest.mark.parametrize("kw", [{"alpha": 2}, {"alpha": 0}, {"delta": 95}, {"delta": 1.0}])
def test_calibrate_rejects_invalid_alpha_delta(tmp_path, kw):
    log = DecisionLog(tmp_path / "r.db")
    with pytest.raises(ValueError):
        calibrate(log, "t", **kw)


def test_recalibrating_active_policy_demotes_to_shadow(tmp_path):
    log = DecisionLog(tmp_path / "r.db")
    for i in range(200):
        log.record("t", {}, S1Prediction("a", {"a": 0.95}, 0.95, 0.0, 1.0), LLMResult("a", 0.01, 1.0), "llm", "a", False, f"s{i}")
    log.set_policy(Policy("t", mode="active", threshold=0.9))
    pol = calibrate(log, "t")
    assert pol.threshold not in (None, 0.9)
    assert pol.mode == "shadow" and pol.note == "recalibrated; run `rflx activate` to approve"
    assert log.get_policy("t").mode == "shadow"
