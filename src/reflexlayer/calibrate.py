import hashlib
import math
from collections import defaultdict

GRID = [round(1 - i / 100, 2) for i in range(51)]  # 1.00 .. 0.50, fixed before seeing data
MIN_SESSIONS = 30


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binom(n, p) <= k), summed in log space so large n does not overflow."""
    if k >= n or p <= 0:
        return 1.0
    if p >= 1:
        return 0.0
    lp, lq, lnf = math.log(p), math.log1p(-p), math.lgamma(n + 1)
    total = sum(math.exp(lnf - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1))
    return min(1.0, total)


def hb_pvalue(risk: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: mean > alpha, given the mean `risk` of n independent [0, 1] losses
    (Angelopoulos et al., Learn then Test, 2021)."""
    a = min(risk, alpha)
    kl = (a * math.log(a / alpha) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - alpha))
    # 1e-9: a float sum of per-session rates must not round an integer n*risk up to the next count
    bentkus = math.e * binom_cdf(math.ceil(n * risk - 1e-9), n, alpha)
    return min(1.0, math.exp(-n * kl), bentkus)


SESSION_ORDER_KEY = "reflex-wsr-order-v1"  # fixed key: session order is a hash of the id, not of the losses
WSR_C = 0.75  # bet truncation from Waudby-Smith & Ramdas' experiments, fixed a priori


def wsr_pvalue(losses, alpha, delta, c=WSR_C):
    """Betting p-value for H0: mean >= alpha over independent [0, 1] losses, taken in the given order
    (Waudby-Smith & Ramdas 2023, hedged capital with a predictable plug-in bet).

    Capital K_t = prod(1 + l_i (alpha - x_i)) is a nonnegative supermartingale under H0, so by Ville
    P(max_t K_t >= 1/p) <= p. Bets use only earlier losses; the order must not depend on the losses.
    Variance update: the plug-in variance at step t uses the mean of x_1..x_(t-1) (predictable, so valid), which is
    not the paper's exact recipe. Valid only when the order is independent of the losses."""
    n = len(losses)
    if n == 0:
        return 1.0
    cap, k, best = c / (1 - alpha), 1.0, 1.0
    s, ss = 0.5, 0.25  # priors of WSR's plug-in mean (1/2) and variance (1/4)
    for t, x in enumerate(losses, 1):
        mu, var = s / t, ss / t
        lam = min(math.sqrt(2 * math.log(1 / delta) / (n * var)), cap)  # fixed-n tuning of the plug-in bet
        k *= 1 + lam * (alpha - x)
        best = max(best, k)
        ss += (x - mu) ** 2
        s += x
    return min(1.0, 1 / best)


def _order_key(sid):
    return hashlib.sha256((SESSION_ORDER_KEY + str(sid)).encode()).hexdigest()


def ltt_threshold(pairs, alpha, delta, min_sessions=MIN_SESSIONS, bound="hb"):
    """Learn-then-Test, fixed sequence over GRID: lowest threshold whose session-averaged disagreement
    with the LLM is <= alpha with probability >= 1 - delta; None if no threshold qualifies.

    pairs are (s1_p, agree, session_id). Sessions are the independent unit: each session's disagreement
    rate among its rows above the threshold counts once, so correlated decisions inside one conversation
    cannot pass as independent evidence. Rows without a session_id are ignored.

    bound="hb" tests with Hoeffding-Bentkus; bound="wsr" (opt-in, not exposed by calibrate()) with the betting
    p-value, feeding sessions in a keyed-hash order of their ids. Assumption: valid when the hashed order is
    independent of the losses; ids must not be chosen adversarially. delta is validated here too."""
    if not (0 < alpha < 1 and 0 < delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got alpha={alpha}, delta={delta}")
    if bound not in ("hb", "wsr"):
        raise ValueError(f"bound must be 'hb' or 'wsr', got {bound!r}")
    best = None
    for lam in GRID:
        sessions = defaultdict(list)
        for p, agree, sid in pairs:
            if sid is not None and p >= lam:
                sessions[sid].append(agree)
        n = len(sessions)
        if n < min_sessions:
            continue  # skipping depends only on S1 scores and session ids, not on labels, so the test stays valid
        losses = [1 - sum(s) / len(s) for _, s in sorted(sessions.items(), key=lambda kv: _order_key(kv[0]))]
        pval = hb_pvalue(sum(losses) / n, n, alpha) if bound == "hb" else wsr_pvalue(losses, alpha, delta)
        if pval > delta:
            break
        best = lam
    return best


def coverage_curve(pairs, thresholds):
    total = len(pairs)
    out = []
    for lam in thresholds:
        selected = [agree for p, agree, *_ in pairs if p >= lam]
        cov = len(selected) / total if total else 0.0
        dis = (len(selected) - sum(selected)) / len(selected) if selected else None
        out.append((lam, cov, dis))
    return out


def calibrate(log, decision_type, alpha=None, delta=None):
    pol = log.get_policy(decision_type)
    was_active, old = pol.mode == "active", (pol.threshold, pol.alpha, pol.delta)
    if alpha is not None:
        pol.alpha = alpha
    if delta is not None:
        pol.delta = delta
    if not (0 < pol.alpha < 1 and 0 < pol.delta < 1):
        raise ValueError(f"alpha and delta must be in (0, 1), got alpha={pol.alpha}, delta={pol.delta}")
    pairs = log.pairs(decision_type)
    pol.threshold = ltt_threshold(pairs, pol.alpha, pol.delta)
    if (pol.threshold, pol.alpha, pol.delta) != old and was_active and pol.threshold is not None:
        pol.mode = "shadow"  # a changed budget needs the customer's approval again
        pol.note = "recalibrated; run `rflx activate` to approve"
    elif pol.threshold is None:
        pol.mode = "shadow"
        n = len({sid for *_, sid in pairs if sid is not None})
        pol.note = (f"no threshold meets <= {pol.alpha:.1%} session-averaged disagreement at {1 - pol.delta:.0%} "
                    f"confidence yet ({n} sessions logged)" if n else "no session_id logged; pass session_id to calibrate")
    else:
        pol.note = ""
    log.set_policy(pol)
    return pol
