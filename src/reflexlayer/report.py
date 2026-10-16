from html import escape
from pathlib import Path
from statistics import median

from .calibrate import GRID, coverage_curve

CSS = """body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fff}
section{border-top:1px solid #ddd;padding:1rem 0}table{border-collapse:collapse}td{padding:2px 12px 2px 0}
.hint{font-size:12px;color:#666}
@media (prefers-color-scheme:dark){body{color:#eee;background:#141414}section{border-color:#333}.hint{color:#999}}"""


def _pct(x):
    return "–" if x is None else f"{x:.1%}"


def _ms(xs):
    xs = [x for x in xs if x is not None]
    return f"{median(xs):.0f} ms" if xs else "–"


def _svg(curve, alpha):
    pts = [(cov, dis) for _, cov, dis in curve if dis is not None]
    top = max([alpha * 2, 0.05] + [d for _, d in pts])

    def xy(cov, dis):
        return f"{20 + cov * 360:.1f},{180 - dis / top * 160:.1f}"

    line = " ".join(xy(c, d) for c, d in pts)
    y_alpha = 180 - alpha / top * 160
    return (f'<svg viewBox="0 0 400 200" width="400" height="200" role="img" aria-label="disagreement vs coverage">'
            f'<line x1="20" y1="180" x2="380" y2="180" stroke="currentColor" stroke-opacity=".4"/>'
            f'<line x1="20" y1="{y_alpha:.1f}" x2="380" y2="{y_alpha:.1f}" stroke="#c0392b" stroke-dasharray="4"/>'
            f'<polyline points="{line}" fill="none" stroke="#2e86de" stroke-width="2"/>'
            f'<text x="20" y="196" font-size="11" fill="currentColor">coverage →</text>'
            f'<text x="300" y="{y_alpha - 4:.1f}" font-size="11" fill="#c0392b">budget {alpha:.1%}</text></svg>')


def render_report(log) -> str:
    sections = []
    for t in log.decision_types():
        pol, rows, pairs = log.get_policy(t), log.rows(t), log.pairs(t)
        served = sum(r["tier"] == "s1" for r in rows)
        agreement = sum(a for _, a, _ in pairs) / len(pairs) if pairs else None
        coverage = (sum(p >= pol.threshold for p, _, _ in pairs) / len(pairs)) if pairs and pol.threshold is not None else None
        scored = [r for r in rows if r["s1_p"] is not None]
        share = sum(r["tier"] == "s1" or r["audited"] for r in scored) / len(scored) if scored else None
        llm_costs = [r["llm_cost"] for r in rows if r["llm_cost"] is not None]
        saved = served * (sum(llm_costs) / len(llm_costs)) if llm_costs else 0.0
        sections.append(
            f"<section><h2>{escape(t)}</h2><table>"
            f"<tr><td>mode</td><td>{escape(pol.mode)}</td></tr>"
            f"<tr><td>threshold</td><td>{'–' if pol.threshold is None else pol.threshold}</td></tr>"
            f"<tr><td>budget</td><td>≤ {pol.alpha:.1%} disagreement at {1 - pol.delta:.0%} confidence</td></tr>"
            f"<tr><td>decisions</td><td>{len(rows)} ({served} served by S1)</td></tr>"
            f"<tr><td>S1 trusted share</td><td>{_pct(share)} of {len(scored)} scored decisions</td></tr>"
            f"<tr><td>S1–LLM agreement (logged pairs)</td><td>{_pct(agreement)} on {len(pairs)} pairs</td></tr>"
            f"<tr><td>coverage at threshold (logged pairs)</td><td>{_pct(coverage)}</td></tr>"
            f"<tr><td>median latency</td><td>S1 {_ms(r['s1_ms'] for r in rows)} · LLM {_ms(r['llm_ms'] for r in rows)}</td></tr>"
            f"<tr><td>est. LLM spend avoided</td><td>${saved:.4f}</td></tr>"
            f"<tr><td>note</td><td>{escape(pol.note) or '–'}</td></tr>"
            f"</table><p class=\"hint\">Pair-based figures use shadow data plus audits; after activation, above-threshold decisions are only audit-sampled.</p>"
            f"{_svg(coverage_curve(pairs, GRID), pol.alpha)}</section>"
        )
    body = "".join(sections) or "<p>No decisions yet.</p>"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Reflex report</title><style>{CSS}</style></head><body><h1>Reflex report</h1>{body}</body></html>")


def write_report(log, path) -> None:
    Path(path).write_text(render_report(log), encoding="utf-8")
