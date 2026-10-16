import json
import logging
import math
import re
import string
import time

import httpx

from .types import S1Prediction


class S1Error(Exception):
    """System 1 could not give a usable answer; callers fall back to the LLM.

    `cost` is what a billed-but-unusable call cost (message stays the bare reason)."""

    def __init__(self, reason="", cost=0.0):
        super().__init__(reason)
        self.cost = cost


def _state(context: dict, max_chars: int) -> dict[str, str]:
    per = max(1, max_chars // max(1, len(context)))
    return {
        k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str))[:per]
        for k, v in context.items()
    }


class SystemOneAdapter:
    """Jev (OpenRouter /api/v1/systemone) and laya-serve (/v1/systemone) share this request schema."""

    def __init__(self, url, model, api_key=None, timeout=2.0, max_state_chars=6000, client=None):
        self.url, self.model, self.api_key = url, model, api_key
        self.timeout, self.max_state_chars = timeout, max_state_chars
        self.client = client or httpx.Client()

    def predict(self, question, options, context) -> S1Prediction:
        criteria = dict(options) if isinstance(options, dict) else {o: o for o in options}
        body = {
            "model": self.model,
            "state": _state(context, self.max_state_chars),
            "questions": {"d": {"type": "choice", "instructions": question, "criteria": criteria}},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        t = time.perf_counter()
        try:
            r = self.client.post(self.url, json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise S1Error(type(e).__name__) from None  # never echo the response body: it may contain the key
        ms = (time.perf_counter() - t) * 1000
        probs = ((data.get("answers") or {}).get("d") or {}).get("probabilities") or {}
        # Jev rounds to 2 decimals, so the sum may be off by up to 0.005 per option
        if set(probs) != set(criteria) or not all(0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > len(criteria) * 0.005:
            raise S1Error("invalid_answer")
        choice = max(probs, key=probs.get)
        cost = float((data.get("usage") or {}).get("cost") or 0.0)
        return S1Prediction(choice, probs, probs[choice], cost, ms)


class StrandsDeciderAdapter(SystemOneAdapter):
    """Strands Decider (`strands-decider serve`): same /v1/systemone schema, local and unauthenticated.

    Its server rounds probabilities to 4 decimals, reports no cost (logged as 0) and ignores `model`.
    """

    DEFAULT_URL = "http://127.0.0.1:8000/v1/systemone"

    def __init__(self, url=DEFAULT_URL, model="strands-decider-latest", **kw):
        super().__init__(url, model, **kw)

    def predict(self, question, options, context) -> S1Prediction:
        if len(options) < 2:  # its choice schema rejects fewer than 2 options with a 422
            raise S1Error("too_few_options")
        return super().predict(question, options, context)


class LogprobLLMAdapter:
    """Small instruct LLM as S1: renormalised first-token logprobs over option letters (not calibrated by themselves)."""

    _warned = False

    def __init__(self, base_url, model, api_key=None, timeout=10.0, max_state_chars=6000, client=None, price_in=None):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.max_state_chars, self.price_in = timeout, max_state_chars, price_in
        self.client = client or httpx.Client()

    def predict(self, question, options, context) -> S1Prediction:
        desc = dict(options) if isinstance(options, dict) else {o: o for o in options}
        if len(desc) < 2:
            raise S1Error("too_few_options")
        if len(desc) > 26:
            raise S1Error("too_many_options")
        labels = {string.ascii_uppercase[i]: name for i, name in enumerate(desc)}
        lines = [f"{lab}. {name}" + (f": {desc[name]}" if desc[name] != name else "") for lab, name in labels.items()]
        ctx = "\n".join(f"{k}: {v}" for k, v in _state(context, self.max_state_chars).items())
        prompt = f"{question}\n\nOptions:\n" + "\n".join(lines) + f"\n\nContext:\n{ctx}\n\nAnswer with the single letter of the best option."
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": "You are a precise classifier."}, {"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "provider": {"require_parameters": True},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        t = time.perf_counter()
        try:
            r = self.client.post(self.base_url + "/chat/completions", json=body, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise S1Error(type(e).__name__) from None
        ms = (time.perf_counter() - t) * 1000
        cost = 0.0
        try:
            usage = data.get("usage") or {}
            try:
                cost = float(usage["cost"])
            except (KeyError, TypeError, ValueError):
                cost = None
            if cost is None:
                cost = (usage.get("prompt_tokens") or 0) * self.price_in / 1e6 if self.price_in else 0.0
                if not self.price_in and not LogprobLLMAdapter._warned:
                    LogprobLLMAdapter._warned = True
                    logging.getLogger(__name__).warning("S1 cost unknown; budget cannot be enforced")
            cost = float(cost)
            choice0 = data["choices"][0]
            content = (choice0.get("logprobs") or {}).get("content")
            if not content:
                raise S1Error("no_logprobs", cost)
            first = content[0]
            msg = choice0.get("message") or {}
            reasoning = bool(msg.get("reasoning") or msg.get("reasoning_content"))
            by_label: dict[str, list[float]] = {}
            top = first["top_logprobs"]
            for e in [*top, *([] if any(t["token"] == first["token"] for t in top) else [first])]:
                m = re.fullmatch(r"[^\w\s]*\s*([A-Z0-9])\s*[^\w\s]*", e["token"].strip())
                lp = float(e["logprob"])
                if m and m.group(1) in labels and math.isfinite(lp) and lp > -1e3:
                    by_label.setdefault(m.group(1), []).append(lp)
        except S1Error:
            raise
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise S1Error("malformed_response", cost) from None
        if not by_label:
            raise S1Error("not_a_label" if reasoning else "no_option_logprobs", cost)
        lse = {lab: max(xs) + math.log(sum(math.exp(x - max(xs)) for x in xs)) for lab, xs in by_label.items()}
        m = max(lse.values())
        z = sum(math.exp(v - m) for v in lse.values())
        probs = {name: math.exp(lse[lab] - m) / z if lab in lse else 0.0 for lab, name in labels.items()}
        choice = max(probs, key=probs.get)
        return S1Prediction(choice, probs, probs[choice], cost, ms)
