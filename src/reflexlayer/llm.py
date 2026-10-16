import json
import time

import httpx

from .types import LLMResult


class LLMError(Exception):
    """The LLM answered, but not with one of the options."""


class OpenAICompatLLM:
    def __init__(self, base_url, model, api_key, price_in=0.0, price_out=0.0, timeout=60.0, client=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key = model, api_key
        self.price_in, self.price_out, self.timeout = price_in, price_out, timeout
        self.client = client or httpx.Client()

    def decide(self, question, options, context) -> LLMResult:
        names = list(options)
        lines = "\n".join(f"- {k}: {v}" for k, v in options.items()) if isinstance(options, dict) else "\n".join(f"- {k}" for k in names)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You make exactly one decision. Reply only with the JSON object."},
                {"role": "user", "content": f"{question}\n\nOptions:\n{lines}\n\nContext:\n{json.dumps(context, ensure_ascii=False, default=str)}"},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": names}},
                "required": ["choice"],
                "additionalProperties": False,
            }}},
        }
        t = time.perf_counter()
        r = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        ms = (time.perf_counter() - t) * 1000
        try:
            choice = json.loads(data["choices"][0]["message"]["content"])["choice"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("unparseable_answer") from None
        if choice not in names:
            raise LLMError("choice_not_in_options")
        usage = data.get("usage") or {}
        cost = usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * self.price_out
        return LLMResult(choice, cost, ms)
