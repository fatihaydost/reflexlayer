import json
import time

import httpx
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .detect import build_response, detect, parse_choice
from .router import Router
from .types import LLMResult

HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}


class Passthrough(Exception):
    def __init__(self, response):
        self.response = response


class ForwardLLM:
    """The gateway's LLM tier is the customer's original request, forwarded unchanged."""

    def __init__(self, client, url, raw, headers, det):
        self.client, self.url, self.raw, self.headers, self.det = client, url, raw, headers, det
        self.response = None

    def decide(self, question, options, context):
        t = time.perf_counter()
        r = self.client.post(self.url, content=self.raw, headers=self.headers)
        if r.status_code != 200:
            raise Passthrough(r)
        try:
            choice = parse_choice(self.det, r.json())
        except ValueError:
            choice = None
        if choice is None:
            raise Passthrough(r)
        self.response = r
        # shortcut: gateway does not price tokens, cost 0; add a per-model price table when the report needs $ savings for gateway traffic
        return LLMResult(choice, 0.0, (time.perf_counter() - t) * 1000)


def _forward_headers(request):
    return {k: v for k, v in request.headers.items()
            if k.lower() not in HOP_HEADERS and not k.lower().startswith("x-reflex-")}


RESPONSE_DROP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def _as_response(r):
    # shortcut: upstream response is buffered; streaming passthrough is out of MVP scope
    # (httpx already decoded the body, so encoding/framing headers no longer apply)
    headers = {k: v for k, v in r.headers.items() if k.lower() not in RESPONSE_DROP}
    return Response(r.content, status_code=r.status_code, headers=headers)


def create_app(s1, log, upstream, client=None, rules=None):
    client = client or httpx.Client(timeout=120.0)
    upstream = upstream.rstrip("/")

    async def chat(request):
        raw = await request.body()
        headers = _forward_headers(request)
        url = upstream + "/chat/completions"
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        det = detect(body, request.headers.get("x-reflex-decision-type")) if isinstance(body, dict) else None
        if det is None:
            return _as_response(await run_in_threadpool(client.post, url, content=raw, headers=headers))
        llm = ForwardLLM(client, url, raw, headers, det)
        router = Router(s1, llm, log, rules=rules)
        try:
            d = await run_in_threadpool(router.decide, det.question, det.options, det.context, det.decision_type,
                                        request.headers.get("x-reflex-session-id"))
        except Passthrough as p:
            return _as_response(p.response)
        if d.tier == "llm":
            return _as_response(llm.response)
        return JSONResponse(build_response(det, d.choice, body.get("model", "reflex")), headers={"x-reflex-tier": d.tier})

    async def proxy(request):
        r = await run_in_threadpool(
            client.request, request.method, upstream + "/" + request.path_params["path"],
            content=await request.body(), headers=_forward_headers(request), params=str(request.query_params),
        )
        return _as_response(r)

    return Starlette(routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
    ])
