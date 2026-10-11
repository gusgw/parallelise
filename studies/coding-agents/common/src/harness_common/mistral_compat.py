"""Make OpenAI-compatible clients work with a Mistral server.

The Mistral v0.3 chat template requires tool-call ids to be 9-char alphanumeric;
frameworks (LangChain, Pydantic AI) pass on the server's short ids verbatim and get
a 400. This httpx transport rewrites tool-call ids in outgoing /chat/completions
requests, consistently within each request, so any framework works unmodified.
The rewrite is safe for compliant servers too (ids only need to match by equality).
"""
from __future__ import annotations

import json

import httpx

from harness_common.model import new_tool_call_id


def relabel_ids_in_body(body: dict) -> dict:
    """Rewrite tool-call ids in a chat-completions body to 9-char ids (in place)."""
    id_map: dict[str, str] = {}

    def fresh(old: str) -> str:
        if old not in id_map:
            id_map[old] = new_tool_call_id()
        return id_map[old]

    for msg in body.get("messages", []):
        for call in msg.get("tool_calls") or []:
            if "id" in call:
                call["id"] = fresh(call["id"])
        if "tool_call_id" in msg:
            msg["tool_call_id"] = fresh(msg["tool_call_id"])
    return body


def _maybe_rewrite(request: httpx.Request) -> httpx.Request:
    """Return a copy of the request with tool-call ids relabelled, or the original."""
    if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
        return request
    try:
        body = json.loads(request.content)
    except (ValueError, UnicodeDecodeError):
        return request
    if not isinstance(body, dict) or "messages" not in body:
        return request
    content = json.dumps(relabel_ids_in_body(body)).encode()
    headers = [(k, v) for k, v in request.headers.raw if k.lower() != b"content-length"]
    return httpx.Request(request.method, request.url, headers=headers, content=content)


class IdFixTransport(httpx.HTTPTransport):
    """Sync transport that relabels tool-call ids before sending."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return super().handle_request(_maybe_rewrite(request))


class IdFixAsyncTransport(httpx.AsyncHTTPTransport):
    """Async transport that relabels tool-call ids before sending."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await super().handle_async_request(_maybe_rewrite(request))


def make_sync_client(timeout: float = 300.0) -> httpx.Client:
    """httpx.Client that relabels tool-call ids (inject into sync OpenAI clients)."""
    return httpx.Client(transport=IdFixTransport(), timeout=timeout)


def make_async_client(timeout: float = 300.0) -> httpx.AsyncClient:
    """httpx.AsyncClient that relabels tool-call ids (inject into async clients)."""
    return httpx.AsyncClient(transport=IdFixAsyncTransport(), timeout=timeout)
