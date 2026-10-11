"""Synchronous client for an OpenAI-compatible /chat/completions endpoint."""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://localhost:4000/v1"
DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"
DEFAULT_TEMPERATURE = 0.7


def default_temperature() -> float:
    """Sampling temperature for agent runs (env MODEL_TEMPERATURE, default 0.7).

    Nonzero on purpose: at 0 the model is deterministic and a repeated/looping
    action can never self-correct; a moderate temperature lets it recover.
    """
    return float(os.environ.get("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE))


def model_config() -> dict[str, str]:
    """Read endpoint settings from the environment, with defaults."""
    return {
        "base_url": os.environ.get("MODEL_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("MODEL_NAME", DEFAULT_MODEL),
        "api_key": os.environ.get("MODEL_API_KEY", ""),
    }


def auth_headers(api_key: str) -> dict[str, str]:
    """Build request headers, adding Bearer auth only when a key is set."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def build_payload(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for a chat-completions request."""
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if params:
        payload.update(params)
    return payload


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    config: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 300.0,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Send one request; return the assistant message dict (may hold tool_calls)."""
    cfg = config or model_config()
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    merged = {"temperature": default_temperature()}
    if params:
        merged.update(params)  # explicit params (e.g. a probe forcing 0) win
    payload = build_payload(cfg["model"], messages, tools, merged)
    headers = auth_headers(cfg["api_key"])
    owns = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    finally:
        if owns:
            client.close()
    if usage is not None:  # accumulate per-call usage for telemetry (no behaviour change)
        u = data.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[k] = usage.get(k, 0) + (u.get(k) or 0)
        usage["calls"] = usage.get("calls", 0) + 1
    return data["choices"][0]["message"]


def new_tool_call_id() -> str:
    """Return a 9-char alphanumeric id (the Mistral chat template requires this)."""
    return uuid.uuid4().hex[:9]


def relabel_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy tool_calls with fresh Mistral-valid ids; send each result back under the new id."""
    return [{**c, "id": new_tool_call_id()} for c in tool_calls]


_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Return pong.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def supports_tool_calls(
    *,
    config: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> bool:
    """Preflight: True if the endpoint emits a structured tool_call.

    Detects the 'narration' failure mode (llama-server started without --jinja),
    where the model replies in prose and no tool_calls are produced. See
    common/MODEL_NOTES.md.
    """
    messages = [{"role": "user", "content": "Call the ping tool now."}]
    try:
        msg = chat(messages, tools=[_PING_TOOL], config=config, client=client,
                   params={"temperature": 0}, timeout=timeout)
    except httpx.HTTPError:
        return False
    return bool(msg.get("tool_calls"))
