"""Tests for harness_common.model — no network (httpx MockTransport)."""
import httpx

from harness_common import model


def test_build_payload_without_tools():
    p = model.build_payload("m", [{"role": "user", "content": "hi"}])
    assert p == {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert "tools" not in p


def test_build_payload_with_tools_adds_tool_choice():
    tools = [{"type": "function", "function": {"name": "f"}}]
    p = model.build_payload("m", [], tools=tools)
    assert p["tools"] == tools
    assert p["tool_choice"] == "auto"


def test_build_payload_merges_params():
    p = model.build_payload("m", [], params={"temperature": 0})
    assert p["temperature"] == 0


def test_auth_headers_without_key():
    h = model.auth_headers("")
    assert "Authorization" not in h
    assert h["Content-Type"] == "application/json"


def test_auth_headers_with_key():
    h = model.auth_headers("secret")
    assert h["Authorization"] == "Bearer secret"


def test_model_config_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://x/v1")
    monkeypatch.setenv("MODEL_NAME", "foo")
    monkeypatch.setenv("MODEL_API_KEY", "k")
    assert model.model_config() == {
        "base_url": "http://x/v1",
        "model": "foo",
        "api_key": "k",
    }


def test_chat_returns_assistant_message():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = {"base_url": "http://x/v1", "model": "m", "api_key": ""}
    msg = model.chat([{"role": "user", "content": "hi"}], config=cfg, client=client)
    assert msg["content"] == "hello"
    assert captured["url"] == "http://x/v1/chat/completions"


def test_chat_passes_through_tool_calls():
    tool_calls = [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": None,
                                           "tool_calls": tool_calls}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = {"base_url": "http://x/v1", "model": "m", "api_key": ""}
    msg = model.chat([], config=cfg, client=client)
    assert msg["tool_calls"] == tool_calls


def test_supports_tool_calls_true_when_tool_call_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "1", "type": "function",
                            "function": {"name": "ping", "arguments": "{}"}}]}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = {"base_url": "http://x/v1", "model": "m", "api_key": ""}
    assert model.supports_tool_calls(config=cfg, client=client) is True


def test_supports_tool_calls_false_on_narration():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": "I will call the ping tool."}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = {"base_url": "http://x/v1", "model": "m", "api_key": ""}
    assert model.supports_tool_calls(config=cfg, client=client) is False


def test_default_temperature_env(monkeypatch):
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    assert model.default_temperature() == 0.7
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.3")
    assert model.default_temperature() == 0.3


def test_chat_applies_default_temperature(monkeypatch):
    monkeypatch.delenv("MODEL_TEMPERATURE", raising=False)
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = {"base_url": "http://x/v1", "model": "m", "api_key": ""}
    model.chat([], config=cfg, client=client)
    assert captured["body"]["temperature"] == 0.7              # default knob
    model.chat([], config=cfg, client=client, params={"temperature": 0})
    assert captured["body"]["temperature"] == 0                # explicit override wins


def test_new_tool_call_id_is_9_char_alnum():
    tid = model.new_tool_call_id()
    assert len(tid) == 9
    assert tid.isalnum()


def test_relabel_tool_calls_replaces_ids_keeps_fields():
    calls = [
        {"id": "1", "type": "function", "function": {"name": "add", "arguments": "{}"}}
    ]
    out = model.relabel_tool_calls(calls)
    assert out[0]["function"] == calls[0]["function"]
    assert out[0]["id"] != "1"
    assert len(out[0]["id"]) == 9 and out[0]["id"].isalnum()
