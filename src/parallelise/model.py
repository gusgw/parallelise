"""Client for an OpenAI-compatible chat endpoint, with measurement built in.

Every call is timed and its token counts are kept, so that a finished run can report how
many requests it made, how many tokens they used and how fast the model generated. This
replaces a separate counting proxy that was used in the first version of this work.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import ModelError

log = logging.getLogger("parallelise.model")

MAX_ATTEMPTS = 3


@dataclass
class ChatResult:
    """One answer from the model."""

    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class CallRecord:
    """One attempt to call the model. Status is "ok", "http-<code>" or "connection-error"."""

    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    status: str


@dataclass
class HealthReport:
    """What a health check found out about the endpoint."""

    reachable: bool
    model_listed: bool
    models: list[str] = field(default_factory=list)
    chat_ok: bool = False
    round_trip_s: float = 0.0
    detail: str = ""

    @property
    def healthy(self) -> bool:
        """True when the endpoint answered, lists the model and completed a chat call."""
        return self.reachable and self.model_listed and self.chat_ok


class ModelClient:
    """Calls a chat-completions endpoint and keeps a record of every call."""

    def __init__(self, settings: Settings, retry_delay_s: float = 2.0) -> None:
        self.settings = settings
        self.retry_delay_s = retry_delay_s
        self.calls: list[CallRecord] = []
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._http = httpx.Client(
            base_url=settings.gateway_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.request_timeout_s,
        )

    def close(self) -> None:
        """Release the network connection."""
        self._http.close()

    def __enter__(self) -> ModelClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatResult:
        """Send one chat request and return the assistant's message with its usage.

        Connection errors and server errors (HTTP 5xx) are retried up to three attempts in
        all, waiting longer each time. Client errors (HTTP 4xx) are not retried, because
        repeating a bad request cannot help.
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            payload["tools"] = tools
        last_problem = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.perf_counter()
            try:
                response = self._http.post("chat/completions", json=payload)
            except httpx.TransportError as exc:
                elapsed = time.perf_counter() - started
                self.calls.append(CallRecord(elapsed, 0, 0, "connection-error"))
                last_problem = f"could not reach {self.settings.gateway_url} ({exc})"
                log.warning("attempt %d of %d: %s", attempt, MAX_ATTEMPTS, last_problem)
            else:
                elapsed = time.perf_counter() - started
                if response.status_code >= 500:
                    self.calls.append(CallRecord(elapsed, 0, 0, f"http-{response.status_code}"))
                    last_problem = f"the endpoint returned HTTP {response.status_code}"
                    log.warning("attempt %d of %d: %s", attempt, MAX_ATTEMPTS, last_problem)
                elif response.status_code >= 400:
                    self.calls.append(CallRecord(elapsed, 0, 0, f"http-{response.status_code}"))
                    raise ModelError(
                        f"The endpoint rejected the request with HTTP {response.status_code}: "
                        f"{response.text[:300]!r}. Check the model name "
                        f"({self.settings.model!r}) and the API key."
                    )
                else:
                    return self._parse(response, elapsed)
            if attempt < MAX_ATTEMPTS:
                time.sleep(self.retry_delay_s * attempt)
        raise ModelError(
            f"The model call failed after {MAX_ATTEMPTS} attempts: {last_problem}. "
            "Check that the gateway is running (parallelise health)."
        )

    def _parse(self, response: httpx.Response, elapsed: float) -> ChatResult:
        try:
            body = response.json()
            message = dict(body["choices"][0]["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.calls.append(CallRecord(elapsed, 0, 0, "bad-response"))
            raise ModelError(
                "The endpoint answered, but the reply is not a chat completion "
                f"({type(exc).__name__}). Check that the URL ends in /v1 and names an "
                "OpenAI-compatible server."
            ) from exc
        if message.get("content") is None:
            message["content"] = ""
        usage = body.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        self.calls.append(CallRecord(elapsed, prompt, completion, "ok"))
        return ChatResult(message, prompt, completion, elapsed)

    def summary(self) -> dict[str, Any]:
        """Totals over the successful calls so far.

        Completion tokens per second divides the tokens generated by the time spent waiting
        for answers, so it measures the endpoint rather than the agent's own thinking time.
        """
        good = [c for c in self.calls if c.status == "ok"]
        latencies = [c.latency_s for c in good]
        prompt = sum(c.prompt_tokens for c in good)
        completion = sum(c.completion_tokens for c in good)
        generation_time = sum(latencies)
        return {
            "requests": len(good),
            "failed_requests": len(self.calls) - len(good),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "mean_latency_s": statistics.fmean(latencies) if latencies else 0.0,
            "median_latency_s": statistics.median(latencies) if latencies else 0.0,
            "completion_tokens_per_second": (
                completion / generation_time if generation_time > 0 else 0.0
            ),
        }

    def health_check(self) -> HealthReport:
        """Check the endpoint without raising: list the models, then make one tiny chat call.

        The tiny call is not counted in the client's call records.
        """
        report = HealthReport(reachable=False, model_listed=False)
        started = time.perf_counter()
        try:
            response = self._http.get("models", timeout=min(30.0, self.settings.request_timeout_s))
        except httpx.TransportError as exc:
            report.detail = (
                f"Could not reach {self.settings.gateway_url} ({exc}). "
                "Start the gateway or set PARALLELISE_GATEWAY_URL."
            )
            return report
        report.reachable = True
        if response.status_code != 200:
            report.detail = (
                f"GET /models returned HTTP {response.status_code}. "
                "Check the API key and that the URL ends in /v1."
            )
            report.round_trip_s = time.perf_counter() - started
            return report
        try:
            report.models = [str(m["id"]) for m in response.json().get("data", [])]
        except (ValueError, KeyError, TypeError, AttributeError):
            report.detail = "GET /models did not return a model list in the expected form."
            report.round_trip_s = time.perf_counter() - started
            return report
        report.model_listed = self.settings.model in report.models
        if not report.model_listed:
            report.detail = (
                f"The model {self.settings.model!r} is not listed; the endpoint offers "
                f"{', '.join(report.models) or 'no models'}. Set PARALLELISE_MODEL."
            )
        mark = len(self.calls)
        try:
            self.chat([{"role": "user", "content": "Reply with the single word: ready."}])
            report.chat_ok = True
        except ModelError as exc:
            report.detail = (report.detail + " " + str(exc)).strip()
        finally:
            del self.calls[mark:]
        report.round_trip_s = time.perf_counter() - started
        return report
