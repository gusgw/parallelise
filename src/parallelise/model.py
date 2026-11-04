"""Client for an OpenAI-compatible chat endpoint, with measurement built in.

Every call is timed and its token counts are kept, so that a finished run can report how
many requests it made, how many tokens they used and how fast the model generated. This
replaces a separate counting proxy that was used in the first version of this work.

A shared model service often refuses requests for a while when it is busy, answering HTTP 429
(too many requests). Giving up at the first such answer would end a run that only needed to
wait, so a 429 answer is waited out: the wait is the number of seconds in the ``Retry-After``
header when there is one, and otherwise grows with each answer. The waits for one request are
limited, in total, by ``Settings.throttle_wait_s``. ``ModelClient`` does this for the plain
agent, and ``ThrottledClient`` and ``ThrottledAsyncClient`` do the same for the HTTP clients
that the two framework agents hand to their frameworks.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import ConfigError, ModelError

log = logging.getLogger("parallelise.model")

MAX_ATTEMPTS = 3

#: The first wait, in seconds, after an HTTP 429 answer that names no time. Each later answer
#: doubles it, up to ``THROTTLE_LONGEST_STEP_S``.
THROTTLE_FIRST_WAIT_S = 2.0
THROTTLE_LONGEST_STEP_S = 30.0
#: No wait is shorter than this, so that a ``Retry-After`` of zero cannot make a loop that
#: never ends.
MIN_THROTTLE_WAIT_S = 1.0


def _sleep(seconds: float) -> None:
    """Wait; the one place that sleeps, so that tests can replace it."""
    time.sleep(seconds)


async def _async_sleep(seconds: float) -> None:
    """Wait without blocking the event loop; the one place that does so."""
    await asyncio.sleep(seconds)


def parse_retry_after(value: str | None) -> float | None:
    """Read a ``Retry-After`` header as a number of seconds.

    Returns None when the header is absent, is a date instead of a number of seconds, or is
    not a usable number, so that the caller falls back to its own back-off.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class Throttle:
    """Decides how long to wait after each HTTP 429 answer to one request.

    ``allowance_s`` is the most that may be waited in total for this request. When the answer
    names a time with ``Retry-After``, that time is used (never less than
    ``MIN_THROTTLE_WAIT_S``), and if it is more than what is left of the allowance there is
    no point in waiting for less, so the answer is None. When it names none, the wait doubles
    from ``THROTTLE_FIRST_WAIT_S`` with each answer up to ``THROTTLE_LONGEST_STEP_S``, and is
    cut to what is left of the allowance. None means: stop waiting and report the failure.
    """

    def __init__(self, allowance_s: float) -> None:
        self.allowance_s = allowance_s
        self.waited_s = 0.0
        self.answers = 0

    def next_wait(self, retry_after: str | None) -> float | None:
        """Return the seconds to wait before asking again, or None to give up."""
        self.answers += 1
        left = self.allowance_s - self.waited_s
        if left <= 0:
            return None
        told = parse_retry_after(retry_after)
        if told is not None:
            wait = max(told, MIN_THROTTLE_WAIT_S)
            if wait > left:
                return None
        else:
            growing = THROTTLE_FIRST_WAIT_S * 2 ** min(self.answers - 1, 20)
            wait = min(growing, THROTTLE_LONGEST_STEP_S, left)
        self.waited_s += wait
        return wait


def log_wait(wait: float, throttle: Throttle) -> None:
    """Log one wait for an HTTP 429 answer, saying how much of the allowance it uses."""
    log.warning(
        "the endpoint answered HTTP 429 (too many requests); waiting %g seconds before asking "
        "again (%g of %g seconds of waiting used)",
        wait, throttle.waited_s, throttle.allowance_s,
    )


def throttle_failure(throttle: Throttle) -> str:
    """Return the plain sentence for a request that was still refused when waiting stopped."""
    return (
        "The endpoint kept answering HTTP 429 (too many requests), and the "
        f"{throttle.allowance_s:g} seconds of waiting that are allowed are spent "
        f"({throttle.waited_s:g} seconds waited). Try again later, lower the load on the "
        "gateway, or raise PARALLELISE_THROTTLE_WAIT."
    )


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

    def __init__(
        self,
        settings: Settings,
        retry_delay_s: float = 2.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Make a client for ``settings.gateway_url``.

        ``retry_delay_s`` is the wait before the second attempt after a connection error or a
        server error (it doubles for the third). ``sleep`` replaces ``time.sleep`` and exists
        for tests.
        """
        self.settings = settings
        self.retry_delay_s = retry_delay_s
        self._sleep = sleep or _sleep
        self.calls: list[CallRecord] = []
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        try:
            self._http = httpx.Client(
                base_url=settings.gateway_url.rstrip("/") + "/",
                headers=headers,
                timeout=settings.request_timeout_s,
            )
        except httpx.InvalidURL as exc:
            raise ConfigError(
                f"The gateway URL {settings.gateway_url!r} is not a usable web address ({exc}). "
                "Set PARALLELISE_GATEWAY_URL, for example to http://localhost:4000/v1."
            ) from exc

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
        all, waiting longer each time. An HTTP 429 answer (too many requests) is not an
        attempt used up: it is waited out, for the time in the ``Retry-After`` header or for a
        growing time when there is none, until ``Settings.throttle_wait_s`` seconds have been
        waited for this request, and each wait is logged. Other client errors (HTTP 4xx) are
        not retried, because repeating a bad request cannot help.
        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            payload["tools"] = tools
        throttle = Throttle(self.settings.throttle_wait_s)
        last_problem = ""
        failures = 0
        while True:
            started = time.perf_counter()
            try:
                response = self._http.post("chat/completions", json=payload)
            except httpx.TransportError as exc:
                elapsed = time.perf_counter() - started
                self.calls.append(CallRecord(elapsed, 0, 0, "connection-error"))
                last_problem = f"could not reach {self.settings.gateway_url} ({exc})"
            else:
                elapsed = time.perf_counter() - started
                status = response.status_code
                if status == 429:
                    self.calls.append(CallRecord(elapsed, 0, 0, "http-429"))
                    wait = throttle.next_wait(response.headers.get("retry-after"))
                    if wait is None:
                        raise ModelError(throttle_failure(throttle))
                    log_wait(wait, throttle)
                    self._sleep(wait)
                    continue
                if status >= 500:
                    self.calls.append(CallRecord(elapsed, 0, 0, f"http-{status}"))
                    last_problem = f"the endpoint returned HTTP {status}"
                elif status >= 400:
                    self.calls.append(CallRecord(elapsed, 0, 0, f"http-{status}"))
                    raise ModelError(
                        f"The endpoint rejected the request with HTTP {status}: "
                        f"{response.text[:300]!r}. Check the model name "
                        f"({self.settings.model!r}) and the API key."
                    )
                else:
                    return self._parse(response, elapsed)
            failures += 1
            log.warning("attempt %d of %d: %s", failures, MAX_ATTEMPTS, last_problem)
            if failures >= MAX_ATTEMPTS:
                break
            self._sleep(self.retry_delay_s * failures)
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


# --- HTTP clients that wait out throttling, for the two framework agents -----------------------
#
# The framework agents make their own calls, so the waiting has to happen in the HTTP client
# they use. These clients repeat a request that was answered with HTTP 429, with the same
# waits as ``ModelClient``. They repeat it inside ``send``, so the client's request and
# response hooks (which time each request) run once for every attempt.
#
# When the allowance is spent, the last 429 answer is passed on to the framework so that it
# reports the failure in its own way. It carries the header ``x-should-retry: false``, which
# the OpenAI library (used under both frameworks) obeys: otherwise that library would try the
# request again, with another full allowance of waiting each time.


class ThrottledClient(httpx.Client):
    """An ``httpx.Client`` that waits out HTTP 429 answers.

    ``throttle_wait_s`` is the most seconds to wait in total for one request. ``sleep``
    replaces ``time.sleep`` and exists for tests.
    """

    def __init__(
        self,
        *args: Any,
        throttle_wait_s: float,
        sleep: Callable[[float], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._throttle_wait_s = throttle_wait_s
        self._own_sleep = sleep

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        """Send the request; if it is refused with HTTP 429, wait and send it again."""
        throttle = Throttle(self._throttle_wait_s)
        while True:
            response = super().send(request, **kwargs)
            if response.status_code != 429:
                return response
            wait = throttle.next_wait(response.headers.get("retry-after"))
            if wait is None:
                response.headers["x-should-retry"] = "false"
                return response
            response.close()
            log_wait(wait, throttle)
            (self._own_sleep or _sleep)(wait)


def throttled_async_client_class(base: Any) -> Any:
    """Return an asynchronous client class, derived from ``base``, that waits out HTTP 429.

    ``base`` is ``httpx.AsyncClient``, or the same class of the ``httpx2`` library that
    Pydantic AI now uses. The returned class takes ``throttle_wait_s`` and ``sleep`` (an
    asynchronous function that replaces ``asyncio.sleep``, for tests) besides the usual
    arguments.
    """

    class ThrottledAsyncClient(base):  # type: ignore[misc, valid-type]
        """An asynchronous HTTP client that waits out HTTP 429 answers."""

        def __init__(
            self,
            *args: Any,
            throttle_wait_s: float,
            sleep: Callable[[float], Awaitable[None]] | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._throttle_wait_s = throttle_wait_s
            self._own_sleep = sleep

        async def send(self, request: Any, **kwargs: Any) -> Any:
            """Send the request; if it is refused with HTTP 429, wait and send it again."""
            throttle = Throttle(self._throttle_wait_s)
            while True:
                response = await super().send(request, **kwargs)
                if response.status_code != 429:
                    return response
                wait = throttle.next_wait(response.headers.get("retry-after"))
                if wait is None:
                    response.headers["x-should-retry"] = "false"
                    return response
                await response.aclose()
                log_wait(wait, throttle)
                await (self._own_sleep or _async_sleep)(wait)

    return ThrottledAsyncClient


ThrottledAsyncClient = throttled_async_client_class(httpx.AsyncClient)
