"""
Shared HTTP client wrappers for external APIs (Mistral, Tavily).

Centralizes retry-with-backoff behavior so every caller (claim extraction
in Phase 2, claim verification in Phase 3) gets the same resilience
guarantees without re-implementing them. This module owns *transport*
concerns only -- prompt content and response parsing belong to the
modules that call it.

Why plain `requests` instead of the official SDKs: the whole point of
this module is custom retry/backoff behavior tuned to the free-tier rate
limits this project runs on (see architecture notes). Wrapping raw HTTP
calls keeps that logic in one place and avoids fighting each SDK's own
differing retry semantics.
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

from config import Settings

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
TAVILY_API_URL = "https://api.tavily.com/search"

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.5  # seconds; exponential backoff base


class ApiClientError(RuntimeError):
    """Raised when an external API call fails after all retries are exhausted,
    or fails in a way that's not worth retrying (e.g. bad auth)."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = _DEFAULT_MAX_RETRIES
    base_delay_seconds: float = _DEFAULT_BASE_DELAY
    retry_on_status: tuple = (429, 500, 502, 503, 504)


DEFAULT_RETRY_POLICY = RetryPolicy()


def _sleep_for_attempt(
    attempt: int, base_delay: float, retry_after: Optional[float] = None
) -> None:
    """Sleep before a retry. Honors a server-supplied Retry-After header
    when present, otherwise uses exponential backoff: base_delay * 2^attempt."""
    delay = retry_after if retry_after is not None else base_delay * (2**attempt)
    logger.warning("Retrying after %.1fs (attempt %d)", delay, attempt + 1)
    time.sleep(delay)


def _call_with_retry(
    request_fn: Callable[[], "requests.Response"],
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    op_name: str = "api_call",
) -> "requests.Response":
    """Run an HTTP request with retry + exponential backoff on transient
    failures (429 rate limit, 5xx server errors, network errors).

    Non-retryable client errors (4xx other than 429) fail immediately --
    retrying a bad request just wastes the rate-limit budget.
    """
    last_error: Optional[Exception] = None

    for attempt in range(policy.max_retries + 1):
        try:
            response = request_fn()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < policy.max_retries:
                _sleep_for_attempt(attempt, policy.base_delay_seconds)
                continue
            raise ApiClientError(
                f"{op_name} failed after {policy.max_retries} retries: {exc}"
            ) from exc
        else:
            if response.status_code in policy.retry_on_status:
                if attempt < policy.max_retries:
                    retry_after_header = response.headers.get("Retry-After")
                    retry_after = float(retry_after_header) if retry_after_header else None
                    _sleep_for_attempt(attempt, policy.base_delay_seconds, retry_after)
                    continue
                raise ApiClientError(
                    f"{op_name} failed after {policy.max_retries} retries: "
                    f"HTTP {response.status_code} -- {response.text[:300]}"
                )

            if not response.ok:
                raise ApiClientError(
                    f"{op_name} returned HTTP {response.status_code}: {response.text[:300]}"
                )

            return response

    raise ApiClientError(f"{op_name} failed: {last_error}")


def call_mistral_chat(
    settings: Settings,
    messages: list,
    model: str = "mistral-small-latest",
    json_mode: bool = True,
    temperature: float = 0.0,
    timeout: float = 60.0,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> dict:
    """Call the Mistral chat completions endpoint with retry/backoff.

    Returns the parsed JSON response body. Raises ApiClientError on
    unrecoverable failure (missing key, auth error, exhausted retries).
    """
    if not settings.mistral_api_key:
        raise ApiClientError("MISTRAL_API_KEY is not configured.")

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }

    def _do_request() -> "requests.Response":
        return requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=timeout)

    response = _call_with_retry(_do_request, policy=policy, op_name=f"mistral:{model}")
    return response.json()


def call_tavily_search(
    settings: Settings,
    query: str,
    max_results: int = 3,
    timeout: float = 30.0,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> dict:
    """Call the Tavily search endpoint with retry/backoff.

    Returns the parsed JSON response body (contains a "results" list of
    {title, url, content, score} entries).
    """
    if not settings.tavily_api_key:
        raise ApiClientError("TAVILY_API_KEY is not configured.")

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
    }

    def _do_request() -> "requests.Response":
        return requests.post(TAVILY_API_URL, json=payload, timeout=timeout)

    response = _call_with_retry(_do_request, policy=policy, op_name="tavily:search")
    return response.json()
