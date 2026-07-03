# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Retry-aware HTTP transport for the Chroma 0.6.x FastAPI backend."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from chromadb.api.fastapi import FastAPI
from chromadb.config import System

_LOGGER = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0)


@dataclass(frozen=True)
class _RetryPolicy:
    """Fixed retry policy for the idempotent requests issued by DMF."""

    max_attempts: int = 5
    base_delay: float = 0.2
    max_delay: float = 5.0


class RetryingHTTPClient(httpx.Client):
    """Synchronous HTTP client retrying only transient Chroma failures."""

    def __init__(
        self,
        *,
        policy: _RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        uniform: Callable[[float, float], float] = random.uniform,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("timeout", _HTTP_TIMEOUT)
        super().__init__(**kwargs)
        self._retry_policy = policy or _RetryPolicy()
        self._retry_sleep = sleep
        self._retry_uniform = uniform

    def request(self, method: str, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        """Send a request with bounded full-jitter retries."""
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                response = super().request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self._retry_policy.max_attempts:
                    raise
                self._wait_before_retry(method, attempt, type(exc).__name__)
                continue

            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            if attempt == self._retry_policy.max_attempts:
                return response

            status_code = response.status_code
            response.close()
            self._wait_before_retry(method, attempt, f"HTTP {status_code}")

        raise RuntimeError("unreachable retry state")  # pragma: no cover

    def _wait_before_retry(self, method: str, attempt: int, reason: str) -> None:
        delay_bound = min(
            self._retry_policy.base_delay * (2 ** (attempt - 1)),
            self._retry_policy.max_delay,
        )
        delay = self._retry_uniform(0.0, delay_bound)
        _LOGGER.warning(
            "Retrying Chroma HTTP %s after attempt %d/%d (%s), delay %.3fs",
            method.upper(),
            attempt,
            self._retry_policy.max_attempts,
            reason,
            delay,
        )
        self._retry_sleep(delay)


class RetryingFastAPI(FastAPI):
    """Chroma FastAPI backend with retry installed before client bootstrap."""

    def __init__(self, system: System) -> None:
        super().__init__(system)

        old_session = self._session
        self._session = RetryingHTTPClient(
            headers=old_session.headers,
            verify=(
                self._settings.chroma_server_ssl_verify
                if self._settings.chroma_server_ssl_verify is not None
                else True
            ),
        )
        old_session.close()

