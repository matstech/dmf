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

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from chromadb.config import Settings, System

from dmf.memory.ltm_hooks.chroma_retry import (
    RetryingFastAPI,
    RetryingHTTPClient,
    _RetryPolicy,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleeps: list[float] | None = None,
    uniform: Callable[[float, float], float] | None = None,
    policy: _RetryPolicy | None = None,
    headers: dict[str, str] | None = None,
) -> RetryingHTTPClient:
    kwargs: dict[str, Any] = {
        "transport": httpx.MockTransport(handler),
        "sleep": (sleeps if sleeps is not None else []).append,
        "policy": policy,
        "headers": headers,
    }
    if uniform is not None:
        kwargs["uniform"] = uniform
    return RetryingHTTPClient(**kwargs)


def test_success_on_first_attempt_does_not_sleep() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request)

    with _client(handler, sleeps=sleeps) as client:
        response = client.get("https://chroma.test/heartbeat")

    assert response.status_code == 200
    assert calls == 1
    assert sleeps == []


def test_transport_errors_are_retried_then_succeed() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("temporary", request=request)
        return httpx.Response(200, request=request)

    with _client(handler, sleeps=sleeps, uniform=lambda low, high: high) as client:
        response = client.get("https://chroma.test/heartbeat")

    assert response.status_code == 200
    assert calls == 3
    assert sleeps == [0.2, 0.4]


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_retryable_status_is_retried(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code if calls == 1 else 200, request=request)

    with _client(handler) as client:
        response = client.get("https://chroma.test/query")

    assert response.status_code == 200
    assert calls == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 429, 500])
def test_non_retryable_status_is_returned_immediately(status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, request=request)

    with _client(handler) as client:
        response = client.get("https://chroma.test/query")

    assert response.status_code == status_code
    assert calls == 1


def test_fifth_transport_error_is_reraised_unchanged() -> None:
    calls = 0
    sleeps: list[float] = []
    failure: httpx.ConnectError | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls, failure
        calls += 1
        failure = httpx.ConnectError("still unavailable", request=request)
        raise failure

    with _client(handler, sleeps=sleeps) as client:
        with pytest.raises(httpx.ConnectError) as raised:
            client.get("https://chroma.test/query")

    assert raised.value is failure
    assert calls == 5
    assert len(sleeps) == 4


def test_fifth_retryable_response_reaches_chroma_error_mapping() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable", request=request)

    backend = RetryingFastAPI.__new__(RetryingFastAPI)
    backend._api_url = "https://chroma.test/api/v2"
    backend._session = _client(handler)
    try:
        with pytest.raises(Exception, match="unavailable"):  # noqa: B017
            backend._make_request("get", "/heartbeat")
    finally:
        backend._session.close()

    assert calls == 5


def test_intermediate_retryable_responses_are_closed() -> None:
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if responses:
            assert responses[0].is_closed
        response = httpx.Response(503 if not responses else 200, request=request)
        responses.append(response)
        return response

    with _client(handler) as client:
        client.get("https://chroma.test/query")

    assert responses[0].is_closed


def test_full_jitter_bounds_and_cap_are_applied() -> None:
    uniform_calls: list[tuple[float, float]] = []
    sleeps: list[float] = []

    def uniform(low: float, high: float) -> float:
        uniform_calls.append((low, high))
        return high

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("temporary", request=request)

    policy = _RetryPolicy(max_attempts=5, base_delay=4.0, max_delay=5.0)
    with _client(
        handler,
        sleeps=sleeps,
        uniform=uniform,
        policy=policy,
    ) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get("https://chroma.test/query")

    assert uniform_calls == [(0.0, 4.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]
    assert sleeps == [4.0, 5.0, 5.0, 5.0]


def test_authorization_is_preserved_but_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0
    secret = "not-for-logs"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"] == f"Bearer {secret}"
        return httpx.Response(503 if calls == 1 else 200, request=request)

    caplog.set_level(logging.WARNING)
    with _client(handler, headers={"Authorization": f"Bearer {secret}"}) as client:
        client.request("GET", "https://chroma.test/query", content="private-body")

    assert "not-for-logs" not in caplog.text
    assert "private-body" not in caplog.text
    assert "Authorization" not in caplog.text


def test_retry_client_has_finite_timeouts() -> None:
    with _client(lambda request: httpx.Response(200, request=request)) as client:
        assert client.timeout.connect == 5.0
        assert client.timeout.pool == 5.0
        assert client.timeout.read == 30.0
        assert client.timeout.write == 30.0


def test_chroma_backend_installs_retry_session_with_headers_before_bootstrap() -> None:
    settings = Settings(
        anonymized_telemetry=False,
        chroma_api_impl="dmf.memory.ltm_hooks.chroma_retry.RetryingFastAPI",
        chroma_server_host="localhost",
        chroma_server_http_port=8000,
        chroma_server_headers={"Authorization": "Bearer backend-secret"},
    )
    backend = RetryingFastAPI(System(settings))
    try:
        assert isinstance(backend._session, RetryingHTTPClient)
        assert backend._session.headers["Authorization"] == "Bearer backend-secret"
        assert backend._session.timeout.connect == 5.0
    finally:
        backend._session.close()
