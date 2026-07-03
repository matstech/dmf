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

from pathlib import Path
from typing import Any

import chromadb.api.fastapi as chroma_fastapi
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.auth import UserIdentity

from dmf.memory.ltm_hooks import chroma_client, chroma_hook
from dmf.memory.ltm_hooks.chroma_client import (
    ChromaConnectionConfig,
    ChromaConnectionMode,
    build_chroma_client,
)
from dmf.memory.ltm_hooks.chroma_hook import ChromaLTMHook
from dmf.memory.ltm_hooks.chroma_retry import RetryingFastAPI

_ORIGINAL_CHROMA_FASTAPI = chroma_fastapi.FastAPI


class _FakeClient:
    def __init__(self) -> None:
        self.collection_names: list[str] = []

    def get_or_create_collection(self, *, name: str, metadata: dict[str, str]) -> object:
        self.collection_names.append(name)
        return object()


def test_embedded_factory_creates_directory_and_persistent_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    expected_client = object()

    def fake_persistent_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return expected_client

    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", fake_persistent_client)
    persist_directory = tmp_path / "nested" / "chroma"

    result = build_chroma_client(
        ChromaConnectionConfig(
            persist_directory=persist_directory,
            tenant="tenant-a",
            database="database-a",
        )
    )

    assert result is expected_client
    assert persist_directory.is_dir()
    assert captured["path"] == str(persist_directory)
    assert captured["tenant"] == "tenant-a"
    assert captured["database"] == "database-a"
    assert captured["settings"].anonymized_telemetry is False
    assert captured["settings"].chroma_api_impl == "chromadb.api.segment.SegmentAPI"


@pytest.mark.parametrize("auth_token", [None, "top-secret-token"])
def test_server_factory_builds_retry_aware_client_without_creating_local_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auth_token: str | None,
) -> None:
    captured: dict[str, Any] = {}
    expected_client = object()

    def fake_client(**kwargs: Any) -> object:
        assert chroma_fastapi.FastAPI is _ORIGINAL_CHROMA_FASTAPI
        captured.update(kwargs)
        return expected_client

    monkeypatch.setattr(chroma_client.chromadb, "Client", fake_client)
    persist_directory = tmp_path / "must-not-exist"
    connection = ChromaConnectionConfig(
        mode=ChromaConnectionMode.SERVER,
        persist_directory=persist_directory,
        host="chroma.internal",
        port=8443,
        ssl=True,
        tenant="tenant-b",
        database="database-b",
        auth_token=auth_token,
    )

    result = build_chroma_client(connection)

    assert result is expected_client
    assert not persist_directory.exists()
    assert captured["tenant"] == "tenant-b"
    assert captured["database"] == "database-b"
    settings = captured["settings"]
    assert settings.chroma_server_host == "chroma.internal"
    assert settings.chroma_server_http_port == 8443
    assert settings.chroma_server_ssl_enabled is True
    assert settings.chroma_api_impl == (
        "dmf.memory.ltm_hooks.chroma_retry.RetryingFastAPI"
    )
    expected_headers = (
        {"Authorization": "Bearer top-secret-token"} if auth_token else None
    )
    assert settings.chroma_server_headers == expected_headers
    assert "top-secret-token" not in repr(connection)
    assert chroma_fastapi.FastAPI is _ORIGINAL_CHROMA_FASTAPI


def test_unknown_mode_fails_without_client_or_directory_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        chroma_client.chromadb,
        "PersistentClient",
        lambda **kwargs: pytest.fail(f"unexpected PersistentClient: {kwargs}"),
    )
    monkeypatch.setattr(
        chroma_client.chromadb,
        "Client",
        lambda **kwargs: pytest.fail(f"unexpected Client: {kwargs}"),
    )
    persist_directory = tmp_path / "must-not-exist"
    connection = ChromaConnectionConfig(
        mode="unsupported",  # type: ignore[arg-type]
        persist_directory=persist_directory,
    )

    with pytest.raises(ValueError, match="Unsupported Chroma connection mode"):
        build_chroma_client(connection)

    assert not persist_directory.exists()


def test_hook_uses_injected_client_without_calling_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    monkeypatch.setattr(
        chroma_hook,
        "build_chroma_client",
        lambda connection: pytest.fail(f"unexpected factory call: {connection}"),
    )

    hook = ChromaLTMHook(client=client, persist_directory=tmp_path / "not-created")  # type: ignore[arg-type]

    assert hook._client is client
    assert client.collection_names == ["dmf_memory"]
    assert not (tmp_path / "not-created").exists()


def test_hook_legacy_persist_directory_builds_embedded_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    captured: list[ChromaConnectionConfig] = []

    def fake_factory(connection: ChromaConnectionConfig) -> _FakeClient:
        captured.append(connection)
        return client

    monkeypatch.setattr(chroma_hook, "build_chroma_client", fake_factory)
    persist_directory = tmp_path / "legacy"

    hook = ChromaLTMHook(persist_directory=persist_directory)

    assert hook._client is client
    assert captured == [ChromaConnectionConfig(persist_directory=persist_directory)]
    assert captured[0].mode is ChromaConnectionMode.EMBEDDED


def test_hook_passes_explicit_connection_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient()
    connection = ChromaConnectionConfig(mode=ChromaConnectionMode.SERVER)
    captured: list[ChromaConnectionConfig] = []
    monkeypatch.setattr(
        chroma_hook,
        "build_chroma_client",
        lambda value: captured.append(value) or client,
    )

    ChromaLTMHook(connection=connection)

    assert captured == [connection]


def test_server_factory_installs_retry_backend_before_real_client_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RetryingFastAPI,
        "get_user_identity",
        lambda self: UserIdentity(user_id="test-user"),  # noqa: ARG005
    )
    monkeypatch.setattr(
        RetryingFastAPI,
        "get_tenant",
        lambda self, name: {"name": name},  # noqa: ARG005
    )
    monkeypatch.setattr(
        RetryingFastAPI,
        "get_database",
        lambda self, name, tenant: {"name": name, "tenant": tenant},  # noqa: ARG005
    )
    SharedSystemClient.clear_system_cache()

    client = build_chroma_client(
        ChromaConnectionConfig(mode=ChromaConnectionMode.SERVER)
    )
    try:
        assert isinstance(client._server, RetryingFastAPI)
        assert client._server._session.timeout.connect == 5.0
    finally:
        client._server._session.close()
        client._system.stop()
        SharedSystemClient.clear_system_cache()


def test_server_factory_restores_identifier_resolver_when_client_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resolver = SharedSystemClient._get_identifier_from_settings

    def fail_client(**kwargs: Any) -> object:
        settings = kwargs["settings"]
        identifier = SharedSystemClient._get_identifier_from_settings(settings)
        assert identifier
        assert chroma_fastapi.FastAPI is _ORIGINAL_CHROMA_FASTAPI
        raise RuntimeError("client bootstrap failed")

    monkeypatch.setattr(chroma_client.chromadb, "Client", fail_client)

    with pytest.raises(RuntimeError, match="client bootstrap failed"):
        build_chroma_client(
            ChromaConnectionConfig(mode=ChromaConnectionMode.SERVER)
        )

    assert SharedSystemClient._get_identifier_from_settings is original_resolver
    assert chroma_fastapi.FastAPI is _ORIGINAL_CHROMA_FASTAPI


def test_retry_backend_allowlist_delegates_standard_chroma_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_resolver = SharedSystemClient._get_identifier_from_settings
    standard_settings = chroma_client.Settings(
        is_persistent=True,
        persist_directory=str(tmp_path / "standard"),
    )
    expected_identifier = original_resolver(standard_settings)

    def fake_client(**kwargs: Any) -> object:
        assert SharedSystemClient._get_identifier_from_settings(
            standard_settings
        ) == expected_identifier
        assert chroma_fastapi.FastAPI is _ORIGINAL_CHROMA_FASTAPI
        return object()

    monkeypatch.setattr(chroma_client.chromadb, "Client", fake_client)

    build_chroma_client(ChromaConnectionConfig(mode=ChromaConnectionMode.SERVER))

    assert SharedSystemClient._get_identifier_from_settings is original_resolver
