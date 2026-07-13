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

import pytest

from dmf.memory import temporal_memory
import dmf.memory.ltm_hooks as ltm_hooks
from dmf.memory.ltm_hooks import ChromaLTMHook, FileLTMHook
from dmf.memory.ltm_hooks.chroma_client import ChromaConnectionMode
from dmf.memory.ltm_hooks.qdrant_client import QdrantConnectionMode
from dmf.memory.ltm_hooks.factory import build_ltm_hook
from dmf.models.ltm_hook import NullLTMHook
from dmf.utils.config import VectorConfig
from dmf.utils.config_loader import DMFConfig, LTMSettings


def test_disabled_ltm_builds_null_hook(tmp_path: Path) -> None:
    settings = LTMSettings(
        enabled=False,
        storage_type="chroma",
        chroma_path=str(tmp_path / "must-not-exist"),
    )

    hook = build_ltm_hook(settings, VectorConfig())

    assert isinstance(hook, NullLTMHook)
    assert not (tmp_path / "must-not-exist").exists()


def test_explicit_null_ltm_builds_null_hook() -> None:
    hook = build_ltm_hook(
        LTMSettings(enabled=True, storage_type="null"),
        VectorConfig(),
    )

    assert isinstance(hook, NullLTMHook)


def test_file_ltm_builds_file_hook_with_card_settings(tmp_path: Path) -> None:
    storage_path = tmp_path / "archive" / "ltm.jsonl"
    cards_path = tmp_path / "cards" / "cards.jsonl"
    settings = LTMSettings(
        storage_type="file",
        storage_path=str(storage_path),
        cards_enabled=True,
        cards_path=str(cards_path),
    )

    hook = build_ltm_hook(settings, VectorConfig())

    assert isinstance(hook, FileLTMHook)
    assert hook.path == storage_path
    assert hook.card_store is not None
    assert hook.card_store.path == cards_path


def test_chroma_embedded_ltm_builds_hook_with_connection_and_vector_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_init(self: ChromaLTMHook, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(ChromaLTMHook, "__init__", fake_init)
    vector_config = VectorConfig(model_name="local-test-model", vector_dim=17, window_size=3)
    settings = LTMSettings(
        storage_type="chroma",
        chroma_path=str(tmp_path / "chroma"),
        collection_name="raw_collection",
        distance_threshold=0.33,
        cards_enabled=True,
        cards_path=str(tmp_path / "cards.jsonl"),
        cards_collection_name="card_collection",
    )

    hook = build_ltm_hook(settings, vector_config)

    assert isinstance(hook, ChromaLTMHook)
    assert captured["collection_name"] == "raw_collection"
    assert captured["persist_directory"] == str(tmp_path / "chroma")
    assert captured["distance_threshold"] == 0.33
    assert captured["vector_config"] is vector_config
    assert captured["cards_enabled"] is True
    assert captured["cards_path"] == str(tmp_path / "cards.jsonl")
    assert captured["cards_collection_name"] == "card_collection"
    connection = captured["connection"]
    assert connection.mode is ChromaConnectionMode.EMBEDDED
    assert connection.persist_directory == str(tmp_path / "chroma")
    assert connection.auth_token is None


def test_chroma_server_ltm_builds_connection_with_auth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    env_reads: list[str] = []

    monkeypatch.setattr(
        ChromaLTMHook,
        "__init__",
        lambda self, **kwargs: captured.update(kwargs),  # noqa: ARG005
    )
    monkeypatch.setattr(
        temporal_memory.os,
        "getenv",
        lambda name: env_reads.append(name) or "top-secret-token",
    )
    settings = LTMSettings(
        storage_type="chroma",
        chroma_mode="server",
        chroma_host="chroma.internal",
        chroma_port=8443,
        chroma_ssl=True,
        chroma_tenant="tenant-a",
        chroma_database="database-a",
        chroma_auth_token_env="DMF_CHROMA_TOKEN",
    )

    build_ltm_hook(settings, VectorConfig())

    connection = captured["connection"]
    assert connection.mode is ChromaConnectionMode.SERVER
    assert connection.host == "chroma.internal"
    assert connection.port == 8443
    assert connection.ssl is True
    assert connection.tenant == "tenant-a"
    assert connection.database == "database-a"
    assert connection.auth_token == "top-secret-token"
    assert "top-secret-token" not in repr(connection)
    assert env_reads == ["DMF_CHROMA_TOKEN"]


def test_chroma_server_missing_auth_token_raises_without_secret_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(temporal_memory.os, "getenv", lambda name: "   ")  # noqa: ARG005
    settings = LTMSettings(
        storage_type="chroma",
        chroma_mode="server",
        chroma_auth_token_env="DMF_CHROMA_TOKEN",
    )

    with pytest.raises(ValueError, match="DMF_CHROMA_TOKEN") as exc_info:
        build_ltm_hook(settings, VectorConfig())

    assert "Authorization" not in str(exc_info.value)


def test_qdrant_ltm_builds_hook_with_connection_and_vector_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeQdrantLTMHook:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(ltm_hooks, "QdrantLTMHook", FakeQdrantLTMHook)
    vector_config = VectorConfig(model_name="local-test-model", vector_dim=13)
    settings = LTMSettings(
        storage_type="qdrant",
        qdrant_mode="memory",
        collection_name="qdrant_raw",
        distance_threshold=0.27,
        cards_enabled=True,
        cards_path="cards.jsonl",
        cards_collection_name="qdrant_cards",
    )

    hook = build_ltm_hook(settings, vector_config)

    assert isinstance(hook, FakeQdrantLTMHook)
    assert captured["collection_name"] == "qdrant_raw"
    assert captured["distance_threshold"] == 0.27
    assert captured["vector_config"] is vector_config
    assert captured["cards_enabled"] is True
    assert captured["cards_path"] == "cards.jsonl"
    assert captured["cards_collection_name"] == "qdrant_cards"
    assert captured["connection"].mode is QdrantConnectionMode.MEMORY


@pytest.mark.parametrize(
    ("settings", "expected_type"),
    [
        (LTMSettings(enabled=False, storage_type="chroma"), NullLTMHook),
        (LTMSettings(storage_type="file"), FileLTMHook),
        (LTMSettings(storage_type="null"), NullLTMHook),
    ],
)
def test_non_active_chroma_server_config_does_not_read_auth_environment(
    monkeypatch: pytest.MonkeyPatch,
    settings: LTMSettings,
    expected_type: type,
) -> None:
    monkeypatch.setattr(
        temporal_memory.os,
        "getenv",
        lambda name: pytest.fail(f"unexpected environment read: {name}"),
    )
    settings = LTMSettings(
        **{
            **settings.__dict__,
            "chroma_mode": "server",
            "chroma_auth_token_env": "DMF_CHROMA_TOKEN",
        }
    )

    hook = build_ltm_hook(settings, VectorConfig())

    assert isinstance(hook, expected_type)


def test_unknown_storage_type_raises_value_error() -> None:
    settings = LTMSettings(storage_type="unsupported")

    with pytest.raises(ValueError, match="Unsupported ltm.storage_type at runtime"):
        build_ltm_hook(settings, VectorConfig())


def test_temporal_memory_explicit_hook_bypasses_factory_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        temporal_memory,
        "build_ltm_hook",
        lambda settings, vector_config: pytest.fail("unexpected factory call"),
    )
    monkeypatch.setattr(
        temporal_memory.os,
        "getenv",
        lambda name: pytest.fail(f"unexpected environment read: {name}"),
    )
    explicit = NullLTMHook()
    config = DMFConfig(
        ltm=LTMSettings(
            storage_type="chroma",
            chroma_mode="server",
            chroma_auth_token_env="DMF_CHROMA_TOKEN",
        )
    )

    tm = temporal_memory.TemporalMemory.from_dmf_config(config, ltm_hook=explicit)

    assert tm.ltm_hook is explicit
