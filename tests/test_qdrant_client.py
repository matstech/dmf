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

import builtins
from typing import Any

import pytest
import qdrant_client

from dmf.memory.ltm_hooks.qdrant_client import (
    QdrantConnectionConfig,
    QdrantConnectionMode,
    build_qdrant_client,
)


def test_memory_factory_builds_in_memory_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    expected_client = object()

    def fake_qdrant_client(*args: Any) -> object:
        calls.append(args)
        return expected_client

    monkeypatch.setattr(qdrant_client, "QdrantClient", fake_qdrant_client)

    result = build_qdrant_client(QdrantConnectionConfig())

    assert result is expected_client
    assert calls == [(":memory:",)]


def test_unknown_mode_fails_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qdrant_client,
        "QdrantClient",
        lambda *args: pytest.fail(f"unexpected QdrantClient: {args}"),
    )
    connection = QdrantConnectionConfig(mode="server")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unsupported Qdrant connection mode"):
        build_qdrant_client(connection)


def test_missing_qdrant_extra_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "qdrant_client":
            raise ModuleNotFoundError(
                "No module named 'qdrant_client'",
                name="qdrant_client",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match=r"dmf-memory\[qdrant\]"):
        build_qdrant_client(QdrantConnectionConfig(mode=QdrantConnectionMode.MEMORY))


def test_unrelated_import_error_is_not_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "qdrant_client":
            raise ModuleNotFoundError(
                "No module named 'transitive_dependency'",
                name="transitive_dependency",
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="transitive_dependency"):
        build_qdrant_client(QdrantConnectionConfig())
