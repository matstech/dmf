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
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from dmf.memory.ltm_hooks.qdrant_client import (
    QdrantConnectionConfig,
    build_qdrant_client,
)


def test_imports_remain_safe_without_qdrant_extra() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(repo_root)
        if not env.get("PYTHONPATH")
        else f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockQdrant(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "qdrant_client" or fullname.startswith("qdrant_client."):
                    raise ModuleNotFoundError(
                        "No module named 'qdrant_client'",
                        name="qdrant_client",
                    )
                return None

        sys.meta_path.insert(0, BlockQdrant())

        import dmf
        import dmf.memory
        from dmf.memory import ChromaLTMHook, FileLTMHook, QdrantLTMHook
        from dmf.memory.ltm_hooks import QdrantLTMHook as HookExport
        from dmf.memory.ltm_hooks.qdrant_client import (
            QdrantConnectionConfig,
            build_qdrant_client,
        )

        assert dmf.memory.ChromaLTMHook is ChromaLTMHook
        assert dmf.memory.FileLTMHook is FileLTMHook
        assert dmf.QdrantLTMHook is QdrantLTMHook
        assert QdrantLTMHook is HookExport

        try:
            build_qdrant_client(QdrantConnectionConfig())
        except ModuleNotFoundError as exc:
            assert "Install the Qdrant backend" in str(exc)
            assert "dmf-memory[qdrant]" in str(exc)
        else:
            raise AssertionError("expected missing qdrant extra")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_qdrant_client_factory_does_not_mask_internal_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "qdrant_client":
            raise ModuleNotFoundError(
                "No module named 'qdrant_client_internal'",
                name="qdrant_client_internal",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        build_qdrant_client(QdrantConnectionConfig())

    assert exc_info.value.name == "qdrant_client_internal"
    assert "Install the Qdrant backend" not in str(exc_info.value)
