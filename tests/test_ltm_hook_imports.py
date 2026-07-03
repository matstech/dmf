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

"""Compatibility tests for canonical and legacy LTM hook imports."""

from __future__ import annotations

import importlib
import sys
import warnings

import dmf
import dmf.memory
import pytest

from dmf.memory.ltm_hooks import ChromaLTMHook, FileLTMHook
from dmf.memory.ltm_hooks.chroma_hook import ChromaLTMHook as CanonicalChromaLTMHook
from dmf.memory.ltm_hooks.file_hook import FileLTMHook as CanonicalFileLTMHook


def test_canonical_reexports_preserve_class_identity() -> None:
    assert ChromaLTMHook is CanonicalChromaLTMHook
    assert FileLTMHook is CanonicalFileLTMHook
    assert dmf.memory.ChromaLTMHook is CanonicalChromaLTMHook
    assert dmf.memory.FileLTMHook is CanonicalFileLTMHook
    assert dmf.ChromaLTMHook is CanonicalChromaLTMHook
    assert dmf.FileLTMHook is CanonicalFileLTMHook


@pytest.mark.parametrize(
    ("module_name", "class_name", "canonical_class"),
    [
        ("dmf.memory.chroma_ltm", "ChromaLTMHook", CanonicalChromaLTMHook),
        ("dmf.memory.ltm_engine", "FileLTMHook", CanonicalFileLTMHook),
    ],
)
def test_legacy_shims_warn_and_preserve_class_identity(
    module_name: str,
    class_name: str,
    canonical_class: type,
) -> None:
    sys.modules.pop(module_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_module = importlib.import_module(module_name)

    deprecations = [
        warning for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
    assert len(deprecations) == 1
    assert getattr(legacy_module, class_name) is canonical_class
    assert legacy_module.__all__ == [class_name]


def test_canonical_imports_do_not_emit_deprecation_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("dmf.memory.ltm_hooks")
        importlib.import_module("dmf.memory.ltm_hooks.chroma_hook")
        importlib.import_module("dmf.memory.ltm_hooks.file_hook")

    assert not [
        warning for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]
