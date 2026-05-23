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

"""Shared linguistic constants for deterministic analysis modules."""

from __future__ import annotations

SEMANTIC_POS_TAGS = frozenset({"NOUN", "VERB", "ADJ", "PROPN"})
PREDICATE_POS_TAGS = frozenset({"VERB", "NOUN", "ADJ"})
NOMINAL_VALUE_POS_TAGS = frozenset({"NOUN", "PROPN", "ADJ"})

STRUCTURAL_OBJECT_DEPS = frozenset({
    "dobj",
    "obj",
    "pobj",
    "attr",
    "oprd",
    "nsubjpass",
})
PREDICATE_DEPS = frozenset({
    "ROOT",
    "xcomp",
    "ccomp",
    "advcl",
    "relcl",
    "pcomp",
    "dobj",
    "obj",
    "attr",
    "oprd",
    "acomp",
    "conj",
})

QUERY_WH_STARTERS = frozenset({"what", "who", "when", "where", "why", "how"})
QUERY_AUX_STARTERS = frozenset({
    "do",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "will",
    "have",
    "has",
    "had",
})
REQUEST_ROOTS = frozenset({
    "summarize",
    "explain",
    "compare",
    "describe",
    "list",
    "tell",
    "propose",
    "outline",
})
