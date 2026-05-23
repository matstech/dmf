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

"""Shared constants for memory retrieval, rendering, and cue matching."""

from __future__ import annotations

CARD_SEMANTIC_CHANNEL = "card_semantic"
CARD_SYMBOLIC_CHANNEL = "card_symbolic"
RAW_SEMANTIC_CHANNEL = "raw_semantic"
RAW_LEXICAL_CHANNEL = "raw_lexical"

COMMON_STOP_WORDS = frozenset({
    "a",
    "an",
    "and",
    "are",
    "at",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
})
QUERY_SPECIFIC_STOP_WORDS = COMMON_STOP_WORDS | frozenset({
    "about",
    "after",
    "anything",
    "before",
    "current",
    "currently",
    "former",
    "formerly",
    "latest",
    "moment",
    "most",
    "newest",
    "now",
    "past",
    "previous",
    "previously",
    "recent",
    "right",
    "tell",
    "these",
})

CURRENT_TEMPORAL_CUES = (
    "currently",
    "current",
    "now",
    "at the moment",
    "these days",
    "right now",
)
HISTORICAL_TEMPORAL_CUES = (
    "previously",
    "previous",
    "before",
    "used to",
    "in the past",
    "historically",
    "formerly",
    "old",
    "original",
)
LATEST_TEMPORAL_CUES = (
    "latest",
    "most recent",
    "newest",
    "last known",
    "last update",
    "recent",
)
RANGE_TEMPORAL_CUES = (
    "between",
    "since",
    "until",
    "from",
    "during",
    "timeline",
    "history of",
)
STALE_TEMPORAL_CUES = frozenset({
    "used to",
    "no longer",
    "formerly",
    "previously",
    "old",
    "outdated",
})

UNIX_TIMESTAMP_MIN = 100_000_000
UNIX_TIMESTAMP_MAX = 10_000_000_000
UTC_RENDER_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
CONTEXT_METADATA_SEPARATOR = " | "
CONTEXT_TIME_PREFIX = "time="
CONTEXT_TURN_PREFIX = "turn="
