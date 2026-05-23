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
