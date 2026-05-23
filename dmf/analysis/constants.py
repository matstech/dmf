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
