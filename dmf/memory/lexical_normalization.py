"""Shared lexical normalization for structured retrieval modules.

The structured query parser, candidate generator, and reranker all need the
same tokenisation and lemmatisation semantics. Centralising that logic here
avoids module-local token maps and keeps matching behaviour consistent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from dmf.analysis.constants import PREDICATE_DEPS, PREDICATE_POS_TAGS
from dmf.utils.constants import DEFAULT_SPACY_MODEL


_SEPARATOR_RE = re.compile(r"[_/]+")
_WHITESPACE_RE = re.compile(r"\s+")
_PREDICATE_POS = PREDICATE_POS_TAGS
_PREDICATE_DEPS = PREDICATE_DEPS
_PREDICATE_BLOCKLIST = frozenset(
    {
        "be",
        "can",
        "could",
        "describe",
        "do",
        "have",
        "how",
        "list",
        "should",
        "summarize",
        "tell",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "would",
    }
)


@dataclass(frozen=True)
class LexicalAnalysis:
    """Cached lexical features for one short text span.
    
    Args:
        normalized: See the function signature and surrounding type hints.
        non_stop_lemmas: See the function signature and surrounding type hints.
        content_lemmas: See the function signature and surrounding type hints.
        predicate_lemmas: See the function signature and surrounding type hints.
        entity_spans: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    normalized: str
    non_stop_lemmas: tuple[str, ...]
    content_lemmas: tuple[str, ...]
    predicate_lemmas: tuple[str, ...]
    entity_spans: tuple[str, ...]


def normalize_text(text: str) -> str:
    """Return casefolded, whitespace-normalized text.
    
    Args:
        text: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    return _WHITESPACE_RE.sub(" ", text.casefold()).strip()


def dedupe_casefold(values: Iterable[str]) -> list[str]:
    """De-duplicate values by casefold while preserving insertion order.
    
    Args:
        values: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def lemma_terms(
    text: str,
    *,
    extra_stop_words: Iterable[str] = (),
    content_only: bool = False,
) -> set[str]:
    """Return matchable lemmatised terms for one text span.
    
    Args:
        text: See the function signature and surrounding type hints.
        extra_stop_words: See the function signature and surrounding type hints.
        content_only: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    analysis = analyze_lexical_text(text)
    blocked = {normalize_text(value) for value in extra_stop_words if value}
    source = analysis.content_lemmas if content_only else analysis.non_stop_lemmas
    return {term for term in source if term and term not in blocked}


def predicate_terms(
    text: str,
    *,
    blocked_terms: Iterable[str] = (),
) -> list[str]:
    """Return ordered predicate-like lemmas extracted from a query span.
    
    Args:
        text: See the function signature and surrounding type hints.
        blocked_terms: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    blocked = {normalize_text(value) for value in blocked_terms if value}
    return [
        term
        for term in analyze_lexical_text(text).predicate_lemmas
        if term and term not in blocked
    ]


def entity_spans(text: str) -> list[str]:
    """Return spaCy-recognised entities for one text span.
    
    Args:
        text: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    return list(analyze_lexical_text(text).entity_spans)


@lru_cache(maxsize=4096)
def analyze_lexical_text(text: str) -> LexicalAnalysis:
    """Analyse one short text span with cached spaCy features.
    
    Args:
        text: See the function signature and surrounding type hints.
    
    Returns:
        See the return type annotation.
    
    Raises:
        None.
    """
    doc = _doc_for_text(text)
    non_stop_lemmas: list[str] = []
    content_lemmas: list[str] = []
    predicate_lemmas: list[str] = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = _token_lemma(token)
        if not lemma:
            continue
        if not token.is_stop:
            non_stop_lemmas.append(lemma)
        if token.pos_ in _PREDICATE_POS and not token.is_stop:
            content_lemmas.append(lemma)
        if _is_predicate_candidate(token):
            predicate_lemmas.append(lemma)
    entities = [
        _clean_entity(entity.text)
        for entity in doc.ents
        if _clean_entity(entity.text)
    ]
    return LexicalAnalysis(
        normalized=normalize_text(doc.text),
        non_stop_lemmas=tuple(dedupe_casefold(non_stop_lemmas)),
        content_lemmas=tuple(dedupe_casefold(content_lemmas)),
        predicate_lemmas=tuple(dedupe_casefold(predicate_lemmas)),
        entity_spans=tuple(dedupe_casefold(entities)),
    )


@lru_cache(maxsize=4096)
def _doc_for_text(text: str) -> Doc:
    return _nlp()(_SEPARATOR_RE.sub(" ", text.strip()))


@lru_cache(maxsize=1)
def _nlp() -> Language:
    try:
        return spacy.load(DEFAULT_SPACY_MODEL)
    except OSError:
        return spacy.blank("en")


def _token_lemma(token: Token) -> str:
    lemma = normalize_text(token.lemma_ or "")
    if not lemma or lemma == "-pron-":
        lemma = normalize_text(token.text)
    if lemma == normalize_text(token.text) and _needs_verbish_fallback(token):
        fallback = _verbish_lemma(token.text)
        if fallback:
            return fallback
    return lemma


def _is_predicate_candidate(token: Token) -> bool:
    if token.pos_ not in _PREDICATE_POS:
        return False
    if token.dep_ not in _PREDICATE_DEPS:
        return False
    lemma = _token_lemma(token)
    if not lemma or lemma in _PREDICATE_BLOCKLIST:
        return False
    return True


def _clean_entity(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip(" .,:;!?\"'")).strip()


def _needs_verbish_fallback(token: Token) -> bool:
    return (
        token.pos_ in {"NOUN", "VERB"}
        and normalize_text(token.text).endswith("ing")
        and token.dep_ in _PREDICATE_DEPS
    )


@lru_cache(maxsize=256)
def _verbish_lemma(text: str) -> str:
    doc = _nlp()(f"to {text}")
    if len(doc) < 2:
        return normalize_text(text)
    lemma = normalize_text(doc[-1].lemma_ or "")
    return lemma or normalize_text(text)
