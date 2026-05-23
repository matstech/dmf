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

"""English-first conversational signal adapter."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from dmf.analysis.constants import (
    QUERY_AUX_STARTERS,
    QUERY_WH_STARTERS,
    REQUEST_ROOTS,
    STRUCTURAL_OBJECT_DEPS,
)
from dmf.models.analysis import InteractionSignals
from dmf.analysis.signals.base import SignalEvidence, SignalExtractionResult

CURRENT_STATE_CUES: tuple[str, ...] = (
    "currently",
    "current",
    "now",
    "at the moment",
)
PAST_STATE_CUES: tuple[str, ...] = (
    "used to",
    "previously",
    "before",
)
PREFERENCE_CUES: tuple[str, ...] = (
    "i prefer",
    "prefer",
    "my favorite",
    "i like",
    "i'd rather",
    "i would rather",
    "i usually choose",
)
CONSTRAINT_CUES: tuple[str, ...] = (
    "do not",
    "don't",
    "must not",
    "cannot",
    "can't",
    "never",
    "without",
)
CORRECTION_CUES: tuple[str, ...] = (
    "correction",
    "actually",
    "i mean",
    "sorry",
)
NEGATION_CUES: tuple[str, ...] = (
    "not",
    "do not",
    "don't",
    "must not",
    "cannot",
    "can't",
    "never",
    "without",
)
REPLACEMENT_PATTERN = re.compile(r"\bnot\s+.+?\s+but\s+.+", re.IGNORECASE)
QUERY_REQUEST_ROOTS: set[str] = set(REQUEST_ROOTS)
ACK_LIKE_STARTERS: tuple[str, ...] = (
    "understood",
    "noted",
    "got it",
    "i understand",
    "i will",
    "i'll",
    "your preference",
    "you set",
    "the current preference",
)
PERSONAL_FACT_FIRST_PERSON_TOKENS: set[str] = {"i", "we", "my", "our"}
PERSONAL_FACT_EXCLUDED_ROOTS: set[str] = {
    "ask",
    "check",
    "compare",
    "describe",
    "explain",
    "find",
    "guess",
    "help",
    "know",
    "list",
    "look",
    "mean",
    "outline",
    "propose",
    "recommend",
    "say",
    "suggest",
    "summarize",
    "tell",
    "think",
    "try",
    "understand",
    "wonder",
}
PERSONAL_FACT_STRUCTURAL_DEPS: set[str] = set(STRUCTURAL_OBJECT_DEPS | {"acomp"})
PERSONAL_FACT_CONTENT_POS: set[str] = {"ADJ", "NOUN", "NUM", "PROPN"}
QUANTITATIVE_ENTITY_LABELS: set[str] = {
    "CARDINAL",
    "DATE",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "TIME",
}
CONSTRAINT_ACTION_PRIORITY: tuple[str, ...] = (
    "use",
    "avoid",
    "call",
    "lookup",
    "forbid",
    "block",
)
CONSTRAINT_ACTION_CANONICAL: dict[str, str] = {
    "use": "use",
    "avoid": "use",
    "call": "use",
    "lookup": "use",
    "forbid": "use",
    "block": "use",
}

MATCHER_RULES: dict[str, tuple[str, ...]] = {
    "current_state": CURRENT_STATE_CUES,
    "past_state": PAST_STATE_CUES,
    "preference": PREFERENCE_CUES,
    "constraint": CONSTRAINT_CUES,
    "correction": CORRECTION_CUES,
    "negation": NEGATION_CUES,
}


class EnglishSignalAdapter:
    """Structured rule-based v1 for English conversational signals.
    
    Args:
        nlp: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, nlp: Language) -> None:
        """Pre-build the static matcher state used by ``extract()``.

        The adapter is intentionally cheap at runtime: all cue phrases are
        converted once into spaCy matcher patterns during construction, then
        reused for every interaction. We also keep a stable mapping from
        generated rule names back to their original lexical cue so audit data
        can expose readable `cue_phrases` without re-deriving them from text.
        """
        self._matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._rule_to_cue: dict[str, tuple[str, str]] = {}
        for label, cues in MATCHER_RULES.items():
            patterns = []
            for cue in cues:
                rule_name = self._rule_name(label, cue)
                self._rule_to_cue[rule_name] = (label, cue)
                patterns.append(nlp.make_doc(cue))
            self._matcher.add(label.upper(), patterns)

    def extract(
        self,
        text: str,
        doc: Doc,
    ) -> SignalExtractionResult:
        """Extract English conversational signals and evidence.
        
        Args:
            text: See the function signature and surrounding type hints.
            doc: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        evidence = self._collect_matcher_evidence(doc)
        evidence.extend(self._collect_replacement_evidence(text, doc))
        evidence.extend(self._collect_personal_fact_evidence(doc))
        signals = self._normalize_signals(evidence)
        is_query_like, is_ack_like = self._extract_pragmatic_flags(text, doc)
        topic_identity, topic_value = self._extract_topic_fields(doc, signals, is_ack_like)
        return SignalExtractionResult(
            signals=signals,
            signal_evidence=evidence,
            topic_identity=topic_identity,
            topic_value=topic_value,
            is_query_like=is_query_like,
            is_ack_like=is_ack_like,
        )

    def _extract_pragmatic_flags(self, text: str, doc: Doc) -> tuple[bool, bool]:
        """Return the pragmatic flags consumed by memory policy."""
        is_query_like = self._is_query_like(text, doc)
        is_ack_like = self._is_ack_like(text, is_query_like)
        return is_query_like, is_ack_like

    def _is_query_like(self, text: str, doc: Doc) -> bool:
        """Return True when the interaction is structurally query-like.

        This signal belongs in the language adapter rather than in
        ``TemporalMemory`` so language-specific heuristics stay confined to
        the NLP layer. The v1 detector is intentionally conservative: it
        catches explicit questions and imperative request prompts without
        trying to solve full dialogue-act classification.
        """
        if text.strip().endswith("?"):
            return True

        for sentence in doc.sents:
            if self._sentence_is_query_like(sentence):
                return True

        return False

    def _sentence_is_query_like(self, sentence: Span) -> bool:
        """Return True when one sentence is structurally query-like."""
        if sentence.text.strip().endswith("?"):
            return True

        first = self._first_content_token(sentence)
        if first is None:
            return False

        if self._starts_with_query_word(first):
            return True

        if self._has_interrogative_aux_inversion(sentence, first):
            return True

        return self._is_imperative_request_sentence(sentence)

    def _is_ack_like(self, text: str, is_query_like: bool) -> bool:
        """Return True when the text is a lightweight acknowledgement.

        The detector is intentionally narrow and lexical for English v1.
        It marks confirmations and short reformulations that are useful in
        dialogue but usually weak recall targets compared with source turns.
        """
        if is_query_like:
            return False

        stripped = text.strip().lower()
        return any(stripped.startswith(prefix) for prefix in ACK_LIKE_STARTERS)

    def _first_content_token(self, sentence: Span) -> Any | None:
        """Return the first non-space, non-punctuation token in a sentence."""
        return next(
            (token for token in sentence if not token.is_space and not token.is_punct),
            None,
        )

    def _starts_with_query_word(self, token: Any) -> bool:
        """Return True when the sentence starts with an English WH question cue."""
        return token.lower_ in QUERY_WH_STARTERS

    def _has_interrogative_aux_inversion(self, sentence: Span, first: Any) -> bool:
        """Return True for simple auxiliary-led English question inversion."""
        if first.lower_ not in QUERY_AUX_STARTERS:
            return False
        return any(
            token.dep_ in {"nsubj", "nsubjpass", "expl"} and token.i > first.i
            for token in sentence
        )

    def _is_imperative_request_sentence(self, sentence: Span) -> bool:
        """Return True for imperative request prompts such as ``Summarize ...``."""
        return (
            sentence.root.tag_ == "VB"
            and sentence.root.lemma_.lower() in QUERY_REQUEST_ROOTS
        )

    def _collect_matcher_evidence(self, doc: Doc) -> list[SignalEvidence]:
        """Collect cue-driven evidence spans from PhraseMatcher hits.

        The matcher finds explicit lexical cues such as "I prefer" or
        "used to". Each cue is then expanded to a slightly wider local span
        so downstream evidence is more informative than the cue token alone.
        """
        evidence: list[SignalEvidence] = []
        for match_id, start, end in self._matcher(doc):
            label = doc.vocab.strings[match_id].lower()
            cue_text = doc[start:end].text.casefold()
            rule_name = self._rule_name(label, cue_text)
            span = self._expand_span(doc, start, end)
            if span is None:
                continue
            evidence.append(
                self._build_evidence(
                    span=span,
                    source="matcher",
                    candidate_labels=self._labels_for_match(label),
                    matched_rules=[rule_name],
                    scores=self._scores_for_labels(self._labels_for_match(label)),
                )
            )
        return self._dedupe_evidence(evidence)

    def _collect_replacement_evidence(self, text: str, doc: Doc) -> list[SignalEvidence]:
        """Extract explicit replacement patterns such as ``not X but Y``."""
        match = REPLACEMENT_PATTERN.search(text)
        if match is None:
            return []
        span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
        if span is None:
            return []
        return [
            self._build_evidence(
                span=span,
                source="rule",
                candidate_labels=["replacement", "negation"],
                matched_rules=["replacement_not_but"],
                scores={"replacement": 1.0, "negation": 1.0},
            )
        ]

    def _collect_personal_fact_evidence(self, doc: Doc) -> list[SignalEvidence]:
        """Extract first-person factual statements with stable recall value."""
        evidence: list[SignalEvidence] = []
        for sentence in doc.sents:
            if self._is_personal_fact_sentence(sentence):
                evidence.append(
                    self._build_evidence(
                        span=sentence,
                        source="rule",
                        candidate_labels=["personal_fact"],
                        matched_rules=["personal_fact_statement"],
                        scores={"personal_fact": 1.0},
                    )
                )
            if self._is_quantitative_sentence(sentence):
                evidence.append(
                    self._build_evidence(
                        span=sentence,
                        source="rule",
                        candidate_labels=["quantitative_payload"],
                        matched_rules=["quantitative_statement"],
                        scores={"quantitative_payload": 1.0},
                    )
                )
        return evidence

    def _is_personal_fact_sentence(self, sentence: Span) -> bool:
        """Return True for declarative first-person statements with factual payload."""
        if self._sentence_is_query_like(sentence):
            return False
        if not self._has_first_person_reference(sentence):
            return False

        root = sentence.root
        if root.lemma_.lower() in PERSONAL_FACT_EXCLUDED_ROOTS:
            return False
        
        return self._has_personal_fact_payload(sentence)

    def _has_first_person_reference(self, sentence: Span) -> bool:
        """Return True when the sentence is grounded in first-person perspective."""
        return any(
            token.lower_ in PERSONAL_FACT_FIRST_PERSON_TOKENS
            and token.pos_ in {"DET", "PRON"}
            for token in sentence
        )

    def _has_personal_fact_payload(self, sentence: Span) -> bool:
        """Return True when the sentence carries stable factual content."""
        if any(token.ent_type_ for token in sentence):
            return True
        if any(token.like_num or token.pos_ == "NUM" for token in sentence):
            return True
        return any(
            token.dep_ in PERSONAL_FACT_STRUCTURAL_DEPS
            and token.pos_ in PERSONAL_FACT_CONTENT_POS
            and token.lower_ not in PERSONAL_FACT_FIRST_PERSON_TOKENS
            for token in sentence
        )

    def _is_quantitative_sentence(self, sentence: Span) -> bool:
        """Return True when a declarative sentence carries explicit numeric content."""
        if self._sentence_is_query_like(sentence):
            return False
        has_quantitative_marker = any(
            token.like_num or token.pos_ == "NUM" for token in sentence
        ) or any(entity.label_ in QUANTITATIVE_ENTITY_LABELS for entity in sentence.ents)
        if not has_quantitative_marker:
            return False
        return (
            self._is_personal_fact_sentence(sentence)
            or self._has_first_person_reference(sentence)
        )

    def _expand_span(self, doc: Doc, start: int, end: int) -> Span | None:
        """Expand a cue match to the sentence containing the cue.

        Sentence-level evidence is easier to audit and more stable than a
        token window while still remaining local to the matched interaction.
        """
        cue_span = doc[start:end]
        sentence = cue_span.sent
        return doc[sentence.start:sentence.end]

    def _labels_for_match(self, label: str) -> list[str]:
        if label == "constraint":
            return ["constraint", "negation"]
        return [label]

    def _extract_topic_fields(
        self,
        doc: Doc,
        signals: InteractionSignals,
        is_ack_like: bool = False,
    ) -> tuple[str | None, str | None]:
        """Extract the first stable topic identity/value pair available.

        The v1 policy is deliberately conservative: scan sentence by
        sentence and stop at the first pattern we know how to normalize
        deterministically. This keeps the adapter precise while the topic
        schema is still narrow.
        """
        for sentence in doc.sents:
            identity, value = self._extract_from_sentence(sentence, signals, is_ack_like)
            if identity is not None:
                return identity, value
        return None, None

    def _extract_from_sentence(
        self,
        sentence: Span,
        signals: InteractionSignals,
        is_ack_like: bool = False,
    ) -> tuple[str | None, str | None]:
        """Route one sentence through the narrow topic extractors.

        We only attempt extraction for patterns already represented in the
        current signal set so topic extraction stays aligned with the
        conversational signal pipeline instead of inventing a parallel NLP
        path inside TemporalMemory.
        """
        if signals.is_preference:
            favorite = self._extract_favorite_topic(sentence)
            if favorite[0] is not None:
                return favorite
            preferred = self._extract_prefer_topic(sentence)
            if preferred[0] is not None:
                return preferred

        if self._should_try_constraint_topic(sentence, signals, is_ack_like):
            constrained = self._extract_constraint_topic(sentence)
            if constrained[0] is not None:
                return constrained

        return None, None

    def _should_try_constraint_topic(
        self,
        sentence: Span,
        signals: InteractionSignals,
        is_ack_like: bool,
    ) -> bool:
        """Return True when constraint topicing is justified for this sentence."""
        if signals.is_constraint:
            return True
        return (
            self._looks_like_constraint_reference(sentence)
            or self._looks_like_ack_constraint(sentence)
            or (is_ack_like and self._constraint_anchor(sentence) is not None)
        )

    def _extract_prefer_topic(self, sentence: Span) -> tuple[str | None, str | None]:
        """Normalize ``prefer X`` into a stable preference topic/value pair."""
        prefer_token = next((t for t in sentence if t.lemma_.lower() == "prefer"), None)
        if prefer_token is None:
            return None, None
        value = self._extract_value_phrase(prefer_token)
        if value is None:
            return None, None
        return "preference|prefer", value

    def _extract_favorite_topic(self, sentence: Span) -> tuple[str | None, str | None]:
        """Normalize ``favorite <noun> is X`` into a field-specific preference topic."""
        favorite_token = next((t for t in sentence if t.lemma_.lower() == "favorite"), None)
        if favorite_token is None:
            return None, None

        field_token = next(
            (
                t for t in sentence
                if t.i > favorite_token.i and t.pos_ in {"NOUN", "PROPN"}
            ),
            None,
        )
        if field_token is None:
            return None, None

        value_token = next(
            (
                t for t in sentence
                if t.i > field_token.i and t.pos_ in {"NOUN", "PROPN", "ADJ"}
            ),
            None,
        )
        if value_token is None:
            return None, None

        return (
            f"preference|favorite|{self._normalize_token(field_token)}",
            self._build_phrase(value_token),
        )

    def _extract_constraint_topic(self, sentence: Span) -> tuple[str | None, str | None]:
        """Normalize simple negative usage constraints such as ``do not use X``."""
        verb = self._constraint_anchor(sentence)
        if verb is None:
            return None, None

        value = self._extract_constraint_value_phrase(verb)
        if value is None:
            return None, None

        action = CONSTRAINT_ACTION_CANONICAL.get(verb.lemma_.lower(), verb.lemma_.lower())
        return f"constraint|{action}", value

    def _looks_like_constraint_reference(self, sentence: Span) -> bool:
        """Return True for narrow assistant-style references to a constraint."""
        lowered = sentence.text.lower()
        if "constraint" not in lowered:
            return False
        return any(lemma in lowered for lemma in CONSTRAINT_ACTION_PRIORITY)

    def _looks_like_ack_constraint(self, sentence: Span) -> bool:
        """Return True for ack-like sentences that restate a constraint action."""
        lowered = sentence.text.lower()
        if "constraint" not in lowered and "rule" not in lowered:
            return False
        anchor = self._constraint_anchor(sentence)
        if anchor is None:
            return False
        if self._is_ack_like(sentence.text, is_query_like=False):
            return self._extract_value_phrase(anchor) is not None
        acknowledgement_lemmas = {"note", "understand", "set", "state"}
        if any(token.lemma_.lower() in acknowledgement_lemmas for token in sentence):
            return self._extract_value_phrase(anchor) is not None
        return False

    def _constraint_anchor(self, sentence: Span) -> Any | None:
        """Select the best anchor verb for constraint topic extraction."""
        for lemma in CONSTRAINT_ACTION_PRIORITY:
            token = next(
                (
                    t for t in sentence
                    if t.pos_ in {"VERB", "AUX"} and t.lemma_.lower() == lemma
                ),
                None,
            )
            if token is not None:
                return token

        return next(
            (
                t for t in sentence
                if t.pos_ in {"VERB", "AUX"} and t.lemma_.lower() not in {"do"}
            ),
            None,
        )

    def _extract_value_phrase(self, anchor: Span | Any) -> str | None:
        """Extract a short phrase representing the value attached to an anchor.

        The function prefers dependency-linked complements/objects and only
        falls back to the next plausible noun/adjective to the right. This
        keeps topic values compact and deterministic without attempting full
        claim extraction.
        """
        direct = self._direct_value_candidates(anchor)
        if direct:
            return self._build_phrase(direct[0])

        sentence = anchor.sent
        structural = [
            token
            for token in sentence
            if token.i > anchor.i
            and token.dep_ in {"dobj", "attr", "pobj", "acomp", "oprd"}
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        if structural:
            return self._build_phrase(structural[0])

        fallback = [
            token
            for token in sentence
            if token.i > anchor.i and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        if fallback:
            return self._build_phrase(fallback[0])

        return None

    def _direct_value_candidates(self, anchor: Span | Any) -> list[Any]:
        """Return direct dependency-linked value candidates for one anchor."""
        return [
            child
            for child in anchor.children
            if child.dep_ in {"dobj", "attr", "pobj", "acomp", "oprd", "nsubjpass", "obj"}
            and child.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]

    def _extract_constraint_value_phrase(self, anchor: Span | Any) -> str | None:
        """Extract a tight value phrase for constraint anchors.

        Constraint restatements are often long and end with abstract nouns like
        ``approach`` or ``solution``. For recall-time topicing we prefer the
        constrained object nearest to the action verb, so this extractor uses a
        much tighter search window than the generic value extractor.
        """
        sentence = anchor.sent
        direct_children = [
            child
            for child in anchor.children
            if child.dep_ in STRUCTURAL_OBJECT_DEPS and child.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        best_direct = self._best_constraint_candidate(direct_children, anchor)
        if best_direct is not None:
            return self._build_phrase(best_direct)

        window_structural = [
            token
            for token in sentence
            if anchor.i < token.i <= anchor.i + 7
            and token.dep_ in STRUCTURAL_OBJECT_DEPS
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        best_window_structural = self._best_constraint_candidate(window_structural, anchor)
        if best_window_structural is not None:
            return self._build_phrase(best_window_structural)

        fallback_window = [
            token
            for token in sentence
            if anchor.i < token.i <= anchor.i + 5
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]
        best_fallback = self._best_constraint_candidate(fallback_window, anchor)
        if best_fallback is not None:
            return self._build_phrase(best_fallback)

        return None

    def _best_constraint_candidate(self, candidates: list[Any], anchor: Span | Any) -> Any | None:
        """Return the best local constraint object without task-specific lexicons."""
        if not candidates:
            return None

        dep_priority = {
            "dobj": 0,
            "obj": 0,
            "nsubjpass": 0,
            "pobj": 1,
            "attr": 2,
            "oprd": 3,
        }
        pos_priority = {
            "NOUN": 0,
            "PROPN": 0,
            "ADJ": 1,
        }

        unique_candidates: list[Any] = []
        seen_ids: set[int] = set()
        for candidate in candidates:
            if candidate.i in seen_ids:
                continue
            seen_ids.add(candidate.i)
            unique_candidates.append(candidate)

        return min(
            unique_candidates,
            key=lambda token: (
                dep_priority.get(token.dep_, 9),
                pos_priority.get(token.pos_, 9),
                abs(token.i - anchor.i),
            ),
        )

    def _build_phrase(self, head: Span | Any) -> str:
        """Build a compact underscore-joined phrase around one head token.

        Only light local structure is included (`compound`, `amod`) so the
        normalized value remains stable and audit-friendly.
        """
        start = head.i
        end = head.i + 1

        while start > head.sent.start and head.doc[start - 1].dep_ in {"amod", "compound"}:
            start -= 1
        while end < head.sent.end and head.doc[end].dep_ in {"compound"}:
            end += 1

        parts = [
            self._normalize_token(token)
            for token in head.doc[start:end]
            if not token.is_stop and token.is_alpha
        ]
        return "_".join(part for part in parts if part)

    def _normalize_token(self, token: Span | Any) -> str:
        """Return the lowercase lemma used in topic identity/value strings."""
        return token.lemma_.lower().replace(" ", "_")

    def _normalize_signals(self, evidence: list[SignalEvidence]) -> InteractionSignals:
        """Collapse evidence items into the canonical ``InteractionSignals``.

        Normalization is deliberately simple in v1: labels are aggregated as
        presence/absence flags, while cue phrases and temporal markers are
        deduplicated for auditability.
        """
        label_counts: dict[str, int] = defaultdict(int)
        cue_phrases: list[str] = []
        temporal_markers: list[str] = []

        for item in evidence:
            for label in item["candidate_labels"]:
                label_counts[label] += 1
            for rule in item["matched_rules"]:
                if rule.startswith("negation_"):
                    continue
                cue = self._cue_from_rule(rule)
                if cue is None:
                    continue
                cue_phrases.append(cue)
                if rule.startswith("current_state_") or rule.startswith("past_state_"):
                    temporal_markers.append(cue)

        return InteractionSignals(
            is_current_state=label_counts["current_state"] > 0,
            is_past_state=label_counts["past_state"] > 0,
            is_preference=label_counts["preference"] > 0,
            is_constraint=label_counts["constraint"] > 0,
            is_correction=label_counts["correction"] > 0,
            has_negation=label_counts["negation"] > 0,
            has_replacement=label_counts["replacement"] > 0,
            personal_relevance=1.0 if label_counts["personal_fact"] > 0 else 0.0,
            quantitative_relevance=1.0 if label_counts["quantitative_payload"] > 0 else 0.0,
            temporal_markers=self._dedupe_values(temporal_markers),
            cue_phrases=self._compress_cue_phrases(self._dedupe_values(cue_phrases)),
        )

    def _build_evidence(
        self,
        span: Span,
        source: str,
        candidate_labels: list[str],
        matched_rules: list[str],
        scores: dict[str, float],
    ) -> SignalEvidence:
        """Build one serialisable evidence record tied to a text span."""
        return {
            "text": span.text,
            "start": span.start_char,
            "end": span.end_char,
            "source": source,
            "candidate_labels": candidate_labels,
            "matched_rules": matched_rules,
            "scores": scores,
        }

    def _scores_for_labels(self, labels: list[str]) -> dict[str, float]:
        """Return rule-based placeholder scores for the matched labels."""
        return {label: 1.0 for label in labels}

    def _cue_from_rule(self, rule_name: str) -> str | None:
        """Resolve a stable rule identifier back to its original lexical cue."""
        return self._rule_to_cue.get(rule_name, (None, None))[1]

    def _rule_name(self, label: str, cue: str) -> str:
        slug = cue.replace("'", "").replace(" ", "_")
        return f"{label}_{slug}"

    def _dedupe_values(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    def _compress_cue_phrases(self, cues: list[str]) -> list[str]:
        """Drop generic cues when a longer cue already subsumes them."""
        compressed: list[str] = []
        for cue in cues:
            if any(cue != other and cue in other for other in cues):
                continue
            compressed.append(cue)
        return compressed

    def _dedupe_evidence(self, evidence: list[SignalEvidence]) -> list[SignalEvidence]:
        """Merge duplicate evidence emitted by overlapping cue matches."""
        seen: set[tuple[Any, ...]] = set()
        deduped: list[SignalEvidence] = []
        for item in evidence:
            key = (
                item["start"],
                item["end"],
                tuple(item["candidate_labels"]),
                tuple(item["matched_rules"]),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
