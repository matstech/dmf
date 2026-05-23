"""
tests/test_nlp_engine.py
------------------------
Unit tests for dmf/core/nlp_engine.py — NLPEngine feature extraction.

The spaCy model (en_core_web_sm) and VADER are loaded once at module scope
via a session-scoped fixture to keep the test suite fast.

Coverage:
  Information Density (ID)
    - Technical text produces high ID (>0.5).
    - Colloquial/filler text produces low ID (<0.35).
    - Empty text returns ID = 0.0 without raising.

  Absolute Sentiment (|S|)
    - Strongly emotional text produces high |S| (>0.8).
    - Neutral/factual text produces low |S| (<0.3).
    - Blank text returns |S| = 0.0 without raising.

  Entity Count (E)
    - Text with known named entities returns the correct count.
    - Entity raw_metadata contains the expected labels.
    - Text with no entities returns entity_count = 0.

  AnalysisReport integrity
    - latency_ms is positive (timer wired correctly).
    - raw_metadata contains pos_counts, token_count, entities keys.
    - is_system_prompt flag is forwarded correctly.

  System-prompt gating
    - is_system=True + analyze_system_prompt=False → neutral zero report.
    - is_system=True + analyze_system_prompt=True  → full analysis.
"""

import pytest

from dmf.models.analysis import AnalysisReport
from dmf.analysis.nlp_engine import NLPEngine
from dmf.utils.config import NLPConfig


# ---------------------------------------------------------------------------
# Fixtures — load heavy models once for the entire test module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_engine() -> NLPEngine:
    """NLPEngine with default config (analyze_system_prompt=False)."""
    return NLPEngine(config=NLPConfig())


@pytest.fixture(scope="module")
def system_prompt_enabled_engine() -> NLPEngine:
    """NLPEngine with analyze_system_prompt=True."""
    return NLPEngine(config=NLPConfig(analyze_system_prompt=True))


# ---------------------------------------------------------------------------
# Information Density (ID)
# ---------------------------------------------------------------------------

def test_technical_text_produces_high_information_density(default_engine: NLPEngine):
    """Technical prose with multiple NOUNs, VERBs, ADJs must yield ID > 0.5.

    Verified manually: 'The asynchronous function optimizes the database query'
    → 5 semantic tokens (ADJ, NOUN, VERB, NOUN, NOUN) out of 7 total → ID ≈ 0.71.
    """
    report = default_engine.analyze_interaction(
        "The asynchronous function optimizes the database query"
    )
    assert report.info_density > 0.5


def test_colloquial_text_produces_low_information_density(default_engine: NLPEngine):
    """Filler / social text with few semantic tokens must yield ID < 0.35."""
    report = default_engine.analyze_interaction(
        "Well, yeah, I mean, sure, ok then"
    )
    assert report.info_density < 0.35


def test_empty_text_returns_zero_information_density(default_engine: NLPEngine):
    """Empty string must return info_density = 0.0 without raising."""
    report = default_engine.analyze_interaction("")
    assert report.info_density == 0.0


# ---------------------------------------------------------------------------
# Absolute Sentiment (|S|)
# ---------------------------------------------------------------------------

def test_emotional_text_produces_high_sentiment_abs(default_engine: NLPEngine):
    """Strongly positive text must yield |S| > 0.8.

    Verified manually: VADER compound = 0.9362 → |S| = 0.936.
    """
    report = default_engine.analyze_interaction(
        "I absolutely love this amazing and wonderful day!"
    )
    assert report.sentiment_abs > 0.8


def test_neutral_factual_text_produces_low_sentiment_abs(default_engine: NLPEngine):
    """A dry technical statement must yield |S| < 0.3.

    Verified manually: VADER compound = 0.0 for purely structural prose
    with no evaluative vocabulary. 'saved' in the previous candidate was
    read as mildly positive by VADER (0.42), so a configuration-domain
    sentence with zero affective content is used instead.
    """
    report = default_engine.analyze_interaction(
        "The configuration file contains the database settings."
    )
    assert report.sentiment_abs < 0.3


def test_blank_text_returns_zero_sentiment_abs(default_engine: NLPEngine):
    """Whitespace-only input must return sentiment_abs = 0.0 without raising."""
    report = default_engine.analyze_interaction("   ")
    assert report.sentiment_abs == 0.0


# ---------------------------------------------------------------------------
# Entity Count (E) — NER
# ---------------------------------------------------------------------------

def test_entity_count_detects_known_named_entities(default_engine: NLPEngine):
    """'Apple is based in Cupertino' must yield entity_count = 2 (ORG + GPE).

    Verified manually with en_core_web_sm:
      Apple → ORG, Cupertino → GPE.
    """
    report = default_engine.analyze_interaction("Apple is based in Cupertino")
    assert report.entity_count == 2


def test_entity_raw_metadata_contains_correct_labels(default_engine: NLPEngine):
    """raw_metadata['entities'] must list Apple=ORG and Cupertino=GPE."""
    report = default_engine.analyze_interaction("Apple is based in Cupertino")
    entity_labels = {e["label"] for e in report.raw_metadata["entities"]}
    entity_texts = {e["text"] for e in report.raw_metadata["entities"]}
    assert "ORG" in entity_labels
    assert "GPE" in entity_labels
    assert "Apple" in entity_texts
    assert "Cupertino" in entity_texts


def test_text_without_entities_returns_zero_entity_count(default_engine: NLPEngine):
    """Plain common-noun text with no proper names must return entity_count = 0."""
    report = default_engine.analyze_interaction("the cat sat on the mat")
    assert report.entity_count == 0


# ---------------------------------------------------------------------------
# AnalysisReport integrity
# ---------------------------------------------------------------------------

def test_latency_ms_is_positive_after_analysis(default_engine: NLPEngine):
    """latency_ms must be > 0, confirming the ExecutionLatencyTimer is wired."""
    report = default_engine.analyze_interaction("The model processes the text.")
    assert report.latency_ms > 0


def test_raw_metadata_contains_required_keys(default_engine: NLPEngine):
    """raw_metadata must always expose NLP metrics and conversational-signal keys."""
    report = default_engine.analyze_interaction("The server handles the request.")
    assert "pos_counts" in report.raw_metadata
    assert "token_count" in report.raw_metadata
    assert "entities" in report.raw_metadata
    assert "signals" in report.raw_metadata
    assert "signal_evidence" in report.raw_metadata
    assert "topic_identity" in report.raw_metadata
    assert "topic_value" in report.raw_metadata


def test_raw_metadata_token_count_matches_text_length(default_engine: NLPEngine):
    """token_count in raw_metadata must equal spaCy's token count for the doc."""
    text = "The asynchronous function optimizes the database query"
    report = default_engine.analyze_interaction(text)
    # Verified manually: spaCy tokenises this sentence into 7 tokens.
    assert report.raw_metadata["token_count"] == 7


def test_is_system_prompt_false_is_forwarded_correctly(default_engine: NLPEngine):
    """is_system_prompt must be False when is_system=False (default)."""
    report = default_engine.analyze_interaction("Hello, how can I help?")
    assert report.is_system_prompt is False


def test_is_system_prompt_true_is_forwarded_when_analysis_enabled(
    system_prompt_enabled_engine: NLPEngine,
):
    """is_system_prompt must be True when is_system=True and analysis is on."""
    report = system_prompt_enabled_engine.analyze_interaction(
        "You are a helpful assistant.", is_system=True
    )
    assert report.is_system_prompt is True


# ---------------------------------------------------------------------------
# System-prompt gating
# ---------------------------------------------------------------------------

def test_system_prompt_is_skipped_when_flag_is_disabled(default_engine: NLPEngine):
    """analyze_system_prompt=False must return a neutral zero-weight report."""
    report = default_engine.analyze_interaction(
        "You are a concise and assertive assistant.", is_system=True
    )
    assert report.info_density == 0.0
    assert report.sentiment_abs == 0.0
    assert report.entity_count == 0
    assert report.is_system_prompt is True
    assert report.raw_metadata.get("skipped") is True


def test_system_prompt_is_analysed_when_flag_is_enabled(
    system_prompt_enabled_engine: NLPEngine,
):
    """analyze_system_prompt=True must run the full pipeline on the system prompt."""
    report = system_prompt_enabled_engine.analyze_interaction(
        "You are a helpful and knowledgeable assistant.", is_system=True
    )
    # Full analysis must produce non-neutral values for this content-rich prompt.
    assert report.info_density > 0.0
    assert report.is_system_prompt is True
    assert report.latency_ms > 0


def test_preference_signal_is_extracted_via_adapter(default_engine: NLPEngine):
    """Preference cues should populate both signals and signal_evidence."""
    report = default_engine.analyze_interaction("I prefer afternoon meetings.")
    assert report.signals.is_preference is True
    assert report.is_query_like is False
    assert "i prefer" in report.signals.cue_phrases
    assert any(
        "preference" in evidence["candidate_labels"]
        for evidence in report.raw_metadata["signal_evidence"]
    )
    assert report.topic_identity == "preference|prefer"
    assert report.topic_value == "afternoon_meeting"


def test_replacement_signal_produces_rule_based_evidence(default_engine: NLPEngine):
    """Replacement evidence should carry offsets and a stable rule identifier."""
    text = "Actually, not carbonara but risotto."
    report = default_engine.analyze_interaction(text)
    replacement = next(
        evidence
        for evidence in report.raw_metadata["signal_evidence"]
        if "replacement" in evidence["candidate_labels"]
    )
    assert report.signals.has_replacement is True
    assert replacement["source"] == "rule"
    assert replacement["matched_rules"] == ["replacement_not_but"]
    assert text[replacement["start"]:replacement["end"]] == replacement["text"]


def test_constraint_signal_is_normalized_without_touching_sentiment(default_engine: NLPEngine):
    """Constraint signals must remain separate from the legacy sentiment metric."""
    report = default_engine.analyze_interaction("Do not use external APIs in this solution.")
    assert report.signals.is_constraint is True
    assert report.signals.has_negation is True
    assert report.topic_identity == "constraint|use"
    assert report.topic_value == "external_api"
    assert isinstance(report.sentiment_abs, float)


def test_currently_sets_current_state_and_emits_stable_evidence(default_engine: NLPEngine):
    """`currently` must produce a current-state signal and auditable evidence."""
    report = default_engine.analyze_interaction("I currently work from Rome.")
    assert report.signals.is_current_state is True
    assert report.signals.is_past_state is False
    assert report.signals.temporal_markers == ["currently"]
    assert "currently" in report.signals.cue_phrases

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["current_state_currently"]
    )
    assert evidence == {
        "text": "I currently work from Rome.",
        "start": 0,
        "end": 27,
        "source": "matcher",
        "candidate_labels": ["current_state"],
        "matched_rules": ["current_state_currently"],
        "scores": {"current_state": 1.0},
    }


def test_used_to_sets_past_state_and_preference_with_stable_evidence(default_engine: NLPEngine):
    """`used to prefer` must remain compositional: past-state + preference."""
    report = default_engine.analyze_interaction("I used to prefer tea.")
    assert report.signals.is_past_state is True
    assert report.signals.is_preference is True
    assert report.signals.is_current_state is False
    assert report.signals.temporal_markers == ["used to"]
    assert report.signals.cue_phrases == ["used to", "prefer"]
    assert report.topic_identity == "preference|prefer"
    assert report.topic_value == "tea"

    past_evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["past_state_used_to"]
    )
    preference_evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["preference_prefer"]
    )

    assert past_evidence["candidate_labels"] == ["past_state"]
    assert past_evidence["scores"] == {"past_state": 1.0}
    assert preference_evidence["candidate_labels"] == ["preference"]
    assert preference_evidence["scores"] == {"preference": 1.0}


def test_not_x_but_y_sets_replacement_and_negation_with_rule_evidence(default_engine: NLPEngine):
    """`not X but Y` must emit explicit replacement evidence with offsets."""
    text = "Use risotto, not carbonara but pasta."
    report = default_engine.analyze_interaction(text)
    assert report.signals.has_replacement is True
    assert report.signals.has_negation is True

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["replacement_not_but"]
    )
    assert evidence == {
        "text": "not carbonara but pasta.",
        "start": 13,
        "end": 37,
        "source": "rule",
        "candidate_labels": ["replacement", "negation"],
        "matched_rules": ["replacement_not_but"],
        "scores": {"replacement": 1.0, "negation": 1.0},
    }


def test_i_prefer_sets_preference_and_emits_stable_matcher_rule(default_engine: NLPEngine):
    """Preference cues must expose a stable rule name and evidence shape."""
    report = default_engine.analyze_interaction("I prefer afternoon meetings.")
    assert report.signals.is_preference is True
    assert report.signals.is_constraint is False
    assert report.signals.cue_phrases == ["i prefer"]
    assert report.topic_identity == "preference|prefer"
    assert report.topic_value == "afternoon_meeting"

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["preference_i_prefer"]
    )
    assert evidence == {
        "text": "I prefer afternoon meetings.",
        "start": 0,
        "end": 28,
        "source": "matcher",
        "candidate_labels": ["preference"],
        "matched_rules": ["preference_i_prefer"],
        "scores": {"preference": 1.0},
    }


def test_do_not_use_sets_constraint_and_negation_with_stable_matcher_rule(default_engine: NLPEngine):
    """Constraint cues must expose both constraint and negation labels."""
    report = default_engine.analyze_interaction("Do not use external APIs.")
    assert report.signals.is_constraint is True
    assert report.signals.has_negation is True
    assert report.signals.cue_phrases == ["do not"]
    assert report.topic_identity == "constraint|use"
    assert report.topic_value == "external_api"

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["constraint_do_not"]
    )
    assert evidence == {
        "text": "Do not use external APIs.",
        "start": 0,
        "end": 25,
        "source": "matcher",
        "candidate_labels": ["constraint", "negation"],
        "matched_rules": ["constraint_do_not"],
        "scores": {"constraint": 1.0, "negation": 1.0},
    }


def test_declarative_personal_fact_sets_personal_relevance(default_engine: NLPEngine):
    """First-person factual statements should surface as recall-worthy personal facts."""
    report = default_engine.analyze_interaction(
        "I graduated with a degree in Business Administration."
    )
    assert report.signals.personal_relevance == 1.0
    assert report.is_query_like is False

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["personal_fact_statement"]
    )
    assert evidence["candidate_labels"] == ["personal_fact"]


def test_mixed_personal_fact_and_question_keeps_personal_relevance(default_engine: NLPEngine):
    """Mixed fact+question turns should stay query-like while preserving the fact signal."""
    report = default_engine.analyze_interaction(
        "I graduated with a degree in Business Administration. "
        "Do you have any advice on staying organized?"
    )
    assert report.is_query_like is True
    assert report.signals.personal_relevance == 1.0


def test_declarative_numeric_statement_sets_quantitative_relevance(default_engine: NLPEngine):
    """Declarative sentences with explicit numbers should surface quantitative salience."""
    report = default_engine.analyze_interaction("I spent 70 hours playing this game.")
    assert report.signals.quantitative_relevance == 1.0

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["quantitative_statement"]
    )
    assert evidence["candidate_labels"] == ["quantitative_payload"]


def test_question_with_number_does_not_set_quantitative_relevance(default_engine: NLPEngine):
    """Pure questions with numbers should not be promoted as quantitative memory content."""
    report = default_engine.analyze_interaction("Can you give me 3 tips for this game?")
    assert report.signals.quantitative_relevance == 0.0


def test_background_numeric_statement_does_not_set_quantitative_relevance(default_engine: NLPEngine):
    """Non-personal numeric background facts should not get the quantitative recall bonus."""
    report = default_engine.analyze_interaction(
        "The game was released in 2020 and received strong reviews."
    )
    assert report.signals.quantitative_relevance == 0.0


def test_actually_sets_correction_with_stable_matcher_rule(default_engine: NLPEngine):
    """Explicit correction cues should surface as correction evidence."""
    report = default_engine.analyze_interaction("Actually, the answer is risotto.")
    assert report.signals.is_correction is True
    assert "actually" in report.signals.cue_phrases

    evidence = next(
        item for item in report.raw_metadata["signal_evidence"]
        if item["matched_rules"] == ["correction_actually"]
    )
    assert evidence == {
        "text": "Actually, the answer is risotto.",
        "start": 0,
        "end": 32,
        "source": "matcher",
        "candidate_labels": ["correction"],
        "matched_rules": ["correction_actually"],
        "scores": {"correction": 1.0},
    }


def test_favorite_pattern_extracts_topic_identity_and_value(default_engine: NLPEngine):
    report = default_engine.analyze_interaction("My favorite dish is risotto.")
    assert report.signals.is_preference is True
    assert report.topic_identity == "preference|favorite|dish"
    assert report.topic_value == "risotto"


# ---------------------------------------------------------------------------
# Pragmatic flags
# ---------------------------------------------------------------------------

def test_question_like_text_sets_query_like_flag(default_engine: NLPEngine):
    report = default_engine.analyze_interaction("What hard implementation constraint did I set?")
    assert report.is_query_like is True
    assert report.raw_metadata["is_query_like"] is True


def test_imperative_request_sets_query_like_flag(default_engine: NLPEngine):
    report = default_engine.analyze_interaction("Summarize the core Aurora facts.")
    assert report.is_query_like is True


def test_constraint_statement_is_not_misclassified_as_query_like(default_engine: NLPEngine):
    report = default_engine.analyze_interaction("Do not use external APIs in this solution.")
    assert report.is_query_like is False


def test_cognitive_self_statement_is_not_promoted_to_personal_fact(default_engine: NLPEngine):
    report = default_engine.analyze_interaction("I think there has been a misunderstanding.")
    assert report.signals.personal_relevance == 0.0


def test_ack_like_confirmation_is_detected(default_engine: NLPEngine):
    report = default_engine.analyze_interaction(
        "Understood. I will avoid external APIs in the proposed solution."
    )
    assert report.is_ack_like is True
    assert report.is_query_like is False
    assert report.raw_metadata["is_ack_like"] is True
    assert report.topic_identity == "constraint|use"
    assert report.topic_value == "external_api"


def test_plain_constraint_is_not_misclassified_as_ack_like(default_engine: NLPEngine):
    report = default_engine.analyze_interaction(
        "Do not use external APIs in any proposed solution."
    )
    assert report.is_ack_like is False


# ---------------------------------------------------------------------------
# Recall-facing topic extraction
# ---------------------------------------------------------------------------

def test_constraint_reference_sentence_extracts_constraint_topic(default_engine: NLPEngine):
    report = default_engine.analyze_interaction(
        "You set a hard implementation constraint stating that no external APIs should be used in any proposed solution."
    )
    assert report.topic_identity == "constraint|use"
    assert report.topic_value == "external_api"


def test_long_constraint_restatement_prefers_local_external_api_value(default_engine: NLPEngine):
    report = default_engine.analyze_interaction(
        "The hard implementation rule is noted: I will not use external APIs, web calls, or remote lookup services in any proposed solution. "
        "If a solution requires external data, I will explicitly state that this constraint blocks that approach."
    )
    assert report.topic_identity == "constraint|use"
    assert report.topic_value == "external_api"
