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

import json

from dmf.memory.query_understanding import QueryUnderstandingParser, parse_query_frame
from dmf.models.memory import QueryFrame


def test_query_frame_serialization_round_trips_with_injected_embedding() -> None:
    frame = parse_query_frame(
        "What is Alice Johnson's current job?",
        query_embedding=[0.1, 0.2, 0.3],
    )

    parsed = QueryFrame.from_dict(json.loads(json.dumps(frame.to_dict())))

    assert parsed == frame
    assert parsed.query_embedding == [0.1, 0.2, 0.3]


def test_query_embedding_is_optional_and_not_computed_by_parser() -> None:
    frame = parse_query_frame("Who is Alice Johnson's manager?")

    assert frame.query_embedding is None
    assert frame.answer_type == "person"


def test_extracts_current_latest_temporal_cues() -> None:
    frame = parse_query_frame("What is Nora's latest current address?")

    assert frame.temporal_intent == "latest"
    assert frame.historical_vs_current == "current"
    assert frame.filters["temporal_cues"] == ["current", "latest"]


def test_distinguishes_historical_focus() -> None:
    frame = parse_query_frame("Where did Marco live previously?")

    assert frame.answer_type == "place"
    assert frame.temporal_intent == "historical"
    assert frame.historical_vs_current == "historical"
    assert frame.predicate_focus == ["live"]


def test_detects_mixed_current_and_historical_focus() -> None:
    frame = parse_query_frame(
        "What is Jane's current role compared with her previous role?"
    )

    assert frame.temporal_intent == "current"
    assert frame.historical_vs_current == "mixed"


def test_detects_negative_polarity_and_negation_filters() -> None:
    frame = parse_query_frame("Which tools should Sam not use?")

    assert frame.polarity == "negative"
    assert frame.filters["polarity"] == "negative"
    assert frame.filters["negation_cues"] == ["not"]


def test_extracts_entities_subject_focus_and_predicate_cues() -> None:
    frame = parse_query_frame("Who is Alice Johnson's manager at Acme?")

    assert frame.entities == ["Alice Johnson", "Acme"]
    assert frame.subject_focus == ["Alice Johnson", "Acme"]
    assert frame.predicate_focus == ["relation"]
    assert frame.answer_type == "person"
    assert frame.aliases["Alice Johnson"] == ["alice johnson", "alice", "johnson"]


def test_strips_question_modal_prefix_from_capitalized_entity() -> None:
    frame = parse_query_frame("Would Nora likely have Dr. Vale books?")

    assert "Nora" in frame.entities
    assert "Would Nora" not in frame.entities
    assert "Nora" in frame.subject_focus
    assert "Would Nora" not in frame.subject_focus


def test_extracts_user_subject_focus_from_possessive_query() -> None:
    frame = parse_query_frame("What is my favorite coffee?")

    assert frame.subject_focus == ["favorite coffee", "user"]
    assert frame.predicate_focus == ["prefer"]
    assert frame.answer_type == "fact"


def test_extracts_surface_predicate_focus_for_action_queries() -> None:
    paint = parse_query_frame("When did Mira paint a mural?")
    camp = parse_query_frame("When is Mira planning on going camping?")
    speech = parse_query_frame("What speech did Nora give at the fundraiser?")
    volunteer = parse_query_frame("When did Nora volunteer at the youth center?")

    assert "paint" in paint.predicate_focus
    assert "go" in camp.predicate_focus
    assert "camp" in camp.predicate_focus
    assert "speech" in speech.predicate_focus
    assert "volunteer" in volunteer.predicate_focus


def test_extracts_past_completed_temporal_profile_for_when_did_queries() -> None:
    frame = parse_query_frame("When did Mira paint a mural?")

    assert frame.answer_type == "time"
    assert frame.historical_vs_current == "historical"
    assert frame.filters["temporal_mode"] == "past"
    assert frame.filters["event_status"] == "completed"


def test_extracts_future_planned_temporal_profile_for_plan_queries() -> None:
    frame = parse_query_frame("When is Mira planning on going camping?")

    assert frame.answer_type == "time"
    assert frame.historical_vs_current == "unspecified"
    assert frame.filters["temporal_mode"] == "future"
    assert frame.filters["event_status"] == "planned"


def test_past_plan_query_does_not_get_forced_into_future_mode() -> None:
    frame = parse_query_frame("When did Mira plan the camping trip?")

    assert frame.answer_type == "time"
    assert frame.historical_vs_current == "historical"
    assert frame.filters["temporal_mode"] == "past"
    assert frame.filters["event_status"] == "completed"


def test_extracts_absolute_time_marker_from_temporal_query() -> None:
    frame = parse_query_frame("What happened on June 9, 2023?")

    assert frame.filters["has_absolute_time"] is True
    assert frame.filters["time_anchor_text"] == "June 9, 2023"
    assert frame.filters["temporal_mode"] in {"past", "range"}


def test_query_frame_has_no_locomem_or_longmemeval_categories() -> None:
    frame = parse_query_frame("What is Nora's current favorite restaurant?")
    payload = json.dumps(frame.to_dict()).casefold()

    assert "locomo" not in payload
    assert "longmemeval" not in payload
    assert frame.answer_type in {
        "boolean",
        "fact",
        "list",
        "person",
        "place",
        "quantity",
        "reason",
        "time",
        "unknown",
    }


def test_parser_can_reuse_existing_nlp_analysis_entities_and_signals() -> None:
    class Analyzer:
        def analyze_interaction(self, text: str, is_system: bool = False):  # noqa: ARG002
            from dmf.models.analysis import AnalysisReport, InteractionSignals

            return AnalysisReport(
                info_density=0.0,
                sentiment_abs=0.0,
                entity_count=1,
                is_system_prompt=False,
                latency_ms=0.0,
                raw_metadata={"entities": [{"text": "Berlin", "label": "GPE"}]},
                signals=InteractionSignals(is_current_state=True),
            )

    frame = QueryUnderstandingParser(analyzer=Analyzer()).parse("Where is the office?")

    assert frame.entities == ["Berlin"]
    assert frame.temporal_intent == "current"
    assert frame.filters["temporal_mode"] == "current"
