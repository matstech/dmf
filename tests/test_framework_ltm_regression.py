# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

"""Framework-level LTM regression parity for Chroma and Qdrant backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

import dmf.memory.temporal_memory as temporal_memory_module
from dmf.memory.api import Memory
from dmf.memory.candidate_generation import CandidateGenerationConfig, CandidateGenerator
from dmf.memory.ltm_hooks.chroma_hook import ChromaLTMHook
from dmf.memory.ltm_hooks.qdrant_hook import QdrantLTMHook
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.analysis import AnalysisReport, InteractionProvenance, InteractionSignals
from dmf.models.memory import MemoryEntry, QueryFrame
from dmf.models.raw_ltm import RawRecallHit
from dmf.models.status import SurvivalStatus
from dmf.utils.config import DecayConfig, VectorConfig
from dmf.utils.config_loader import DMFConfig, LTMSettings, load_dmf_config


_VECTOR_CFG = VectorConfig(vector_dim=2, window_size=4)
_TEXT_VECTORS: dict[str, list[float]] = {
    "alpha": [1.0, 0.0],
    "beta": [0.0, 1.0],
    "edge": [0.8, 0.6],
    "I prefer green tea.": [1.0, 0.0],
    "I prefer black tea.": [0.9, 0.1],
    "I live in Rome.": [0.0, 1.0],
    "active note": [0.7, 0.3],
    "archived note": [1.0, 0.0],
}


@dataclass(frozen=True)
class _Backend:
    name: str
    hook: object


class _EmbeddingEngine:
    def get_embedding(self, text: str) -> np.ndarray:  # noqa: ARG002
        return np.array([1.0, 0.0], dtype=np.float32)


class _NLPStub:
    def analyze_interaction(self, text: str) -> AnalysisReport:
        if "prefer" in text:
            return _report(
                role="user",
                score=0.91,
                signals=InteractionSignals(is_preference=True, personal_relevance=1.0),
                topic_identity="preference|tea",
                topic_value="green" if "green" in text else "black",
            )
        if "live" in text:
            return _report(
                role="user",
                score=0.88,
                signals=InteractionSignals(is_current_state=True, personal_relevance=1.0),
                topic_identity="state|home",
                topic_value="Rome",
            )
        return _report(role="user", score=0.75)


class _FailingHook:
    def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
        pass

    def search_raw(
        self,
        query_vector: list[float],  # noqa: ARG002
        k: int = 5,  # noqa: ARG002
        *,
        recall_filter=None,  # noqa: ANN001, ARG002
    ) -> list[RawRecallHit]:
        raise RuntimeError("backend failure")

    def read_all(self) -> list[object]:
        raise RuntimeError("backend failure")


def _embed(text: str) -> np.ndarray:
    return np.array(_TEXT_VECTORS[text], dtype=np.float32)


def _report(
    *,
    role: str,
    score: float,
    signals: InteractionSignals | None = None,
    topic_identity: str | None = None,
    topic_value: str | None = None,
) -> AnalysisReport:
    return AnalysisReport(
        info_density=0.7,
        sentiment_abs=0.0,
        entity_count=1,
        is_system_prompt=False,
        latency_ms=0.0,
        survival_score=score,
        status=SurvivalStatus.HEALTHY if score > 0.6 else SurvivalStatus.CRITICAL,
        provenance=InteractionProvenance(role=role, source_turn=1),
        signals=signals or InteractionSignals(),
        topic_identity=topic_identity,
        topic_value=topic_value,
    )


def _entry(
    interaction_id: int,
    text: str,
    *,
    timestamp: float | None = None,
    report: AnalysisReport | None = None,
) -> MemoryEntry:
    return MemoryEntry(
        interaction_id=interaction_id,
        text=text,
        report=report or _report(role="user", score=0.8),
        vector=_embed(text),
        token_count=len(text.split()),
        timestamp=float(interaction_id if timestamp is None else timestamp),
    )


def _backends(tmp_path: Path, *, cards_enabled: bool = False) -> list[_Backend]:
    return [
        _Backend(
            "chroma",
            ChromaLTMHook(
                collection_name="framework_raw",
                persist_directory=tmp_path / "chroma",
                distance_threshold=0.4,
                vector_config=_VECTOR_CFG,
                embed_text=_embed,
                cards_enabled=cards_enabled,
                cards_path=tmp_path / "chroma_cards.jsonl",
            ),
        ),
        _Backend(
            "qdrant",
            QdrantLTMHook(
                collection_name="framework_raw",
                cards_collection_name="framework_cards",
                distance_threshold=0.4,
                vector_config=_VECTOR_CFG,
                embed_text=_embed,
                cards_enabled=cards_enabled,
                cards_path=tmp_path / "qdrant_cards.jsonl",
                client=QdrantClient(":memory:"),
            ),
        ),
    ]


def _collect(backends: list[_Backend], fn: Callable[[object], object]) -> list[object]:
    return [fn(backend.hook) for backend in backends]


def test_insertion_before_eviction_is_backend_neutral(
    tmp_path: Path,
) -> None:
    results = []
    for backend in _backends(tmp_path):
        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=10_000, pruning_frequency=999),
            vector_config=_VECTOR_CFG,
            ltm_hook=backend.hook,
        )
        first = tm.add_interaction("active note", _report(role="user", score=0.7), _embed("active note"))
        second = tm.add_interaction("archived note", _report(role="user", score=0.9), _embed("archived note"))
        results.append(
            (
                [(entry.interaction_id, entry.text, entry.token_count) for entry in tm.queue],
                [(item["interaction_id"], item["token_count"], item["status_effective"]) for item in tm.get_effective_state()],
                first.interaction_id,
                second.interaction_id,
                backend.hook.count(),
            )
        )

    assert results[0] == results[1]
    assert results[0][-1] == 0


def test_eviction_archives_equivalent_raw_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(temporal_memory_module.time, "time", lambda: 1000.0)
    results = []
    for backend in _backends(tmp_path):
        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=1, pruning_frequency=999),
            vector_config=_VECTOR_CFG,
            ltm_hook=backend.hook,
        )
        tm.add_interaction("I prefer green tea.", _report(role="user", score=0.2), _embed("I prefer green tea."))
        tm.add_interaction("I live in Rome.", _report(role="user", score=0.2), _embed("I live in Rome."))
        results.append([record.to_dict() for record in backend.hook.read_all()])

    assert results[0] == results[1]
    assert [record["record_id"] for record in results[0]] == ["record:0", "record:1"]


def test_raw_recall_order_and_threshold_are_backend_neutral(tmp_path: Path) -> None:
    results = _collect(
        _backends(tmp_path),
        lambda hook: (
            hook.archive(_entry(1, "alpha")),
            hook.archive(_entry(2, "beta")),
            hook.archive(_entry(3, "edge")),
            [(hit.record.record_id, hit.rank_hint) for hit in hook.search_raw([1.0, 0.0], k=3)],
        )[-1],
    )

    assert results[0] == results[1] == [("record:1", 0), ("record:3", 1)]


def test_contextualized_recall_and_rendering_are_backend_neutral(tmp_path: Path) -> None:
    results = []
    for backend in _backends(tmp_path):
        backend.hook.archive(_entry(1, "I prefer green tea.", timestamp=1000.0))
        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=10_000, pruning_frequency=999),
            vector_config=_VECTOR_CFG,
            ltm_hook=backend.hook,
            nlp_engine=_NLPStub(),
        )
        tm.add_interaction("active note", _report(role="user", score=0.9), _embed("active note"))
        hits = tm.get_raw_recall_hits(np.array([1.0, 0.0], dtype=np.float32), k=2)
        contextualized = tm.contextualize_raw_recall_hits(hits)
        reranked = tm.rerank_contextualized_recall_candidates(contextualized)
        results.append(
            (
                [(item.record.record_id, item.report.topic_identity, item.suppression_reason) for item in reranked],
                tm.get_full_context(np.array([1.0, 0.0], dtype=np.float32)),
                tm.get_recall_diagnostics(),
            )
        )

    assert results[0] == results[1]
    assert "I prefer green tea." in results[0][1]
    assert "active note" in results[0][1]


def test_candidate_generation_merges_raw_and_card_semantic_channels(
    tmp_path: Path,
) -> None:
    results = []
    for backend in _backends(tmp_path, cards_enabled=True):
        backend.hook.archive(
            _entry(
                1,
                "I prefer green tea.",
                report=_report(
                    role="user",
                    score=0.9,
                    signals=InteractionSignals(is_preference=True),
                    topic_identity="preference|tea",
                    topic_value="green",
                ),
            )
        )
        generator = CandidateGenerator(
            cards=backend.hook.card_store.read_all(),
            ltm_hook=backend.hook,
            raw_records=backend.hook.read_all(),
            config=CandidateGenerationConfig(
                enable_card_symbolic=False,
                enable_raw_lexical=False,
                card_prefetch_k=2,
                raw_prefetch_k=2,
            ),
        )
        pool = generator.generate(QueryFrame(query_text="tea", query_embedding=[1.0, 0.0]))
        results.append([candidate.to_dict() for candidate in pool.candidates])

    assert results[0] == results[1]
    assert results[0][0]["evidence_id"] == "record:1"
    assert results[0][0]["source"] == "card_semantic+raw_semantic"


def test_structured_memory_api_results_are_backend_neutral(tmp_path: Path) -> None:
    results = []
    for backend in _backends(tmp_path):
        backend.hook.archive(_entry(1, "I prefer green tea."))
        memory = Memory.from_dmf_config(
            DMFConfig(),
            TemporalMemory(vector_config=_VECTOR_CFG, ltm_hook=backend.hook),
            _EmbeddingEngine(),
        )
        results.append([item.to_dict() for item in memory.retrieve("What tea do I prefer?")])

    assert results[0] == results[1]
    assert [item["evidence_id"] for item in results[0]] == ["record:1"]


def test_explicit_hook_override_and_default_framework_config() -> None:
    explicit = object()
    cfg = DMFConfig(ltm=LTMSettings(storage_type="qdrant"))

    tm = TemporalMemory.from_dmf_config(cfg, ltm_hook=explicit)
    loaded = load_dmf_config()

    assert tm.ltm_hook is explicit
    assert loaded.ltm.storage_type == "chroma"
    assert loaded.ltm.recall_limit == 5
    assert loaded.ltm.distance_threshold == 0.7


def test_backend_errors_propagate_from_temporal_and_memory_api() -> None:
    tm = TemporalMemory(ltm_hook=_FailingHook())

    with pytest.raises(RuntimeError, match="backend failure"):
        tm.get_raw_recall_hits(np.array([1.0, 0.0], dtype=np.float32))

    memory = Memory.from_dmf_config(DMFConfig(), tm, _EmbeddingEngine())
    with pytest.raises(RuntimeError, match="backend failure"):
        memory.retrieve("What failed?")
