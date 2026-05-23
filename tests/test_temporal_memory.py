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

"""
tests/test_temporal_memory.py
------------------------------
Unit tests for Module 4 Phase 2 + Phase 3 + Phase 4:
  - ``MemoryEntry`` (src/dmf/data_models/memory.py)
  - ``TemporalMemory`` (src/dmf/core/temporal_memory.py)
  - ``LTMHook`` / ``NullLTMHook`` (src/dmf/data_models/ltm_hook.py)

No NLP models, no embedding inference, no I/O.
Vectors are hand-crafted numpy arrays; AnalysisReports are built directly.

Coverage
--------
  MemoryEntry
    - All six fields stored correctly.
    - ``omega`` property returns survival_score from report.
    - ``omega`` property returns 0.0 when survival_score is None.
    - ``status`` property proxies report.status.

  TemporalMemory — construction
    - Default config is applied when no args given.
    - Custom DecayConfig is stored correctly.
    - Queue is empty after construction.
    - Matrix is empty after construction.
    - next_id starts at 0.

  TemporalMemory — add_interaction
    - Entry is appended to the queue.
    - Returned entry is the same object as the one in the queue.
    - interaction_id is monotonically increasing (0, 1, 2, …).
    - next_id increments by 1 after each insertion.
    - token_count is positive for non-empty text.
    - token_count is 0 for empty string.
    - Matrix size grows with each insertion (up to window_size).
    - timestamp is a recent Unix epoch float.

  TemporalMemory — get_effective_state (decay view)
    - Empty queue returns empty list.
    - Single entry has Δn = 0 → omega_eff == omega exactly.
    - After N insertions, oldest entry has Δn = N − 1.
    - omega_eff ≤ omega for all entries.
    - omega_eff is strictly less for older entries (decay is monotone).
    - INERTIA: HEALTHY message has higher omega_eff than CRITICAL message
      at the same age (Δn = 20).
    - All required keys are present in every result dict.
    - status_effective reflects omega_eff tier (not original omega tier).

  TemporalMemory — get_total_tokens
    - Returns 0 for empty queue.
    - Returns sum of individual token counts.
    - Stays consistent with manual sum.

  TemporalMemory — size / is_empty properties
    - is_empty True for fresh instance.
    - is_empty False after one insertion.
    - size matches queue length.

  LTMHook / NullLTMHook
    - NullLTMHook satisfies the LTMHook Protocol.
    - NullLTMHook.archive is a no-op (no exception raised).
    - Custom hook injected and called on eviction.

  TemporalMemory — _get_pruning_candidates
    - Empty queue returns empty list.
    - HEALTHY entries are excluded from candidates.
    - CRITICAL entries precede UNSTABLE entries.
    - Within each bucket entries are sorted oldest-first (interaction_id ASC).
    - An all-HEALTHY queue returns an empty candidate list.

  TemporalMemory — prune_to_budget
    - No eviction when already under budget.
    - Returns list of evicted entries.
    - Each evicted entry is removed from the queue.
    - Each evicted entry is passed to LTM hook.
    - SPEC SCENARIO: 2 HEALTHY + 5 CRITICAL + 3 UNSTABLE with budget=6
      → exactly 4 CRITICALs evicted, both HEALTHYs survive.
    - All-HEALTHY queue: budget not met but no eviction (HEALTHY protected).
    - Evicted entries are also removed from InteractionMatrix.

  TemporalMemory — periodic_cleanup
    - Does not evict entries above hard_kill_threshold.
    - Evicts entries below hard_kill_threshold (including HEALTHY).
    - Returns list of evicted entries.
    - Triggered automatically by add_interaction every pruning_frequency turns.

  TemporalMemory — get_full_context
    - Empty queue → only active header present.
    - Non-empty queue → active entry texts appear under active header.
    - Recalled raw records precede the active header.
    - Blank line separates recalled and active sections.
    - Active texts appear in queue order (oldest → newest).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest


from dmf.models.analysis import (
    AnalysisReport,
    InteractionProvenance,
    InteractionSignals,
    MemoryLineage,
)
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.ltm_hook import LTMHook, NullLTMHook
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import ContextualizedRecallCandidate, RawLTMRecord, RawRecallHit
from dmf.models.status import SurvivalStatus, classify_survival_status
from dmf.utils.config import DecayConfig, PruningPriorityConfig, VectorConfig


# ---------------------------------------------------------------------------
# Test fixtures — no NLP / embedding inference
# ---------------------------------------------------------------------------

def _make_report(
    omega: float | None = 0.72,
    info_density: float = 0.5,
    sentiment_abs: float = 0.2,
    entity_count: int = 2,
    latency_ms: float = 1.0,
    signals: InteractionSignals | None = None,
    topic_identity: str | None = None,
    topic_value: str | None = None,
    is_query_like: bool = False,
    is_ack_like: bool = False,
) -> AnalysisReport:
    """Construct a minimal scored AnalysisReport without running any models."""
    report = AnalysisReport(
        info_density=info_density,
        sentiment_abs=sentiment_abs,
        entity_count=entity_count,
        is_system_prompt=False,
        latency_ms=latency_ms,
        signals=signals or InteractionSignals(),
        topic_identity=topic_identity,
        topic_value=topic_value,
        is_query_like=is_query_like,
        is_ack_like=is_ack_like,
    )
    report.survival_score = omega
    if omega is not None:
        report.status = classify_survival_status(
            omega=omega,
            critical_threshold=_DECAY_CFG.critical_threshold,
            healthy_threshold=_DECAY_CFG.healthy_threshold,
        )
    return report


def _unit_vector(dim: int = 8, seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised vector of shape (dim,)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# Shared config — tiny window, no model inference.
_VECTOR_CFG = VectorConfig(window_size=5, vector_dim=8)
_DECAY_CFG = DecayConfig()  # approved defaults

# Config that disables ALL automatic pruning (for tests that control it manually).
_NO_PRUNE_CFG = DecayConfig(token_budget=999_999, pruning_frequency=999_999)


# ---------------------------------------------------------------------------
# Recording LTM hook — captures archived entries for test assertions
# ---------------------------------------------------------------------------

class RecordingLTMHook:
    """Test double: records every archive() call for assertion.

    Implements the raw-only LTMHook protocol used by the pivoted runtime.
    """

    def __init__(self) -> None:
        self.archived: list[MemoryEntry] = []

    def archive(self, entry: MemoryEntry) -> None:
        self.archived.append(entry)

    def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
        """No-op raw search — RecordingLTMHook is write-only by design."""
        return []

    def read_all(self) -> list:
        """No-op read_all — RecordingLTMHook is write-only by design."""
        return []


class _FakeNLPEngine:
    """Minimal test double for recall-time contextualization."""

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def analyze_interaction(self, text: str, is_system: bool = False) -> AnalysisReport:  # noqa: ARG002
        self.seen_texts.append(text)
        return AnalysisReport(
            info_density=0.4,
            sentiment_abs=0.0,
            entity_count=0,
            is_system_prompt=False,
            latency_ms=1.0,
            topic_identity="preference|prefer",
            topic_value="coffee",
            signals=InteractionSignals(
                is_current_state=True,
                is_preference=True,
            ),
        )


class _MappedRecallNLPEngine:
    """Test double that returns a predefined report per recalled text."""

    def __init__(self, mapping: dict[str, AnalysisReport]) -> None:
        self._mapping = mapping

    def analyze_interaction(self, text: str, is_system: bool = False) -> AnalysisReport:  # noqa: ARG002
        return self._mapping[text]


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    """MemoryEntry must store all six fields and expose clean properties."""

    def _make_entry(self, omega: float | None = 0.80) -> MemoryEntry:
        report = _make_report(omega=omega)
        vec = _unit_vector()
        return MemoryEntry(
            interaction_id=0,
            text="Hello world",
            report=report,
            vector=vec,
            token_count=2,
            timestamp=1_700_000_000.0,
        )

    def test_interaction_id_stored(self) -> None:
        assert self._make_entry().interaction_id == 0

    def test_text_stored(self) -> None:
        assert self._make_entry().text == "Hello world"

    def test_report_stored(self) -> None:
        entry = self._make_entry()
        assert isinstance(entry.report, AnalysisReport)

    def test_vector_stored(self) -> None:
        entry = self._make_entry()
        assert isinstance(entry.vector, np.ndarray)

    def test_token_count_stored(self) -> None:
        assert self._make_entry().token_count == 2

    def test_timestamp_stored(self) -> None:
        assert self._make_entry().timestamp == 1_700_000_000.0

    def test_omega_property_returns_survival_score(self) -> None:
        entry = self._make_entry(omega=0.75)
        assert entry.omega == 0.75

    def test_omega_property_returns_zero_when_unscored(self) -> None:
        """survival_score=None must not raise; returns 0.0 (dead weight)."""
        entry = self._make_entry(omega=None)
        assert entry.omega == 0.0

    def test_status_property_proxies_report(self) -> None:
        entry = self._make_entry(omega=0.80)
        assert entry.status == SurvivalStatus.HEALTHY

    def test_provenance_property_proxies_report(self) -> None:
        entry = self._make_entry(omega=0.80)
        entry.report.provenance = InteractionProvenance(
            role="user",
            source_turn=3,
            is_user_correction=True,
        )
        assert entry.provenance.role == "user"
        assert entry.provenance.source_turn == 3
        assert entry.provenance.is_user_correction is True


# ---------------------------------------------------------------------------
# TemporalMemory — construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """Fresh TemporalMemory must be empty and have sensible defaults."""

    def test_default_decay_config_applied(self) -> None:
        tm = TemporalMemory()
        assert tm.config == DecayConfig()

    def test_custom_decay_config_stored(self) -> None:
        cfg = DecayConfig(lambda_decay=0.05)
        tm = TemporalMemory(decay_config=cfg)
        assert tm.config.lambda_decay == 0.05

    def test_queue_empty_after_construction(self) -> None:
        tm = TemporalMemory()
        assert len(tm.queue) == 0

    def test_matrix_empty_after_construction(self) -> None:
        tm = TemporalMemory()
        assert tm.matrix.is_empty

    def test_next_id_starts_at_zero(self) -> None:
        tm = TemporalMemory()
        assert tm.next_id == 0

    def test_is_empty_true_after_construction(self) -> None:
        tm = TemporalMemory()
        assert tm.is_empty is True

    def test_size_zero_after_construction(self) -> None:
        tm = TemporalMemory()
        assert tm.size == 0


# ---------------------------------------------------------------------------
# TemporalMemory — add_interaction
# ---------------------------------------------------------------------------

class TestAddInteraction:
    """add_interaction must update the queue, matrix, and counter correctly."""

    def setup_method(self) -> None:
        self.tm = TemporalMemory(decay_config=_DECAY_CFG, vector_config=_VECTOR_CFG)

    def _add(self, text: str = "test", omega: float = 0.70, seed: int = 0) -> MemoryEntry:
        report = _make_report(omega=omega)
        vec = _unit_vector(dim=8, seed=seed)
        return self.tm.add_interaction(text, report, vec)

    # --- queue bookkeeping ---

    def test_entry_appended_to_queue(self) -> None:
        entry = self._add()
        assert len(self.tm.queue) == 1
        assert self.tm.queue[-1] is entry

    def test_returned_entry_is_in_queue(self) -> None:
        entry = self._add()
        assert entry in self.tm.queue

    def test_interaction_id_is_zero_for_first_entry(self) -> None:
        entry = self._add()
        assert entry.interaction_id == 0

    def test_interaction_ids_are_monotonically_increasing(self) -> None:
        ids = [self._add(seed=i).interaction_id for i in range(5)]
        assert ids == list(range(5))

    def test_next_id_increments_after_each_add(self) -> None:
        assert self.tm.next_id == 0
        self._add(seed=0)
        assert self.tm.next_id == 1
        self._add(seed=1)
        assert self.tm.next_id == 2

    # --- token counting ---

    def test_token_count_is_positive_for_non_empty_text(self) -> None:
        entry = self._add(text="The mitochondria is the powerhouse of the cell.")
        assert entry.token_count > 0

    def test_token_count_is_zero_for_empty_string(self) -> None:
        entry = self._add(text="")
        assert entry.token_count == 0

    def test_token_count_is_consistent_with_tiktoken(self) -> None:
        """token_count on the entry must match a fresh tiktoken encode."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        text = "Artificial intelligence is transforming software engineering."
        entry = self._add(text=text)
        assert entry.token_count == len(enc.encode(text))

    # --- geometry sync ---

    def test_matrix_size_grows_with_insertions(self) -> None:
        for i in range(4):
            self._add(seed=i)
        assert self.tm.matrix.size == 4

    def test_matrix_capped_at_window_size(self) -> None:
        for i in range(10):   # window_size = 5
            self._add(seed=i)
        assert self.tm.matrix.size == 5

    # --- timestamp ---

    def test_timestamp_is_recent(self) -> None:
        before = time.time()
        entry = self._add()
        after = time.time()
        assert before <= entry.timestamp <= after


# ---------------------------------------------------------------------------
# TemporalMemory — get_effective_state (decay view)
# ---------------------------------------------------------------------------

class TestGetEffectiveState:
    """get_effective_state must return correctly decayed entries."""

    def setup_method(self) -> None:
        self.tm = TemporalMemory(decay_config=_DECAY_CFG, vector_config=_VECTOR_CFG)

    def _add(self, text: str = "msg", omega: float = 0.70, seed: int = 0) -> MemoryEntry:
        return self.tm.add_interaction(text, _make_report(omega=omega), _unit_vector(8, seed))

    # --- empty queue ---

    def test_empty_queue_returns_empty_list(self) -> None:
        assert self.tm.get_effective_state() == []

    # --- Δn calculation ---

    def test_single_entry_delta_n_is_zero(self) -> None:
        self._add(omega=0.72)
        state = self.tm.get_effective_state()
        assert state[0]["delta_n"] == 0

    def test_single_entry_omega_eff_equals_omega(self) -> None:
        """Δn = 0 → Ω_eff must equal original Ω exactly."""
        self._add(omega=0.72)
        state = self.tm.get_effective_state()
        assert state[0]["omega_eff"] == state[0]["omega"]

    def test_oldest_entry_has_largest_delta_n(self) -> None:
        for i in range(5):
            self._add(seed=i)
        state = self.tm.get_effective_state()
        delta_ns = [s["delta_n"] for s in state]
        # Ordered oldest → newest: delta_n decreases (4, 3, 2, 1, 0)
        assert delta_ns == sorted(delta_ns, reverse=True)

    def test_oldest_delta_n_equals_n_minus_one(self) -> None:
        n = 6
        for i in range(n):
            self._add(seed=i)
        state = self.tm.get_effective_state()
        assert state[0]["delta_n"] == n - 1
        assert state[-1]["delta_n"] == 0

    # --- decay correctness ---

    def test_omega_eff_never_exceeds_omega(self) -> None:
        for i in range(5):
            self._add(omega=0.60 + i * 0.05, seed=i)
        for s in self.tm.get_effective_state():
            assert s["omega_eff"] <= s["omega"]

    def test_newest_entry_omega_eff_equals_omega(self) -> None:
        """Most recent message always has Δn = 0 → no decay."""
        for i in range(4):
            self._add(omega=0.65, seed=i)
        state = self.tm.get_effective_state()
        newest = state[-1]
        assert newest["omega_eff"] == newest["omega"]

    def test_omega_eff_decreases_with_age(self) -> None:
        """Messages with identical Ω must have strictly lower Ω_eff as they age.

        The queue is ordered oldest → newest.  Older entries have larger Δn
        and therefore *smaller* Ω_eff.  Iterating the result list from index 0
        (oldest) to -1 (newest) therefore yields an *ascending* sequence of
        Ω_eff values:

            state[0]  →  Δn = N-1  →  smallest Ω_eff  (most decayed)
            state[-1] →  Δn = 0    →  Ω_eff = Ω       (no decay)
        """
        for i in range(5):
            self._add(omega=0.70, seed=i)
        state = self.tm.get_effective_state()
        eff_scores = [s["omega_eff"] for s in state]
        # Ascending: oldest (most decayed) → newest (no decay)
        assert eff_scores == sorted(eff_scores)

    def test_omega_eff_matches_formula_directly(self) -> None:
        """Cross-check one specific entry against the raw formula."""
        lam = _DECAY_CFG.lambda_decay
        eta = _DECAY_CFG.inertia_strength
        omega = 0.85

        for i in range(3):  # 3 insertions → oldest has Δn = 2
            self._add(omega=omega, seed=i)

        state = self.tm.get_effective_state()
        oldest = state[0]
        assert oldest["delta_n"] == 2

        mu = 1.0 - eta * omega
        expected = round(omega * math.exp(-lam * mu * 2), 6)
        assert oldest["omega_eff"] == expected

    # --- INERTIA EFFECT (the key architectural test) ---

    def test_inertia_healthy_beats_critical_after_20_turns(self) -> None:
        """HEALTHY (Ω=0.85) must retain higher Ω_eff than CRITICAL (Ω=0.15)
        when both are 20 turns old (Δn = 20).

        This is the definitive validation of the inertia property:
        because μ_healthy = 0.575 < μ_critical = 0.925, the HEALTHY
        message decays at 62% of the CRITICAL message's rate and must
        therefore outrank it at any Δn > 0.
        """
        # We need exactly Δn = 20 for both entries.
        # Insert them, then add 20 more interactions to age them.
        healthy_report = _make_report(omega=0.85)
        critical_report = _make_report(omega=0.15)

        # Insert HEALTHY first (it will be oldest after 20 more inserts)
        self.tm.add_interaction("HEALTHY anchor", healthy_report, _unit_vector(8, seed=0))
        # Insert CRITICAL immediately after
        self.tm.add_interaction("CRITICAL noise", critical_report, _unit_vector(8, seed=1))

        # Add 20 more filler turns so both original entries have Δn ≥ 20
        for i in range(20):
            self.tm.add_interaction(f"filler {i}", _make_report(omega=0.50), _unit_vector(8, seed=i + 10))

        state = self.tm.get_effective_state()
        # state[0] = HEALTHY, state[1] = CRITICAL (oldest two entries)
        healthy_state = state[0]
        critical_state = state[1]

        assert healthy_state["delta_n"] == 21  # 20 fillers + CRITICAL itself
        assert critical_state["delta_n"] == 20

        assert healthy_state["omega_eff"] > critical_state["omega_eff"], (
            f"Inertia violation: HEALTHY omega_eff={healthy_state['omega_eff']:.4f} "
            f"should exceed CRITICAL omega_eff={critical_state['omega_eff']:.4f}"
        )

    def test_inertia_ratio_is_greater_than_one(self) -> None:
        """HEALTHY / CRITICAL Ω_eff ratio must be > 1 at Δn = 20."""
        healthy_report = _make_report(omega=0.85)
        critical_report = _make_report(omega=0.15)
        self.tm.add_interaction("h", healthy_report, _unit_vector(8, seed=0))
        self.tm.add_interaction("c", critical_report, _unit_vector(8, seed=1))
        for i in range(20):
            self.tm.add_interaction(f"f{i}", _make_report(omega=0.5), _unit_vector(8, seed=i + 5))

        state = self.tm.get_effective_state()
        ratio = state[0]["omega_eff"] / state[1]["omega_eff"]
        assert ratio > 1.0

    # --- dict schema ---

    def test_result_dict_has_all_required_keys(self) -> None:
        self._add()
        required = {
            "interaction_id", "text", "omega", "omega_eff",
            "delta_n", "token_count", "status_effective", "timestamp",
        }
        state = self.tm.get_effective_state()
        assert required.issubset(state[0].keys())

    def test_status_effective_reflects_omega_eff_tier(self) -> None:
        """status_effective must be derived from Ω_eff, not from original Ω."""
        # After heavy decay an originally HEALTHY message may drop to CRITICAL.
        # Craft a scenario where Δn is large enough to drop status tier.
        omega = 0.65  # Starts as HEALTHY (> 0.6)
        self._add(omega=omega, seed=0)

        # Add enough turns to push Ω_eff below 0.6 (into UNSTABLE/CRITICAL)
        # With Ω=0.65, μ = 0.675, eff_λ = 0.023625
        # At Δn=30: Ω_eff = 0.65 * exp(-0.023625 * 30) ≈ 0.318 → CRITICAL
        for i in range(30):
            self.tm.add_interaction(f"f{i}", _make_report(omega=0.5), _unit_vector(8, seed=i + 20))

        state = self.tm.get_effective_state()
        oldest = state[0]
        # The original omega = 0.65 is HEALTHY; after 30 turns the effective
        # score should be substantially lower
        assert oldest["omega_eff"] < 0.65
        # status_effective must match the decayed score
        assert oldest["status_effective"] == classify_survival_status(
            omega=oldest["omega_eff"],
            critical_threshold=self.tm.config.critical_threshold,
            healthy_threshold=self.tm.config.healthy_threshold,
        ).value

    def test_interaction_id_in_state_matches_entry(self) -> None:
        for i in range(3):
            self._add(seed=i)
        state = self.tm.get_effective_state()
        assert [s["interaction_id"] for s in state] == [0, 1, 2]

    def test_status_effective_uses_custom_runtime_thresholds(self) -> None:
        cfg = DecayConfig(
            token_budget=999_999,
            pruning_frequency=999_999,
            critical_threshold=0.3,
            healthy_threshold=0.75,
        )
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG)

        tm.add_interaction(
            "custom-threshold entry",
            _make_report(omega=0.72),
            _unit_vector(8, seed=99),
        )

        state = tm.get_effective_state()

        assert state[0]["omega_eff"] == pytest.approx(0.72)
        assert state[0]["status_effective"] == SurvivalStatus.UNSTABLE.value


# ---------------------------------------------------------------------------
# TemporalMemory — get_total_tokens
# ---------------------------------------------------------------------------

class TestGetTotalTokens:
    """get_total_tokens must return the exact sum of entry token counts."""

    def setup_method(self) -> None:
        self.tm = TemporalMemory(decay_config=_DECAY_CFG, vector_config=_VECTOR_CFG)

    def _add(self, text: str, seed: int = 0) -> MemoryEntry:
        return self.tm.add_interaction(text, _make_report(), _unit_vector(8, seed))

    def test_empty_queue_returns_zero(self) -> None:
        assert self.tm.get_total_tokens() == 0

    def test_single_entry_returns_its_token_count(self) -> None:
        entry = self._add("hello world", seed=0)
        assert self.tm.get_total_tokens() == entry.token_count

    def test_total_equals_manual_sum(self) -> None:
        entries = [
            self._add("First message.", seed=0),
            self._add("Second, slightly longer message.", seed=1),
            self._add("Third.", seed=2),
        ]
        expected = sum(e.token_count for e in entries)
        assert self.tm.get_total_tokens() == expected

    def test_total_increases_with_each_insertion(self) -> None:
        prev = 0
        for i in range(4):
            self._add(f"message number {i}", seed=i)
            current = self.tm.get_total_tokens()
            assert current > prev
            prev = current


# ---------------------------------------------------------------------------
# TemporalMemory — size / is_empty
# ---------------------------------------------------------------------------

class TestSizeAndIsEmpty:
    """Convenience properties must reflect the deque state accurately."""

    def setup_method(self) -> None:
        self.tm = TemporalMemory(vector_config=_VECTOR_CFG)

    def _add(self, seed: int = 0) -> None:
        self.tm.add_interaction("x", _make_report(), _unit_vector(8, seed))

    def test_is_empty_true_initially(self) -> None:
        assert self.tm.is_empty is True

    def test_is_empty_false_after_insertion(self) -> None:
        self._add()
        assert self.tm.is_empty is False

    def test_size_tracks_queue_length(self) -> None:
        for i in range(5):
            assert self.tm.size == i
            self._add(seed=i)
        assert self.tm.size == 5


# ---------------------------------------------------------------------------
# LTMHook / NullLTMHook
# ---------------------------------------------------------------------------

class TestLTMHook:
    """LTMHook Protocol and NullLTMHook concrete implementation."""

    def test_null_ltm_hook_satisfies_protocol(self) -> None:
        """NullLTMHook must be recognised as a LTMHook at runtime."""
        assert isinstance(NullLTMHook(), LTMHook)

    def test_recording_hook_satisfies_protocol(self) -> None:
        """RecordingLTMHook (test double) must also satisfy LTMHook."""
        assert isinstance(RecordingLTMHook(), LTMHook)

    def test_null_ltm_hook_archive_does_not_raise(self) -> None:
        """NullLTMHook.archive must silently discard entries."""
        hook = NullLTMHook()
        entry = MemoryEntry(
            interaction_id=0,
            text="test",
            report=_make_report(),
            vector=_unit_vector(),
            token_count=1,
            timestamp=1_700_000_000.0,
        )
        hook.archive(entry)  # must not raise

    def test_custom_hook_is_called_on_eviction(self) -> None:
        """When TemporalMemory evicts an entry, it must call archive()."""
        hook = RecordingLTMHook()
        # Budget = 1 token: the second insertion will trigger eviction
        cfg = DecayConfig(token_budget=1, pruning_frequency=999_999)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        # First insertion: 1 token, at budget (no eviction)
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, seed=0))
        assert len(hook.archived) == 0
        # Second insertion: 2 tokens > budget, evict first CRITICAL
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, seed=1))
        assert len(hook.archived) == 1

    def test_null_hook_is_default(self) -> None:
        """TemporalMemory must use NullLTMHook when no hook is provided."""
        tm = TemporalMemory()
        assert isinstance(tm._ltm_hook, NullLTMHook)

    def test_ltm_hook_property_returns_injected_hook(self) -> None:
        """ltm_hook property must return the same hook object that was injected."""
        hook = RecordingLTMHook()
        tm = TemporalMemory(ltm_hook=hook)
        assert tm.ltm_hook is hook


# ---------------------------------------------------------------------------
# TemporalMemory — _get_pruning_candidates
# ---------------------------------------------------------------------------

class TestPruningCandidates:
    """_get_pruning_candidates must return the correct bucket-sorted list."""

    def setup_method(self) -> None:
        # No auto-pruning: we test the selection logic in isolation.
        self.tm = TemporalMemory(
            decay_config=_NO_PRUNE_CFG,
            vector_config=_VECTOR_CFG,
        )

    def _add(self, omega: float, seed: int) -> MemoryEntry:
        return self.tm.add_interaction("x", _make_report(omega=omega), _unit_vector(8, seed))

    def test_empty_queue_returns_empty_candidates(self) -> None:
        assert self.tm._get_pruning_candidates() == []

    def test_healthy_entries_excluded_from_candidates(self) -> None:
        self._add(omega=0.85, seed=0)
        self._add(omega=0.75, seed=1)
        assert self.tm._get_pruning_candidates() == []

    def test_all_healthy_queue_returns_empty(self) -> None:
        for i in range(5):
            self._add(omega=0.85, seed=i)
        assert self.tm._get_pruning_candidates() == []

    def test_critical_entries_precede_unstable(self) -> None:
        """CRITICAL bucket must appear before UNSTABLE in candidate list."""
        critical = self._add(omega=0.15, seed=0)
        unstable = self._add(omega=0.45, seed=1)
        candidates = self.tm._get_pruning_candidates()
        # critical must appear before unstable
        crit_pos = candidates.index(critical)
        unst_pos = candidates.index(unstable)
        assert crit_pos < unst_pos

    def test_within_critical_bucket_sorted_oldest_first(self) -> None:
        c_old = self._add(omega=0.15, seed=0)  # ID=0
        c_new = self._add(omega=0.15, seed=1)  # ID=1
        candidates = self.tm._get_pruning_candidates()
        assert candidates[0] is c_old
        assert candidates[1] is c_new

    def test_within_unstable_bucket_sorted_oldest_first(self) -> None:
        u_old = self._add(omega=0.45, seed=0)  # ID=0
        u_new = self._add(omega=0.45, seed=1)  # ID=1
        candidates = self.tm._get_pruning_candidates()
        assert candidates[0] is u_old
        assert candidates[1] is u_new

    def test_full_ordering_critical_then_unstable_oldest_first(self) -> None:
        """Full scenario: HEALTHY excluded, CRITICALs first, UNSTABLEs after."""
        healthy = self._add(omega=0.85, seed=0)  # ID=0, excluded
        c1 = self._add(omega=0.15, seed=1)       # ID=1, CRITICAL older
        c2 = self._add(omega=0.15, seed=2)       # ID=2, CRITICAL newer
        u1 = self._add(omega=0.45, seed=3)       # ID=3, UNSTABLE older
        u2 = self._add(omega=0.45, seed=4)       # ID=4, UNSTABLE newer

        candidates = self.tm._get_pruning_candidates()
        assert len(candidates) == 4
        assert healthy not in candidates
        assert candidates == [c1, c2, u1, u2]

    def test_candidate_list_does_not_modify_queue_order(self) -> None:
        """_get_pruning_candidates must not reorder the live queue."""
        entries = [self._add(omega=o, seed=i)
                   for i, o in enumerate([0.85, 0.15, 0.45, 0.15, 0.85])]
        queue_before = list(self.tm.queue)
        self.tm._get_pruning_candidates()
        queue_after = list(self.tm.queue)
        assert queue_before == queue_after

    def test_operational_bonus_can_reorder_candidates_across_tiers(self) -> None:
        """Pressure pruning may keep a CRITICAL operational memory over plain noise."""
        tm = TemporalMemory(
            decay_config=_NO_PRUNE_CFG,
            pruning_priority_config=PruningPriorityConfig(rho_constraint=0.4),
            vector_config=_VECTOR_CFG,
        )
        critical_constraint = tm.add_interaction(
            "Do not use external APIs.",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_constraint=True),
            ),
            _unit_vector(8, 0),
        )
        unstable_noise = tm.add_interaction(
            "Unrelated note.",
            _make_report(omega=0.45),
            _unit_vector(8, 1),
        )

        candidates = tm._get_pruning_candidates()

        assert candidates[0] is unstable_noise
        assert candidates[1] is critical_constraint

    def test_topic_superseded_past_state_is_ranked_before_newer_current_state(self) -> None:
        tm = TemporalMemory(
            decay_config=_NO_PRUNE_CFG,
            pruning_priority_config=PruningPriorityConfig(superseded_past_penalty=0.35),
            vector_config=_VECTOR_CFG,
        )
        past = tm.add_interaction(
            "x",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_past_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="tea",
            ),
            _unit_vector(8, 0),
        )
        current = tm.add_interaction(
            "y",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_current_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="coffee",
            ),
            _unit_vector(8, 1),
        )

        candidates = tm._get_pruning_candidates()

        assert candidates[0] is past
        assert candidates[1] is current

    def test_pruning_candidates_compute_topic_supersession_once_per_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tm = TemporalMemory(
            decay_config=_NO_PRUNE_CFG,
            pruning_priority_config=PruningPriorityConfig(superseded_past_penalty=0.35),
            vector_config=_VECTOR_CFG,
        )
        tm.add_interaction(
            "I used to prefer tea.",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_past_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="tea",
            ),
            _unit_vector(8, 0),
        )
        tm.add_interaction(
            "I currently prefer coffee.",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_current_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="coffee",
            ),
            _unit_vector(8, 1),
        )
        tm.add_interaction(
            "Another unstable memory.",
            _make_report(omega=0.45),
            _unit_vector(8, 2),
        )

        original = tm._topic_superseded_record_ids_from_entries
        calls = 0

        def counting(entries, stats=None):
            nonlocal calls
            calls += 1
            return original(entries, stats=stats)

        monkeypatch.setattr(tm, "_topic_superseded_record_ids_from_entries", counting)

        tm._get_pruning_candidates()

        assert calls == 1


# ---------------------------------------------------------------------------
# TemporalMemory — prune_to_budget
# ---------------------------------------------------------------------------

class TestPruneToBudget:
    """prune_to_budget must enforce the token budget via bucket-sorted eviction."""

    def _build_tm(
        self,
        budget: int,
        pruning_priority_config: PruningPriorityConfig | None = None,
    ) -> tuple[TemporalMemory, RecordingLTMHook]:
        hook = RecordingLTMHook()
        cfg = DecayConfig(token_budget=budget, pruning_frequency=999_999)
        tm = TemporalMemory(
            decay_config=cfg,
            pruning_priority_config=pruning_priority_config,
            vector_config=_VECTOR_CFG,
            ltm_hook=hook,
        )
        return tm, hook

    def test_no_eviction_when_under_budget(self) -> None:
        tm, hook = self._build_tm(budget=100)
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        assert len(hook.archived) == 0

    def test_returns_empty_list_when_no_eviction_needed(self) -> None:
        tm, _ = self._build_tm(budget=100)
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        assert tm.prune_to_budget() == []

    def test_evicted_entries_removed_from_queue(self) -> None:
        tm, hook = self._build_tm(budget=1)
        e1 = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        # Adding second entry pushes to 2 tokens > 1 → e1 evicted by auto-prune
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        assert e1 not in tm.queue

    def test_evicted_entries_forwarded_to_ltm_hook(self) -> None:
        tm, hook = self._build_tm(budget=1)
        e1 = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        assert e1 in hook.archived

    def test_prune_returns_evicted_entries(self) -> None:
        """prune_to_budget return value must match what was actually evicted."""
        hook = RecordingLTMHook()
        cfg = DecayConfig(token_budget=2, pruning_frequency=999_999)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        for i in range(3):
            tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, i))
        # After 3 insertions with budget=2, 1 should have been evicted auto-
        # Call prune_to_budget explicitly to see if it returns remaining victim
        # At this point total = 2 = budget, so no more evictions needed
        result = tm.prune_to_budget()
        assert result == []  # already at budget

    def test_all_healthy_queue_not_evicted_even_when_over_budget(self) -> None:
        """HEALTHY messages must survive even if total tokens exceed budget.

        This is by design: HEALTHY messages are protected from budget
        pressure.  The deficit is not resolved by evicting them.
        """
        # 5 HEALTHY messages × 1 token, budget = 3
        tm, hook = self._build_tm(budget=3)
        # Auto-prune fires from message 4 onward but finds no candidates
        for i in range(5):
            tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, i))
        # All HEALTHY: nothing archived
        assert len(hook.archived) == 0
        # Queue still has all 5 entries (budget not enforced on HEALTHY)
        assert tm.size == 5

    def test_pruning_bonus_keeps_constraint_memory_longer_under_pressure(self) -> None:
        tm, hook = self._build_tm(
            budget=1,
            pruning_priority_config=PruningPriorityConfig(rho_constraint=0.35),
        )
        constraint = tm.add_interaction(
            "x",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_constraint=True),
            ),
            _unit_vector(8, 0),
        )
        noise = tm.add_interaction(
            "y",
            _make_report(omega=0.15),
            _unit_vector(8, 1),
        )

        assert constraint in tm.queue
        assert noise not in tm.queue
        assert hook.archived == [noise]

    def test_superseded_past_state_is_evicted_before_newer_current_state(self) -> None:
        tm, hook = self._build_tm(
            budget=1,
            pruning_priority_config=PruningPriorityConfig(superseded_past_penalty=0.35),
        )
        past = tm.add_interaction(
            "x",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_past_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="tea",
            ),
            _unit_vector(8, 0),
        )
        current = tm.add_interaction(
            "y",
            _make_report(
                omega=0.15,
                signals=InteractionSignals(is_current_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="coffee",
            ),
            _unit_vector(8, 1),
        )

        assert current in tm.queue
        assert past not in tm.queue
        assert hook.archived == [past]

    # ------------------------------------------------------------------
    # THE SPEC SCENARIO
    # ------------------------------------------------------------------

    def test_spec_scenario_criticals_evicted_before_healthy(self) -> None:
        """SPEC: 2 HEALTHY (oldest) + 5 CRITICAL + 3 UNSTABLE, budget=6.

        Insertion order deliberately places HEALTHY messages FIRST (oldest
        IDs 0,1) to prove they are protected even though they are the
        *oldest* messages in the queue.

        Expected outcome after all 10 insertions:
          - Exactly 4 evictions, all from the CRITICAL bucket.
          - Both HEALTHY entries survive (IDs 0 and 1 remain in queue).
          - 3 UNSTABLE entries survive.
          - The most recent CRITICAL (ID=6) is the only CRITICAL survivor
            because it was always last in the candidate list.
          - LTM hook received exactly 4 entries, all Ω=0.15.
        """
        hook = RecordingLTMHook()
        # budget=6, 1 token per message → triggers eviction from msg 7 onward
        cfg = DecayConfig(token_budget=6, pruning_frequency=999_999)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)

        # Insert 2 HEALTHY (IDs 0, 1) — oldest, highest Ω
        for i in range(2):
            tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, i))

        # Insert 5 CRITICAL (IDs 2–6) — more recent than HEALTHY
        for i in range(5):
            tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, i + 2))

        # Insert 3 UNSTABLE (IDs 7–9)
        for i in range(3):
            tm.add_interaction("x", _make_report(omega=0.45), _unit_vector(8, i + 7))

        # --- assertions ---
        assert len(hook.archived) == 4, (
            f"Expected 4 evictions, got {len(hook.archived)}"
        )
        assert all(e.omega == 0.15 for e in hook.archived), (
            "All evicted entries must be CRITICAL (Ω=0.15)"
        )
        assert tm.size == 6

        surviving_omegas = [e.omega for e in tm.queue]
        # Both HEALTHYs must have survived (even though they are OLDEST)
        assert surviving_omegas.count(0.85) == 2, (
            "Both HEALTHY messages must survive budget pressure"
        )
        # All 3 UNSTABLEs must have survived
        assert surviving_omegas.count(0.45) == 3

        # The surviving CRITICAL is the most recent one (ID=6, Δn=0 at eviction time)
        surviving_criticals = [e for e in tm.queue if e.omega == 0.15]
        assert len(surviving_criticals) == 1
        assert surviving_criticals[0].interaction_id == 6


# ---------------------------------------------------------------------------
# TemporalMemory — periodic_cleanup
# ---------------------------------------------------------------------------

class TestPeriodicCleanup:
    """periodic_cleanup must hard-kill entries below the Ω_eff floor."""

    def _fast_decay_cfg(self, pruning_frequency: int = 999_999) -> DecayConfig:
        """Config with λ=2.0 so entries hit the hard-kill floor in 2–3 turns."""
        return DecayConfig(
            lambda_decay=2.0,
            inertia_strength=0.5,
            hard_kill_threshold=0.05,
            token_budget=999_999,
            pruning_frequency=pruning_frequency,
        )

    def test_cleanup_does_not_evict_entries_above_threshold(self) -> None:
        """Entries with Ω_eff ≥ hard_kill_threshold must be left alone."""
        cfg = self._fast_decay_cfg()
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG)
        # Add one HEALTHY entry — Δn=0, Ω_eff=0.85 >> 0.05
        tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 0))
        evicted = tm.periodic_cleanup()
        assert evicted == []
        assert tm.size == 1

    def test_cleanup_evicts_entries_below_threshold(self) -> None:
        """Entries with Ω_eff < hard_kill_threshold must be evicted.

        With λ=2.0 and Ω=0.15:
          μ = 1 − 0.5*0.15 = 0.925
          Ω_eff(Δn=2) = 0.15 * exp(−2.0*0.925*2) ≈ 0.15*exp(−3.7) ≈ 0.0038
          0.0038 < 0.05 → hard-killed.
        """
        hook = RecordingLTMHook()
        cfg = self._fast_decay_cfg()
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        # Add 3 entries: 2 CRITICAL at the start, 1 HEALTHY at the end
        c0 = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        c1 = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        h2 = tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 2))
        # At this point: c0(Δn=2), c1(Δn=1), h2(Δn=0)
        # c0: Ω_eff = 0.15*exp(-2*0.925*2) ≈ 0.0038 < 0.05 → kill
        # c1: Ω_eff = 0.15*exp(-2*0.925*1) ≈ 0.15*exp(-1.85) ≈ 0.024 < 0.05 → kill
        # h2: Ω_eff = 0.85 > 0.05 → safe
        evicted = tm.periodic_cleanup()
        assert len(evicted) == 2
        assert c0 in evicted
        assert c1 in evicted
        assert h2 not in evicted
        assert tm.size == 1
        assert h2 in tm.queue

    def test_cleanup_also_hard_kills_healthy_if_below_floor(self) -> None:
        """periodic_cleanup must hard-kill HEALTHY messages at the floor.

        Unlike prune_to_budget (which protects HEALTHY), periodic_cleanup
        applies to ALL entries — HEALTHY messages that have organically
        decayed below 0.05 are evicted from active memory.

        With λ=2.0 and Ω=0.85:
          μ = 1 − 0.5*0.85 = 0.575
          Ω_eff(Δn=3) = 0.85 * exp(−2*0.575*3) ≈ 0.85*exp(−3.45) ≈ 0.027
          0.027 < 0.05 → hard-killed.
        """
        hook = RecordingLTMHook()
        cfg = self._fast_decay_cfg()
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        # Insert HEALTHY first (ID=0), then 3 fillers to age it by Δn=3
        h_old = tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 0))
        for i in range(3):
            tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, i + 1))
        # h_old: Δn=3, Ω_eff ≈ 0.027 < 0.05 → must be hard-killed
        evicted = tm.periodic_cleanup()
        assert h_old in evicted

    def test_cleanup_forwards_to_ltm_hook(self) -> None:
        """Hard-killed entries must be archived via LTM hook."""
        hook = RecordingLTMHook()
        cfg = self._fast_decay_cfg()
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        c0 = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 2))
        tm.periodic_cleanup()
        assert c0 in hook.archived

    def test_cleanup_auto_triggered_by_add_interaction(self) -> None:
        """add_interaction must trigger periodic_cleanup every pruning_frequency turns."""
        hook = RecordingLTMHook()
        # pruning_frequency=3: cleanup fires after 3rd, 6th, ... insertion
        cfg = self._fast_decay_cfg(pruning_frequency=3)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        # Insert 2 CRITICALs at the start, then 1 HEALTHY to trigger cleanup
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        # 3rd insertion → turn_counter=3, 3%3=0 → cleanup fires
        tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 2))
        # Both CRITICALs (Δn=2 and Δn=1) have Ω_eff < 0.05 → evicted
        assert len(hook.archived) == 2

    def test_cleanup_returns_evicted_entries(self) -> None:
        cfg = self._fast_decay_cfg()
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG)
        c = tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        tm.add_interaction("x", _make_report(omega=0.85), _unit_vector(8, 2))
        result = tm.periodic_cleanup()
        assert c in result


# ---------------------------------------------------------------------------
# InteractionMatrix sync on eviction
# ---------------------------------------------------------------------------

class TestMatrixSync:
    """Eviction must remove the vector from InteractionMatrix when in window."""

    def test_evicted_vector_removed_from_matrix(self) -> None:
        """After eviction, matrix.size must decrease if vector was in window."""
        hook = RecordingLTMHook()
        # Small window (5) and budget forcing eviction from within window
        cfg = DecayConfig(token_budget=1, pruning_frequency=999_999)
        vcfg = VectorConfig(window_size=10, vector_dim=8)
        tm = TemporalMemory(decay_config=cfg, vector_config=vcfg, ltm_hook=hook)
        # Insert first CRITICAL — in matrix window, then add second to trigger eviction
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        size_after_first = tm.matrix.size  # should be 1
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 1))
        # First entry evicted; matrix should have lost its vector
        assert tm.matrix.size < size_after_first + 1

    def test_matrix_centroid_valid_after_eviction(self) -> None:
        """After eviction matrix centroid must still be computable (not None)."""
        cfg = DecayConfig(token_budget=1, pruning_frequency=999_999)
        vcfg = VectorConfig(window_size=10, vector_dim=8)
        tm = TemporalMemory(decay_config=cfg, vector_config=vcfg)
        tm.add_interaction("x", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.45), _unit_vector(8, 1))
        # First evicted; second remains — centroid should be the second's vector
        assert tm.matrix.get_centroid() is not None


# ===========================================================================
# Phase 4 — TemporalMemory: get_full_context
# ===========================================================================

class TestGetFullContext:
    """get_full_context: structure and content of the assembled context string."""

    def _build_tm(self, *, ltm_hook: LTMHook | None = None, nlp_engine: Any | None = None) -> TemporalMemory:
        cfg = DecayConfig(token_budget=999_999, pruning_frequency=999_999)
        return TemporalMemory(
            decay_config=cfg,
            vector_config=_VECTOR_CFG,
            ltm_hook=ltm_hook,
            nlp_engine=nlp_engine,
        )

    # ------------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------------

    def test_empty_state_contains_active_header(self) -> None:
        """Even with an empty queue, the active header must be present."""
        tm = self._build_tm()
        ctx = tm.get_full_context()
        assert "=== ACTIVE CONVERSATION ===" in ctx

    def test_empty_state_no_legacy_recall_header(self) -> None:
        """The legacy recall header must never appear in the assembled context."""
        tm = self._build_tm()
        ctx = tm.get_full_context()
        assert "=== SYSTEM MEMORY (FOSSILIZED) ===" not in ctx

    def test_empty_queue_no_entry_text(self) -> None:
        """Empty queue → no message text below the active header."""
        tm = self._build_tm()
        ctx = tm.get_full_context()
        lines = ctx.splitlines()
        active_idx = next(i for i, l in enumerate(lines) if "ACTIVE CONVERSATION" in l)
        assert lines[active_idx + 1:] == []

    # ------------------------------------------------------------------
    # Active section
    # ------------------------------------------------------------------

    def test_active_entries_appear_in_order(self) -> None:
        """Entry texts must appear in queue order (oldest → newest)."""
        tm = self._build_tm()
        texts = ["First message.", "Second message.", "Third message."]
        for text in texts:
            tm.add_interaction(text, _make_report(omega=0.85), _unit_vector(8, 0))

        ctx = tm.get_full_context()
        lines = ctx.splitlines()
        active_idx = next(i for i, l in enumerate(lines) if "ACTIVE CONVERSATION" in l)
        active_body = lines[active_idx + 1:]
        assert active_body[0].startswith("[A1] ")
        assert active_body[1] == "First message."
        assert active_body[3].startswith("[A2] ")
        assert active_body[4] == "Second message."
        assert active_body[6].startswith("[A3] ")
        assert active_body[7] == "Third message."

    def test_active_section_contains_all_entry_texts(self) -> None:
        """Every entry's text must appear somewhere after the active header."""
        tm = self._build_tm()
        tm.add_interaction("Alpha", _make_report(omega=0.85), _unit_vector(8, 0))
        tm.add_interaction("Beta", _make_report(omega=0.85), _unit_vector(8, 1))
        ctx = tm.get_full_context()
        assert "Alpha" in ctx
        assert "Beta" in ctx

    def test_corrected_active_entry_is_hidden_from_context(self) -> None:
        tm = self._build_tm()
        corrected = _make_report(omega=0.85)
        corrected.provenance.corrected_by_user = True

        tm.add_interaction("Outdated answer", corrected, _unit_vector(8, 0))
        tm.add_interaction("Corrected answer", _make_report(omega=0.85), _unit_vector(8, 1))

        ctx = tm.get_full_context()

        assert "Outdated answer" not in ctx
        assert "Corrected answer" in ctx
        assert tm.get_context_metrics()["correction_miss_count"] == 1

    def test_recalled_conflict_prefers_user_sourced_raw_memory(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:3",
                            interaction_id=3,
                            role="assistant",
                            text="Oracle said the wrong CTO.",
                            created_at=100.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=3),
                        ),
                        distance=0.2,
                        similarity_score=0.8,
                        source="test",
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:4",
                            interaction_id=4,
                            role="user",
                            text="User corrected the CTO.",
                            created_at=101.0,
                            provenance=InteractionProvenance(role="user", source_turn=4),
                        ),
                        distance=0.1,
                        similarity_score=0.9,
                        source="test",
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "Oracle said the wrong CTO.": _make_report(
                        signals=InteractionSignals(is_preference=True),
                        topic_identity="fact|cto",
                        topic_value="wrong_cto",
                    ),
                    "User corrected the CTO.": _make_report(
                        signals=InteractionSignals(is_preference=True),
                        topic_identity="fact|cto",
                        topic_value="right_cto",
                    ),
                }
            ),
        )

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))

        assert "[R1] user | time=101" in ctx
        assert "User corrected the CTO." in ctx
        assert "Oracle said the wrong CTO." not in ctx

    def test_active_section_uses_compact_numbered_metadata_lines(self) -> None:
        tm = self._build_tm()
        report = _make_report(
            omega=0.85,
            signals=InteractionSignals(is_current_state=True),
        )
        report.provenance.role = "user"
        report.provenance.source_turn = 7
        tm.add_interaction("I now live in Rome.", report, _unit_vector(8, 0))

        ctx = tm.get_full_context()

        assert "prefer active unless the query is historical." not in ctx
        assert "[A1] user | turn=7 | current" in ctx
        assert "I now live in Rome." in ctx

    def test_active_section_shows_short_precedence_note_only_when_recalled_exists(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:8",
                            interaction_id=8,
                            role="user",
                            text="Alice moved to Rome.",
                            created_at=108.0,
                            provenance=InteractionProvenance(role="user", source_turn=8),
                        ),
                        similarity_score=0.9,
                        source="test",
                    )
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "Alice moved to Rome.": _make_report(
                        signals=InteractionSignals(is_current_state=True),
                        topic_identity="state|location",
                        topic_value="rome",
                    )
                }
            ),
        )
        report = _make_report(omega=0.85)
        report.provenance.role = "assistant"
        tm.add_interaction("Current conversation line.", report, _unit_vector(8, 0))

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))

        assert "prefer active unless the query is historical." in ctx
        assert "[R1] user | time=108 | current" in ctx
        assert "[A1] assistant | turn=0" in ctx

    def test_noncanonical_hook_output_raises_type_error(self) -> None:
        class LegacyHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[Any]:  # noqa: ARG002
                return [
                    RawLTMRecord(
                        record_id="record:4",
                        interaction_id=4,
                        role="user",
                        text="Remember Paris.",
                        created_at=101.0,
                        provenance=InteractionProvenance(role="user", source_turn=4),
                    )
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=LegacyHook(),  # type: ignore[arg-type]
        )

        with pytest.raises(TypeError, match=r"LTMHook\.search_raw\(\) must return list\[RawRecallHit\]"):
            tm.get_full_context(query_vector=_unit_vector(8, 42))

    def test_non_list_hook_output_raises_type_error(self) -> None:
        class TupleHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> tuple[RawRecallHit, ...]:  # noqa: ARG002
                return (
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:4",
                            interaction_id=4,
                            role="user",
                            text="Remember Paris.",
                            created_at=101.0,
                            provenance=InteractionProvenance(role="user", source_turn=4),
                        ),
                        distance=0.1,
                        similarity_score=0.9,
                    ),
                )

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=TupleHook(),  # type: ignore[arg-type]
        )

        with pytest.raises(TypeError, match=r"got non-list result of type: tuple"):
            tm.get_full_context(query_vector=_unit_vector(8, 42))

    def test_context_metrics_count_model_derived_and_supported_entries(self) -> None:
        tm = self._build_tm()
        model_report = _make_report(omega=0.85)
        model_report.provenance.derived_from_model = True
        user_report = _make_report(omega=0.85)
        user_report.provenance.role = "user"

        tm.add_interaction("Model memory", model_report, _unit_vector(8, 0))
        tm.add_interaction("User memory", user_report, _unit_vector(8, 1))

        tm.get_full_context()
        metrics = tm.get_context_metrics()

        assert metrics["model_derived_memory_count"] == 1
        assert metrics["context_supported_memory_count"] == 1

    def test_active_context_suppresses_entries_explicitly_superseded_by_lineage(self) -> None:
        tm = self._build_tm()
        older = tm.add_interaction("Old plan for Paris", _make_report(omega=0.85), _unit_vector(8, 0))
        newer = tm.add_interaction("Updated plan for Paris", _make_report(omega=0.85), _unit_vector(8, 1))
        newer.lineage.supersedes.append(older.record_id)

        ctx = tm.get_full_context()

        assert "Updated plan for Paris" in ctx
        assert "Old plan for Paris" not in ctx
        assert tm.get_context_metrics()["superseded_memory_count"] == 1

    def test_active_context_suppresses_historical_state_superseded_by_current_topic_value(self) -> None:
        tm = self._build_tm()
        tm.add_interaction(
            "I used to prefer tea.",
            _make_report(
                omega=0.85,
                signals=InteractionSignals(is_past_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="tea",
            ),
            _unit_vector(8, 0),
        )
        tm.add_interaction(
            "I currently prefer coffee.",
            _make_report(
                omega=0.85,
                signals=InteractionSignals(is_current_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="coffee",
            ),
            _unit_vector(8, 1),
        )

        ctx = tm.get_full_context()

        assert "I currently prefer coffee." in ctx
        assert "I used to prefer tea." not in ctx
        assert tm.get_context_metrics()["superseded_memory_count"] == 1

    def test_recalled_raw_memory_that_updates_a_topic_prevails_even_if_less_similar(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:3",
                            interaction_id=3,
                            role="assistant",
                            text="I used to prefer tea.",
                            created_at=103.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=3),
                        ),
                        similarity_score=0.95,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:4",
                            interaction_id=4,
                            role="user",
                            text="I currently prefer coffee.",
                            created_at=104.0,
                            provenance=InteractionProvenance(role="user", source_turn=4),
                        ),
                        similarity_score=0.80,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "I used to prefer tea.": _make_report(
                        signals=InteractionSignals(is_past_state=True, is_preference=True),
                        topic_identity="preference|prefer",
                        topic_value="tea",
                    ),
                    "I currently prefer coffee.": _make_report(
                        signals=InteractionSignals(is_current_state=True, is_preference=True),
                        topic_identity="preference|prefer",
                        topic_value="coffee",
                    ),
                }
            ),
        )

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))

        assert "I currently prefer coffee." in ctx
        assert "I used to prefer tea." not in ctx
        assert tm.get_context_metrics()["superseded_memory_count"] == 1

    def test_recalled_raw_hit_is_suppressed_by_active_lineage_before_nlp(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:0",
                            interaction_id=0,
                            role="user",
                            text="I live in Paris.",
                            created_at=100.0,
                            provenance=InteractionProvenance(role="user", source_turn=0),
                        ),
                        similarity_score=0.95,
                        rank_hint=0,
                    ),
                ]

        fake_nlp = _FakeNLPEngine()
        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=fake_nlp,
        )
        stale = tm.add_interaction(
            "I live in Paris.",
            _make_report(
                omega=0.85,
                signals=InteractionSignals(is_current_state=True),
                topic_identity="location|live",
                topic_value="paris",
            ),
            _unit_vector(8, 0),
        )
        latest = tm.add_interaction(
            "I live in Rome.",
            _make_report(
                omega=0.85,
                signals=InteractionSignals(is_current_state=True),
                topic_identity="location|live",
                topic_value="rome",
            ),
            _unit_vector(8, 1),
        )
        latest.lineage.corrects.append(stale.record_id)

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))
        diagnostics = tm.get_recall_diagnostics()

        assert "I live in Rome." in ctx
        assert "I live in Paris." not in ctx
        assert fake_nlp.seen_texts == []
        assert any(
            item["reason"] == "active_lineage_suppression"
            and item.get("record_id") == stale.record_id
            for item in diagnostics["suppressed"]
        )

    def test_recalled_topic_is_suppressed_when_newer_active_winner_exists(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:ltm-tea",
                            interaction_id=-1,
                            role="assistant",
                            text="I used to prefer tea.",
                            created_at=99.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=-1),
                        ),
                        similarity_score=0.95,
                        rank_hint=0,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "I used to prefer tea.": _make_report(
                        signals=InteractionSignals(is_past_state=True, is_preference=True),
                        topic_identity="preference|prefer",
                        topic_value="tea",
                    ),
                }
            ),
        )
        tm.add_interaction(
            "I currently prefer coffee.",
            _make_report(
                omega=0.85,
                signals=InteractionSignals(is_current_state=True, is_preference=True),
                topic_identity="preference|prefer",
                topic_value="coffee",
            ),
            _unit_vector(8, 0),
        )

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))
        diagnostics = tm.get_recall_diagnostics()

        assert "I currently prefer coffee." in ctx
        assert "I used to prefer tea." not in ctx
        assert any(
            item["reason"] == "active_topic_supersession"
            and item.get("topic_identity") == "preference|prefer"
            for item in diagnostics["suppressed"]
        )

    def test_query_like_recalled_candidates_are_filtered_before_ranking(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:10",
                            interaction_id=10,
                            role="user",
                            text="What hard implementation constraint did I set?",
                            created_at=110.0,
                            provenance=InteractionProvenance(role="user", source_turn=10),
                        ),
                        similarity_score=0.95,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:11",
                            interaction_id=11,
                            role="assistant",
                            text="Do not use external APIs in any proposed solution.",
                            created_at=111.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=11),
                        ),
                        similarity_score=0.85,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "What hard implementation constraint did I set?": _make_report(
                        is_query_like=True,
                    ),
                    "Do not use external APIs in any proposed solution.": _make_report(
                        signals=InteractionSignals(is_constraint=True, has_negation=True),
                        topic_identity="constraint|use",
                        topic_value="external_api",
                    ),
                }
            ),
        )

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))
        diagnostics = tm.get_recall_diagnostics()

        assert "What hard implementation constraint did I set?" not in ctx
        assert "Do not use external APIs in any proposed solution." in ctx
        assert len(diagnostics["raw_candidates"]) == 2
        assert len(diagnostics["ranked_candidates"]) == 1
        assert diagnostics["suppressed"][0]["reason"] == "query_like_filter"

    def test_query_like_personal_fact_recall_survives_filtering(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:12",
                            interaction_id=12,
                            role="user",
                            text=(
                                "I graduated with a degree in Business Administration. "
                                "Do you have any advice on staying organized?"
                            ),
                            created_at=112.0,
                            provenance=InteractionProvenance(role="user", source_turn=12),
                        ),
                        similarity_score=0.93,
                        rank_hint=0,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    (
                        "I graduated with a degree in Business Administration. "
                        "Do you have any advice on staying organized?"
                    ): _make_report(
                        is_query_like=True,
                        signals=InteractionSignals(personal_relevance=1.0),
                    ),
                }
            ),
        )

        ctx = tm.get_full_context(query_vector=_unit_vector(8, 42))
        diagnostics = tm.get_recall_diagnostics()

        assert "I graduated with a degree in Business Administration." in ctx
        assert len(diagnostics["ranked_candidates"]) == 1
        assert not any(item["reason"] == "query_like_filter" for item in diagnostics["suppressed"])

    def test_personal_fact_bonus_can_rank_user_fact_above_similar_assistant_text(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:13",
                            interaction_id=13,
                            role="assistant",
                            text="The game was released in 2020 and received strong reviews.",
                            created_at=113.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=13),
                        ),
                        similarity_score=0.90,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:14",
                            interaction_id=14,
                            role="user",
                            text="I spent 70 hours playing this game.",
                            created_at=114.0,
                            provenance=InteractionProvenance(role="user", source_turn=14),
                        ),
                        similarity_score=0.82,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "The game was released in 2020 and received strong reviews.": _make_report(),
                    "I spent 70 hours playing this game.": _make_report(
                        signals=InteractionSignals(personal_relevance=1.0),
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )

        assert ranked[0].record.record_id == "record:14"

    def test_quantitative_bonus_can_rank_numeric_fact_above_non_numeric_text(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:15",
                            interaction_id=15,
                            role="user",
                            text="I enjoyed playing this game.",
                            created_at=115.0,
                            provenance=InteractionProvenance(role="user", source_turn=15),
                        ),
                        similarity_score=0.80,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:16",
                            interaction_id=16,
                            role="user",
                            text="I spent 70 hours playing this game.",
                            created_at=116.0,
                            provenance=InteractionProvenance(role="user", source_turn=16),
                        ),
                        similarity_score=0.76,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "I enjoyed playing this game.": _make_report(),
                    "I spent 70 hours playing this game.": _make_report(
                        signals=InteractionSignals(quantitative_relevance=1.0),
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )

        assert ranked[0].record.record_id == "record:16"

    def test_ack_like_assistant_recall_is_ranked_below_operational_user_memory(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:20",
                            interaction_id=20,
                            role="assistant",
                            text="Understood. I will avoid external APIs in the proposed solution.",
                            created_at=120.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=20),
                        ),
                        similarity_score=0.90,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:21",
                            interaction_id=21,
                            role="user",
                            text="Do not use external APIs in any proposed solution.",
                            created_at=121.0,
                            provenance=InteractionProvenance(role="user", source_turn=21),
                        ),
                        similarity_score=0.82,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "Understood. I will avoid external APIs in the proposed solution.": _make_report(
                        is_ack_like=True,
                        info_density=0.30,
                    ),
                    "Do not use external APIs in any proposed solution.": _make_report(
                        signals=InteractionSignals(is_constraint=True, has_negation=True),
                        topic_identity="constraint|use",
                        topic_value="external_api",
                        info_density=0.40,
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )

        assert ranked[0].record.record_id == "record:21"

    def test_same_topic_value_recalled_candidates_are_deduped(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:30",
                            interaction_id=30,
                            role="assistant",
                            text="Understood. I will avoid external APIs in the proposed solution.",
                            created_at=130.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=30),
                        ),
                        similarity_score=0.90,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:31",
                            interaction_id=31,
                            role="assistant",
                            text="You set a hard implementation constraint stating that no external APIs should be used in any proposed solution.",
                            created_at=131.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=31),
                        ),
                        similarity_score=0.88,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "Understood. I will avoid external APIs in the proposed solution.": _make_report(
                        is_ack_like=True,
                        topic_identity="constraint|use",
                        topic_value="external_api",
                    ),
                    "You set a hard implementation constraint stating that no external APIs should be used in any proposed solution.": _make_report(
                        is_ack_like=True,
                        topic_identity="constraint|use",
                        topic_value="external_api",
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )
        diagnostics = tm.get_recall_diagnostics()

        assert len(ranked) == 1
        assert diagnostics["suppressed"][0]["reason"] == "same_topic_value_dedupe"

    def test_topicless_signal_less_recall_is_filtered_when_memory_bearing_candidate_exists(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:40",
                            interaction_id=40,
                            role="user",
                            text="Long distraction six. Discuss urban planning tradeoffs across zoning density and retrofit sequencing.",
                            created_at=140.0,
                            provenance=InteractionProvenance(role="user", source_turn=40),
                        ),
                        similarity_score=0.92,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:41",
                            interaction_id=41,
                            role="assistant",
                            text="You set a hard implementation constraint to avoid using external APIs in all proposed solutions.",
                            created_at=141.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=41),
                        ),
                        similarity_score=0.80,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "Long distraction six. Discuss urban planning tradeoffs across zoning density and retrofit sequencing.": _make_report(
                        info_density=0.85,
                    ),
                    "You set a hard implementation constraint to avoid using external APIs in all proposed solutions.": _make_report(
                        is_ack_like=True,
                        topic_identity="constraint|use",
                        topic_value="external_api",
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )
        diagnostics = tm.get_recall_diagnostics()

        assert len(ranked) == 1
        assert ranked[0].record.record_id == "record:41"
        assert any(item["reason"] == "non_memory_bearing_filter" for item in diagnostics["suppressed"])

    def test_topicless_personal_fact_recall_is_not_filtered_as_non_memory_bearing(self) -> None:
        class RecallingHook:
            def archive(self, entry: MemoryEntry) -> None:  # noqa: ARG002
                return None

            def search_raw(self, query_vector: list[float], k: int = 5) -> list[RawRecallHit]:  # noqa: ARG002
                return [
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:42",
                            interaction_id=42,
                            role="user",
                            text="I spent 70 hours playing this game.",
                            created_at=142.0,
                            provenance=InteractionProvenance(role="user", source_turn=42),
                        ),
                        similarity_score=0.92,
                        rank_hint=0,
                    ),
                    RawRecallHit(
                        record=RawLTMRecord(
                            record_id="record:43",
                            interaction_id=43,
                            role="assistant",
                            text="You set a hard implementation constraint to avoid using external APIs in all proposed solutions.",
                            created_at=143.0,
                            provenance=InteractionProvenance(role="assistant", source_turn=43),
                        ),
                        similarity_score=0.80,
                        rank_hint=1,
                    ),
                ]

        tm = TemporalMemory(
            decay_config=DecayConfig(token_budget=999_999, pruning_frequency=999_999),
            vector_config=_VECTOR_CFG,
            ltm_hook=RecallingHook(),  # type: ignore[arg-type]
            nlp_engine=_MappedRecallNLPEngine(
                {
                    "I spent 70 hours playing this game.": _make_report(
                        info_density=0.45,
                        signals=InteractionSignals(personal_relevance=1.0),
                    ),
                    "You set a hard implementation constraint to avoid using external APIs in all proposed solutions.": _make_report(
                        is_ack_like=True,
                        topic_identity="constraint|use",
                        topic_value="external_api",
                    ),
                }
            ),
        )

        ranked = tm.rerank_contextualized_recall_candidates(
            tm.contextualize_raw_recall_hits(tm.get_raw_recall_hits(_unit_vector(8, 42)))
        )
        diagnostics = tm.get_recall_diagnostics()

        assert {candidate.record.record_id for candidate in ranked} == {"record:42", "record:43"}
        assert not any(
            item["reason"] == "non_memory_bearing_filter"
            and item.get("candidate", {}).get("record", {}).get("record_id") == "record:42"
            for item in diagnostics["suppressed"]
        )

    def test_active_context_counts_invalidated_and_corrected_lineage_targets(self) -> None:
        tm = self._build_tm()
        invalidated = tm.add_interaction("Deprecated note", _make_report(omega=0.85), _unit_vector(8, 0))
        corrected = tm.add_interaction("Wrong answer", _make_report(omega=0.85), _unit_vector(8, 1))
        latest = tm.add_interaction("Replacement answer", _make_report(omega=0.85), _unit_vector(8, 2))
        latest.lineage.invalidates.append(invalidated.record_id)
        latest.lineage.corrects.append(corrected.record_id)

        ctx = tm.get_full_context()
        metrics = tm.get_context_metrics()

        assert "Replacement answer" in ctx
        assert "Deprecated note" not in ctx
        assert "Wrong answer" not in ctx
        assert metrics["invalidated_memory_count"] == 1
        assert metrics["suppressed_conflict_count"] == 1
