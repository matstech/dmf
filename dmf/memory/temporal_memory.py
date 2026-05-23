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

"""Active-memory lifecycle manager for DMF runtime state.

``TemporalMemory`` is the lifecycle manager for the active context window.
It owns the working memory deque, keeps the geometry matrix in sync, and
enforces both budget-based and time-based eviction policies.

Responsibilities
----------------
1. **Insertion**: Accept a scored interaction (text + AnalysisReport +
   embedding vector), tokenise it, stamp it with a monotonic ID, push
   it into the deque and the InteractionMatrix, then trigger cleanup.
2. **Decay view**: Compute Ω_eff for every retained entry.
3. **Budget enforcement** (``prune_to_budget``): Evict lowest-priority
   entries until total tokens ≤ ``DecayConfig.token_budget``.
4. **Periodic hard-kill** (``periodic_cleanup``): Every
   ``DecayConfig.pruning_frequency`` turns, sweep the entire queue and
   evict any entry — including HEALTHY — whose Ω_eff has fallen below
   ``DecayConfig.hard_kill_threshold``.
5. **Archival**: Before removal completes, the entry is handed to the
   configured ``LTMHook`` so the long-term-memory backend can archive the
   raw record.

Pruning priority (``_get_pruning_candidates``)
-----------------------------------------------
Only CRITICAL and UNSTABLE messages are pressure candidates — HEALTHY
messages are never evicted by budget pressure alone. Candidate ordering
adds a pruning-only operational retention bonus:

    pruning_bonus = (
        ρ_constraint   · is_constraint +
        ρ_preference   · is_preference +
        ρ_current_state· is_current_state +
        ρ_correction   · is_correction +
        ρ_replacement  · has_replacement
    )

    effective_pruning_score = Ω_eff + pruning_bonus

Lower ``effective_pruning_score`` values are evicted first; ties break
oldest-first by ``interaction_id``. This lets operationally important
memories survive pressure pruning without changing their decay tier.

HEALTHY messages that organically decay below ``hard_kill_threshold``
are caught by ``periodic_cleanup``.

InteractionMatrix sync
----------------------
When an entry is evicted, ``remove_vector`` is called on the matrix.  If
the vector has already been scrolled out of the sliding window (because
more than ``window_size`` later interactions arrived), the call is a
harmless no-op.  The matrix tracks *conversation trajectory*; vectors
that have left the window no longer influence the centroid regardless.

LTM archival
------------
Every evicted entry is passed to an injected ``LTMHook.archive()``
implementation.  The default is ``NullLTMHook`` (silent discard).
Concrete raw-LTM hooks such as file and Chroma backends can be injected or
resolved from configuration.

Performance
-----------
- ``add_interaction``: O(1) append + O(W) matrix centroid update.
- ``_get_pruning_candidates``: O(N) decay calculation + O(N log N) sort.
- ``prune_to_budget``: O(K) iterations where K = number of evictions.
- ``periodic_cleanup``: O(N) decay scan + O(K) evictions.
- ``get_total_tokens``: O(N) integer sum.
- ``get_full_context``: O(R + N) string join (R = recalled raw candidate count).
All per-entry decay calculations are sub-microsecond (one ``math.exp``).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import tiktoken

from dmf.analysis.decay import calculate_effective_score
from dmf.analysis.geometry import InteractionMatrix
from dmf.memory.constants import (
    CONTEXT_TIME_PREFIX,
    CONTEXT_TURN_PREFIX,
    UNIX_TIMESTAMP_MAX,
    UNIX_TIMESTAMP_MIN,
    UTC_RENDER_FORMAT,
)
from dmf.models.analysis import AnalysisReport, MemoryLineage
from dmf.models.ltm_hook import LTMHook, NullLTMHook
from dmf.models.memory import MemoryEntry
from dmf.models.raw_ltm import ContextualizedRecallCandidate, RawLTMRecord, RawRecallHit
from dmf.models.status import classify_survival_status
from dmf.utils.config import DecayConfig, PruningPriorityConfig, VectorConfig
from dmf.utils.constants import LTM_BACKEND_CHROMA, LTM_BACKEND_FILE, LTM_BACKEND_NULL

if TYPE_CHECKING:
    # DMFConfig is only needed for the from_dmf_config factory; importing it
    # at runtime would create a circular dependency risk if config_loader
    # ever imports from the runtime facade.  TYPE_CHECKING keeps the reference
    # annotation-only at runtime.
    from dmf.analysis.nlp_engine import NLPEngine
    from dmf.utils.config_loader import DMFConfig



# Instantiated once at import time — tiktoken BPE vocabulary is loaded once
# and reused across all TemporalMemory instances.  Thread-safe for reads.
_TOKENIZER: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")


# Context section headers (UI strings — not business logic, not configurable)

_RECALL_HEADER: str = "=== LONG-TERM MEMORY (RECALLED) ==="
_ACTIVE_HEADER: str = "=== ACTIVE CONVERSATION ==="
_ACTIVE_PRECEDENCE_NOTE: str = (
    "If active and recalled conflict, prefer active unless the query is historical."
)

# Thresholds live in configuration objects; TemporalMemory no longer keeps
# any legacy summary-specific module constants.




@dataclass(frozen=True)
class _ActiveContextGuard:
    """Resolved active-context state reused across recall filtering steps."""

    active_entries: list[MemoryEntry]
    active_record_ids: set[str]
    suppressed_record_ids: set[str]
    topic_winners: dict[str, MemoryEntry]


def _render_context_item_block(
    *,
    label: str,
    metadata: list[str],
    text: str,
) -> list[str]:
    header = f"[{label}]"
    if metadata:
        header = f"{header} {' | '.join(metadata)}"
    lines = [header]
    lines.extend(text.strip().splitlines())
    return lines


def _append_context_item(lines: list[str], block: list[str]) -> None:
    if not block:
        return
    if lines:
        lines.append("")
    lines.extend(block)


def _context_metadata_parts(
    *,
    role: object = None,
    time_value: object = None,
    turn_value: object = None,
    state: str | None = None,
    prefer_time_over_turn: bool = False,
) -> list[str]:
    metadata: list[str] = []
    role_text = _string_or_none(role)
    if role_text:
        metadata.append(role_text)
    locator = _context_locator_part(
        time_value=time_value,
        turn_value=turn_value,
        prefer_time_over_turn=prefer_time_over_turn,
    )
    if locator is not None:
        metadata.append(locator)
    if state:
        metadata.append(state)
    return metadata


def _context_locator_part(
    *,
    time_value: object = None,
    turn_value: object = None,
    prefer_time_over_turn: bool = False,
) -> str | None:
    formatted_time = _format_context_time(time_value)
    if prefer_time_over_turn and formatted_time is not None:
        return f"{CONTEXT_TIME_PREFIX}{formatted_time}"
    formatted_turn = _format_context_turn(turn_value)
    if formatted_turn is not None:
        return f"{CONTEXT_TURN_PREFIX}{formatted_turn}"
    if formatted_time is not None:
        return f"{CONTEXT_TIME_PREFIX}{formatted_time}"
    return None


def _format_context_time(value: object) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if UNIX_TIMESTAMP_MIN <= numeric <= UNIX_TIMESTAMP_MAX:
            return datetime.fromtimestamp(numeric, UTC).strftime(UTC_RENDER_FORMAT)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.3f}".rstrip("0").rstrip(".")
    return _string_or_none(value)


def _format_context_turn(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return _string_or_none(value)


def _state_from_report(report: AnalysisReport) -> str | None:
    if report.signals.is_current_state:
        return "current"
    if report.signals.is_past_state:
        return "past"
    return None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class TemporalMemory:
    """Active context window manager with temporal decay and eviction.
    
        Parameters
        ----------
        decay_config : DecayConfig | None
            Immutable decay/budget parameters.  Defaults to ``DecayConfig()``.
        vector_config : VectorConfig | None
            Immutable geometry parameters.  Defaults to ``VectorConfig()``.
        ltm_hook : LTMHook | None
            Archive sink for evicted entries.  Defaults to ``NullLTMHook()``
            (silent discard).  Pass a recording or concrete LTM hook for
            testing or production use.
    
        Attributes
        ----------
        queue : deque[MemoryEntry]
            Working-memory entries ordered oldest → newest.  Length is
            unbounded; ``token_budget`` is enforced by ``prune_to_budget``.
        matrix : InteractionMatrix
            Sliding-window centroid tracker.  Receives the same vectors as
            the queue so divergence calculations reflect recent context.
        config : DecayConfig
            Immutable decay and budget configuration.
        No local summary is maintained. Historical context is recovered only
        through raw-LTM recall.
    
    Args:
        decay_config: See the function signature and surrounding type hints.
        pruning_priority_config: See the function signature and surrounding type hints.
        vector_config: See the function signature and surrounding type hints.
        ltm_hook: See the function signature and surrounding type hints.
        nlp_engine: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        TypeError: Raised by class validation or initialization.
        ValueError: Raised by class validation or initialization.
    """

    def __init__(
        self,
        decay_config: DecayConfig | None = None,
        pruning_priority_config: PruningPriorityConfig | None = None,
        vector_config: VectorConfig | None = None,
        ltm_hook: LTMHook | None = None,
        nlp_engine: NLPEngine | None = None,
    ) -> None:
        self.config: DecayConfig = decay_config or DecayConfig()
        self._pruning_priority_config: PruningPriorityConfig = (
            pruning_priority_config or PruningPriorityConfig()
        )
        self._vector_config: VectorConfig = vector_config or VectorConfig()
        self._ltm_hook: LTMHook = ltm_hook or NullLTMHook()
        self._nlp_engine = nlp_engine

        self.queue: deque[MemoryEntry] = deque()

        self.matrix: InteractionMatrix = InteractionMatrix(self._vector_config)

        self._next_id: int = 0

        self._turn_counter: int = 0

        self._context_metrics: dict[str, int] = {
            "correction_miss_count": 0,
            "conflict_override_count": 0,
            "model_derived_memory_count": 0,
            "context_supported_memory_count": 0,
            "invalidated_memory_count": 0,
            "suppressed_conflict_count": 0,
            "superseded_memory_count": 0,
        }
        self._recall_diagnostics: dict[str, Any] = self._empty_recall_diagnostics()

        self._tokenizer: tiktoken.Encoding = _TOKENIZER


    @classmethod
    def from_dmf_config(
        cls,
        config: DMFConfig,
        ltm_hook: LTMHook | None = None,
        nlp_engine: NLPEngine | None = None,
    ) -> TemporalMemory:
        """Construct a ``TemporalMemory`` from a universal ``DMFConfig``.
        
                This factory bridges the TOML-loaded ``DMFConfig`` to
                the typed internal dataclasses (``DecayConfig``, ``VectorConfig``)
                without changing the existing constructor signature or breaking any
                existing test.
        
                Translation map
                ---------------
                ====================== ======================== =====================
                TOML key               DMFConfig path           DecayConfig / VectorConfig field
                ====================== ======================== =====================
                temporal_decay.lambda_base         decay.lambda_base          lambda_decay
                temporal_decay.inertia_strength    decay.inertia_strength     inertia_strength
                temporal_decay.hard_kill_threshold decay.hard_kill_threshold  hard_kill_threshold
                memory_tiers.critical_max          tiers.critical_max         critical_threshold
                memory_tiers.healthy_min           tiers.healthy_min          healthy_threshold
                capacity.token_budget              capacity.token_budget      token_budget
                capacity.pruning_frequency_x       capacity.pruning_frequency_x pruning_frequency
                pruning_priority.rho_*             pruning_priority.*         PruningPriorityConfig
                nlp.vector_dim                     nlp.vector_dim             VectorConfig.vector_dim
                capacity.window_size               capacity.window_size       VectorConfig.window_size
                ltm.enabled + storage_type="file"    ltm.*                    → FileLTMHook(storage_path)
                ltm.enabled + storage_type="chroma"  ltm.*                    → ChromaLTMHook(...)
                ltm.enabled + storage_type="null"    ltm.*                    → NullLTMHook
                ltm.enabled=false                     ltm.*                    → NullLTMHook
                ====================== ======================== =====================
        
                Parameters
                ----------
                config : DMFConfig
                    Fully populated config object returned by ``load_dmf_config()``.
                ltm_hook : LTMHook | None
                    Archive sink for evicted entries. Defaults to ``NullLTMHook()``.
                nlp_engine : NLPEngine | None
                    Optional NLP engine reused for recall-time contextualization of
                    raw LTM hits.
                Returns
                -------
                TemporalMemory
                    Fully initialised instance.
        
        Args:
            config: See the function signature and surrounding type hints.
            ltm_hook: See the function signature and surrounding type hints.
            nlp_engine: See the function signature and surrounding type hints.
        
        Raises:
            ValueError: Raised when validation fails or an invariant is violated.
        """
        decay_cfg = DecayConfig(
            lambda_decay=config.decay.lambda_base,
            inertia_strength=config.decay.inertia_strength,
            hard_kill_threshold=config.decay.hard_kill_threshold,
            token_budget=config.capacity.token_budget,
            pruning_frequency=config.capacity.pruning_frequency_x,
            critical_threshold=config.tiers.critical_max,
            healthy_threshold=config.tiers.healthy_min,
            ltm_recall_limit=config.ltm.recall_limit,
            ltm_threshold=config.ltm.distance_threshold,
        )
        pruning_cfg = PruningPriorityConfig(
            rho_constraint=config.pruning_priority.rho_constraint,
            rho_preference=config.pruning_priority.rho_preference,
            rho_current_state=config.pruning_priority.rho_current_state,
            rho_correction=config.pruning_priority.rho_correction,
            rho_replacement=config.pruning_priority.rho_replacement,
            superseded_past_penalty=config.pruning_priority.superseded_past_penalty,
        )
        vector_cfg = VectorConfig(
            model_name=config.nlp.model_name,
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )

        # Resolve LTM hook:
        #   1. Explicit injection always wins (testing / custom backends).
        #   2. ltm.enabled=false                    → NullLTMHook.
        #   3. ltm.enabled + storage_type="file"   → FileLTMHook.
        #   4. ltm.enabled + storage_type="chroma" → ChromaLTMHook.
        #   5. ltm.enabled + storage_type="null"   → NullLTMHook.
        # The import is deferred to this call site to avoid any future
        # circular-import risk if config_loader grows additional imports.
        resolved_hook: LTMHook
        if ltm_hook is not None:
            resolved_hook = ltm_hook
        elif not config.ltm.enabled:
            resolved_hook = NullLTMHook()
        elif config.ltm.storage_type == LTM_BACKEND_FILE:
            from dmf.memory.ltm_engine import FileLTMHook  # deferred import
            resolved_hook = FileLTMHook(
                config.ltm.storage_path,
                cards_enabled=config.ltm.cards_enabled,
                cards_path=config.ltm.cards_path,
            )
        elif config.ltm.storage_type == LTM_BACKEND_CHROMA:
            from dmf.memory.chroma_ltm import ChromaLTMHook  # deferred import
            resolved_hook = ChromaLTMHook(
                collection_name=config.ltm.collection_name,
                persist_directory=config.ltm.chroma_path,
                distance_threshold=config.ltm.distance_threshold,
                cards_enabled=config.ltm.cards_enabled,
                cards_path=config.ltm.cards_path,
                cards_collection_name=config.ltm.cards_collection_name,
            )
        elif config.ltm.storage_type == LTM_BACKEND_NULL:
            resolved_hook = NullLTMHook()
        else:
            raise ValueError(
                "Unsupported ltm.storage_type at runtime: "
                f"{config.ltm.storage_type!r}"
            )

        return cls(
            decay_config=decay_cfg,
            pruning_priority_config=pruning_cfg,
            vector_config=vector_cfg,
            ltm_hook=resolved_hook,
            nlp_engine=nlp_engine,
        )


    def add_interaction(
        self,
        text: str,
        report: AnalysisReport,
        vector: np.ndarray,
    ) -> MemoryEntry:
        """Insert a new interaction and enforce cleanup policies.
        
                Execution order
                ---------------
                1. Tokenise ``text`` via tiktoken (cached on entry).
                2. Assign monotonic ``interaction_id``, increment ``_next_id``.
                3. Update ``InteractionMatrix`` with the new vector.
                4. Construct ``MemoryEntry``, append to queue.
                5. Increment turn counter.
                6. If turn counter is a multiple of ``pruning_frequency``:
                   run ``periodic_cleanup`` (hard-kill sweep).
                7. If ``get_total_tokens() > token_budget``:
                   run ``prune_to_budget`` (pressure eviction).
        
                The returned entry is guaranteed to be in the queue at the time
                of return.  In extreme cases where the entry's own Ω is very low
                and a cascading eviction empties all other candidates, the entry
                may theoretically be evicted in the same call — this is correct
                behaviour and is guarded against in the test suite.
        
                Returns
                -------
                MemoryEntry
                    The newly created entry.
        
        Args:
            text: See the function signature and surrounding type hints.
            report: See the function signature and surrounding type hints.
            vector: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        token_count: int = len(self._tokenizer.encode(text))

        interaction_id = self._next_id
        self._next_id += 1

        self.matrix.add_vector(vector)

        entry = MemoryEntry(
            interaction_id=interaction_id,
            text=text,
            report=report,
            vector=vector,
            token_count=token_count,
            timestamp=time.time(),
        )
        self.queue.append(entry)

        self._turn_counter += 1

        if (
            self.config.pruning_frequency > 0
            and self._turn_counter % self.config.pruning_frequency == 0
        ):
            self.periodic_cleanup()

        if self.get_total_tokens() > self.config.token_budget:
            self.prune_to_budget()

        return entry

    def get_effective_state(self) -> list[dict[str, Any]]:
        """Return the decay-adjusted state of every entry in the queue.
        
                For each entry computes:
        
                    Δn      = (_next_id − 1) − entry.interaction_id
                    Ω_eff   = calculate_effective_score(Ω, Δn, λ, η)
        
                The most recently added entry always has Δn = 0 → Ω_eff = Ω.
        
                Returns
                -------
                list[dict[str, Any]]
                    One dict per entry, ordered oldest → newest.  Keys:
                    ``interaction_id``, ``text``, ``omega``, ``omega_eff``,
                    ``delta_n``, ``token_count``, ``status_effective``,
                    ``timestamp``.
        
        Raises:
            None.
        """
        if not self.queue:
            return []

        most_recent_id = self._next_id - 1
        result: list[dict[str, Any]] = []
        for entry in self.queue:
            delta_n = most_recent_id - entry.interaction_id
            omega_eff = calculate_effective_score(
                omega=entry.omega,
                delta_n=delta_n,
                lambda_decay=self.config.lambda_decay,
                inertia_strength=self.config.inertia_strength,
            )
            result.append({
                "interaction_id": entry.interaction_id,
                "text": entry.text,
                "omega": entry.omega,
                "omega_eff": omega_eff,
                "delta_n": delta_n,
                "token_count": entry.token_count,
                "status_effective": classify_survival_status(
                    omega=omega_eff,
                    critical_threshold=self.config.critical_threshold,
                    healthy_threshold=self.config.healthy_threshold,
                ).value,
                "timestamp": entry.timestamp,
            })
        return result

    def get_total_tokens(self) -> int:
        """Return the total token count across all entries in the queue.
        
                O(N) sum over pre-calculated integers — no tokenisation on hot path.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return sum(entry.token_count for entry in self.queue)

    def get_full_context(
        self,
        query_vector: np.ndarray | None = None,
    ) -> str:
        """Assemble the complete context string for LLM prompt injection.
        
        Args:
            query_vector: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        self._reset_context_metrics()
        self._reset_recall_diagnostics()
        parts: list[str] = []
        active_guard = self._build_active_context_guard(record_metrics=True)
        recalled: list[ContextualizedRecallCandidate] = []

        if query_vector is not None:
            recalled = self.rerank_contextualized_recall_candidates(
                self.contextualize_raw_recall_hits(
                    self.get_raw_recall_hits(
                        query_vector,
                        active_guard=active_guard,
                    ),
                    active_guard=active_guard,
                )
            )
            if recalled:
                parts.append(_RECALL_HEADER)
                parts.extend(self._render_recalled_context_items(recalled))
                parts.append("")

        parts.append(_ACTIVE_HEADER)
        if recalled:
            parts.append(_ACTIVE_PRECEDENCE_NOTE)
        active_lines = self._render_active_context_items(active_guard.active_entries)
        if active_lines:
            if recalled:
                parts.append("")
            parts.extend(active_lines)

        return "\n".join(parts)

    def _render_recalled_context_items(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[str]:
        lines: list[str] = []
        for index, candidate in enumerate(candidates, start=1):
            metadata = _context_metadata_parts(
                role=candidate.record.role or candidate.record.provenance.role,
                time_value=candidate.record.created_at,
                turn_value=(
                    candidate.record.provenance.source_turn
                    if candidate.record.provenance.source_turn is not None
                    else candidate.record.interaction_id
                ),
                state=_state_from_report(candidate.report),
                prefer_time_over_turn=True,
            )
            _append_context_item(
                lines,
                _render_context_item_block(
                    label=f"R{index}",
                    metadata=metadata,
                    text=candidate.record.text,
                ),
            )
        return lines

    def _render_active_context_items(
        self,
        entries: list[MemoryEntry],
    ) -> list[str]:
        lines: list[str] = []
        for index, entry in enumerate(entries, start=1):
            metadata = _context_metadata_parts(
                role=entry.provenance.role,
                time_value=entry.timestamp,
                turn_value=(
                    entry.provenance.source_turn
                    if entry.provenance.source_turn is not None
                    else entry.interaction_id
                ),
                state=_state_from_report(entry.report),
            )
            active_time = _format_context_time(entry.timestamp)
            if active_time is not None and not any(
                part.startswith("time=") for part in metadata
            ):
                metadata.append(f"time={active_time}")
            _append_context_item(
                lines,
                _render_context_item_block(
                    label=f"A{index}",
                    metadata=metadata,
                    text=entry.text,
                ),
            )
        return lines

    def _entries_for_active_context(self) -> list[MemoryEntry]:
        """Return active entries that should remain visible in the prompt."""
        return self._build_active_context_guard(record_metrics=True).active_entries

    def get_active_raw_records(self) -> list[RawLTMRecord]:
        """Return visible active entries as raw records for retrieval.
        
                Structured retrieval can use these records alongside archived LTM
                records. This keeps still-active memories searchable without changing
                the raw-LTM archival contract.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return [
            entry.to_raw_ltm_record()
            for entry in self._build_active_context_guard(record_metrics=False).active_entries
        ]

    def _build_active_context_guard(
        self,
        *,
        record_metrics: bool,
    ) -> _ActiveContextGuard:
        """Resolve active visibility, suppression, and topic winners once."""
        suppression_stats: dict[str, int] = {}
        suppressed_record_ids = self._suppressed_record_ids_from_entries(
            self.queue,
            stats=suppression_stats,
        )
        suppressed_record_ids.update(
            self._topic_superseded_record_ids_from_entries(
                self.queue,
                stats=suppression_stats,
            )
        )
        if record_metrics:
            self._context_metrics["invalidated_memory_count"] += suppression_stats.get("invalidated", 0)
            self._context_metrics["suppressed_conflict_count"] += suppression_stats.get("conflict", 0)
            self._context_metrics["superseded_memory_count"] += suppression_stats.get("superseded", 0)
        active_entries: list[MemoryEntry] = []
        for entry in self.queue:
            if entry.provenance.corrected_by_user:
                if record_metrics:
                    self._context_metrics["correction_miss_count"] += 1
                continue
            if entry.record_id in suppressed_record_ids:
                continue
            if record_metrics:
                if entry.provenance.derived_from_model:
                    self._context_metrics["model_derived_memory_count"] += 1
                else:
                    self._context_metrics["context_supported_memory_count"] += 1
            active_entries.append(entry)

        topic_winners: dict[str, MemoryEntry] = {}
        for entry in active_entries:
            identity = entry.report.topic_identity
            if not identity:
                continue
            current = topic_winners.get(identity)
            if current is None:
                topic_winners[identity] = entry
                continue
            if self._active_context_topic_rank_key(entry) > self._active_context_topic_rank_key(current):
                topic_winners[identity] = entry

        return _ActiveContextGuard(
            active_entries=active_entries,
            active_record_ids={entry.record_id for entry in active_entries},
            suppressed_record_ids=suppressed_record_ids,
            topic_winners=topic_winners,
        )

    def get_raw_recall_hits(
        self,
        query_vector: np.ndarray,
        k: int | None = None,
        *,
        active_guard: _ActiveContextGuard | None = None,
    ) -> list[RawRecallHit]:
        """Fetch raw recall hits from the configured LTM hook.
        
        Args:
            query_vector: See the function signature and surrounding type hints.
            k: See the function signature and surrounding type hints.
            active_guard: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        raw_hits = self._ltm_hook.search_raw(
            query_vector.tolist(),
            k=k if k is not None else self.config.ltm_recall_limit,
        )
        hits = self._validate_raw_recall_hits(raw_hits)
        self._recall_diagnostics["raw_candidates"] = [
            self._serialise_raw_recall_hit(hit)
            for hit in hits
        ]
        active_guard = active_guard or self._build_active_context_guard(record_metrics=False)
        return self._filter_raw_recall_hits_against_active_context(hits, active_guard)

    def contextualize_raw_recall_hits(
        self,
        hits: list[RawRecallHit],
        *,
        active_guard: _ActiveContextGuard | None = None,
    ) -> list[ContextualizedRecallCandidate]:
        """Run recall-time NLP on raw recall hits.
        
                Storage returns raw records only. Conversational interpretation
                belongs to the active-memory orchestration layer so the current
                NLP engine can be applied consistently to recalled content.
        
        Args:
            hits: See the function signature and surrounding type hints.
            active_guard: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        if self._nlp_engine is None:
            return []

        contextualized: list[ContextualizedRecallCandidate] = []
        for hit in hits:
            report = self._nlp_engine.analyze_interaction(hit.record.text)
            contextualized.append(
                ContextualizedRecallCandidate(
                    record=hit.record,
                    report=report,
                    similarity_score=hit.similarity_score,
                    distance=hit.distance,
                    rank_hint=hit.rank_hint,
                    source=hit.source,
                )
            )
        active_guard = active_guard or self._build_active_context_guard(record_metrics=False)
        return self._filter_contextualized_recall_candidates_against_active_context(
            contextualized,
            active_guard,
        )

    def rerank_contextualized_recall_candidates(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Apply deterministic selection and suppression to raw recall candidates.
        
        Args:
            candidates: See the function signature and surrounding type hints.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        filtered = self._filter_contextualized_recall_candidates(candidates)
        selected = self._select_contextualized_recall_candidates(filtered)
        ranked = self._rank_contextualized_recall_candidates(selected)
        topic_filtered = self._rank_contextualized_recall_candidates(
            self._filter_contextualized_candidates_by_topic_supersession(ranked)
        )
        deduped = self._dedupe_contextualized_candidates_by_same_topic_value(topic_filtered)
        memory_filtered = self._filter_contextualized_candidates_without_memory_signal(deduped)
        self._recall_diagnostics["ranked_candidates"] = [
            self._serialise_contextualized_recall_candidate(candidate)
            for candidate in memory_filtered
        ]
        final_candidates = self._limit_contextualized_recall_per_topic(memory_filtered)
        self._recall_diagnostics["final_candidates"] = [
            self._serialise_contextualized_recall_candidate(candidate)
            for candidate in final_candidates
        ]
        return final_candidates

    def _filter_contextualized_recall_candidates(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Drop recalled candidates that are clearly request-like, not memory-like."""
        filtered: list[ContextualizedRecallCandidate] = []
        for candidate in candidates:
            if candidate.report.is_query_like and not self._is_personal_fact_candidate(candidate):
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "query_like_filter",
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
                continue
            filtered.append(candidate)
        return filtered

    def _validate_raw_recall_hits(
        self,
        hits: object,
    ) -> list[RawRecallHit]:
        """Validate that the LTM backend returned canonical raw recall hits."""
        if not isinstance(hits, list):
            raise TypeError(
                "LTMHook.search_raw() must return list[RawRecallHit]; "
                f"got non-list result of type: {type(hits).__name__}"
            )
        invalid = [
            type(hit).__name__
            for hit in hits
            if not isinstance(hit, RawRecallHit)
        ]
        if invalid:
            joined = ", ".join(invalid)
            raise TypeError(
                "LTMHook.search_raw() must return list[RawRecallHit]; "
                f"got non-canonical hit types: {joined}"
            )
        return hits

    def _serialise_raw_recall_hit(self, hit: RawRecallHit) -> dict[str, Any]:
        """Return a JSON-serialisable representation of one raw recall hit."""
        return hit.to_dict()

    def _serialise_contextualized_recall_candidate(
        self,
        candidate: ContextualizedRecallCandidate,
    ) -> dict[str, Any]:
        """Return a JSON-serialisable representation of one contextualized candidate."""
        return candidate.to_dict()

    def _rank_contextualized_recall_candidates(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Return contextualized recall candidates ordered by deterministic policy."""
        return sorted(candidates, key=self._contextualized_recall_rank_key, reverse=True)

    def _filter_raw_recall_hits_against_active_context(
        self,
        hits: list[RawRecallHit],
        active_guard: _ActiveContextGuard,
    ) -> list[RawRecallHit]:
        """Drop raw recalls already covered or invalidated by active memory."""
        filtered: list[RawRecallHit] = []
        for hit in hits:
            record_id = hit.record.record_id
            if record_id in active_guard.active_record_ids:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "active_record_duplicate",
                        "record_id": record_id,
                        "candidate": self._serialise_raw_recall_hit(hit),
                    }
                )
                continue
            if record_id in active_guard.suppressed_record_ids:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "active_lineage_suppression",
                        "record_id": record_id,
                        "candidate": self._serialise_raw_recall_hit(hit),
                    }
                )
                continue
            filtered.append(hit)
        return filtered

    def _filter_contextualized_recall_candidates_against_active_context(
        self,
        candidates: list[ContextualizedRecallCandidate],
        active_guard: _ActiveContextGuard,
    ) -> list[ContextualizedRecallCandidate]:
        """Drop recalled candidates made redundant by newer active winners."""
        filtered: list[ContextualizedRecallCandidate] = []
        for candidate in candidates:
            identity = candidate.report.topic_identity
            if not identity:
                filtered.append(candidate)
                continue

            winner = active_guard.topic_winners.get(identity)
            if winner is None:
                filtered.append(candidate)
                continue

            candidate_value = candidate.report.topic_value
            winner_value = winner.report.topic_value
            if candidate_value and winner_value and candidate_value == winner_value:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "active_topic_duplicate",
                        "topic_identity": identity,
                        "topic_value": candidate_value,
                        "active_record_id": winner.record_id,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
                continue

            if self._is_contextualized_candidate_superseded_by_active_entry(candidate, winner):
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "active_topic_supersession",
                        "topic_identity": identity,
                        "active_record_id": winner.record_id,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
                continue

            filtered.append(candidate)
        return filtered

    def _select_contextualized_recall_candidates(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Keep one candidate per recalled raw record before topic suppression.

        Topic-level conflicts are resolved in the dedicated supersession
        stage. This pass only avoids retaining duplicate records.
        """
        selected: dict[str, ContextualizedRecallCandidate] = {}
        for candidate in candidates:
            key = candidate.record.record_id
            current = selected.get(key)
            if current is None:
                selected[key] = candidate
                continue
            if self._contextualized_recall_rank_key(candidate) > self._contextualized_recall_rank_key(current):
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "record_id_dedupe",
                        "record_id": key,
                        "candidate": self._serialise_contextualized_recall_candidate(current),
                    }
                )
                selected[key] = candidate
            else:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "record_id_dedupe",
                        "record_id": key,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
        return list(selected.values())

    def _filter_contextualized_candidates_by_topic_supersession(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Suppress contextualized recall candidates superseded on the same topic."""
        grouped: dict[str, list[ContextualizedRecallCandidate]] = {}
        passthrough: list[ContextualizedRecallCandidate] = []

        for candidate in candidates:
            identity = candidate.report.topic_identity
            value = candidate.report.topic_value
            if not identity or not value:
                passthrough.append(candidate)
                continue
            grouped.setdefault(identity, []).append(candidate)

        kept: list[ContextualizedRecallCandidate] = list(passthrough)
        for identity, group in grouped.items():
            values = {
                candidate.report.topic_value
                for candidate in group
                if candidate.report.topic_value
            }
            if len(values) <= 1:
                kept.extend(group)
                continue

            winner = max(group, key=self._contextualized_topic_supersession_rank_key)
            kept.append(winner)
            self._context_metrics["superseded_memory_count"] += max(len(group) - 1, 0)

            for candidate in group:
                if candidate is winner:
                    continue
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "topic_supersession",
                        "topic_identity": identity,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )

        return kept

    def _limit_contextualized_recall_per_topic(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Keep at most one contextualized candidate per resolved topic identity."""
        selected: list[ContextualizedRecallCandidate] = []
        seen_topics: set[str] = set()

        for candidate in candidates:
            topic = candidate.report.topic_identity or candidate.record.record_id
            if topic in seen_topics:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "topic_identity_limit",
                        "topic_identity": topic,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
                continue
            seen_topics.add(topic)
            selected.append(candidate)
        return selected

    def _dedupe_contextualized_candidates_by_same_topic_value(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Collapse duplicate recalls that resolve to the same topic/value pair."""
        selected: list[ContextualizedRecallCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        for candidate in candidates:
            identity = candidate.report.topic_identity
            value = candidate.report.topic_value
            if not identity or not value:
                selected.append(candidate)
                continue

            key = (identity, value)
            if key in seen_pairs:
                self._recall_diagnostics["suppressed"].append(
                    {
                        "reason": "same_topic_value_dedupe",
                        "topic_identity": identity,
                        "topic_value": value,
                        "candidate": self._serialise_contextualized_recall_candidate(candidate),
                    }
                )
                continue

            seen_pairs.add(key)
            selected.append(candidate)

        return selected

    def _filter_contextualized_candidates_without_memory_signal(
        self,
        candidates: list[ContextualizedRecallCandidate],
    ) -> list[ContextualizedRecallCandidate]:
        """Drop topicless/signal-less recalls when stronger memory candidates exist."""
        if not candidates:
            return []

        has_memory_bearing = any(self._is_memory_bearing_candidate(candidate) for candidate in candidates)
        if not has_memory_bearing:
            return candidates

        selected: list[ContextualizedRecallCandidate] = []
        for candidate in candidates:
            if self._is_memory_bearing_candidate(candidate):
                selected.append(candidate)
                continue
            self._recall_diagnostics["suppressed"].append(
                {
                    "reason": "non_memory_bearing_filter",
                    "candidate": self._serialise_contextualized_recall_candidate(candidate),
                }
            )
        return selected

    def _is_memory_bearing_candidate(
        self,
        candidate: ContextualizedRecallCandidate,
    ) -> bool:
        """Return whether a recalled candidate carries stable memory structure."""
        if candidate.report.topic_identity:
            return True

        signals = candidate.report.signals
        return any(
            (
                signals.is_constraint,
                signals.is_preference,
                signals.is_current_state,
                signals.is_correction,
                self._is_personal_fact_candidate(candidate),
            )
        )

    def _is_personal_fact_candidate(
        self,
        candidate: ContextualizedRecallCandidate,
    ) -> bool:
        """Return whether a candidate expresses a user autobiographical fact."""
        return (
            candidate.record.role == "user"
            and float(candidate.report.signals.personal_relevance) > 0.0
        )

    def _contextualized_recall_rank_key(
        self,
        candidate: ContextualizedRecallCandidate,
    ) -> tuple[float, int, float, int, float, int, int]:
        """Return the deterministic rank tuple for contextualized raw recalls."""
        provenance = candidate.record.provenance
        signals = candidate.report.signals
        operational_bonus = (
            (0.30 if signals.is_constraint else 0.0)
            + (0.20 if signals.is_correction else 0.0)
            + (0.15 if signals.is_current_state else 0.0)
            + (0.10 if signals.is_preference else 0.0)
        )
        personal_fact_bonus = (
            0.10 * max(float(signals.personal_relevance), 0.0)
            if candidate.record.role == "user"
            else 0.0
        )
        quantitative_bonus = 0.1 * max(float(signals.quantitative_relevance), 0.0)
        ack_penalty = (
            0.35
            if candidate.record.role == "assistant" and candidate.report.is_ack_like
            else 0.0
        )
        effective_similarity = (
            float(candidate.similarity_score)
            if candidate.similarity_score is not None
            else float("-inf")
        )
        effective_similarity += operational_bonus
        effective_similarity += personal_fact_bonus
        effective_similarity += quantitative_bonus
        effective_similarity -= ack_penalty
        return (
            effective_similarity,
            0 if provenance.corrected_by_user else 1,
            float(candidate.report.info_density),
            1 if candidate.record.role == "user" else 0,
            float(candidate.record.created_at),
            int(candidate.record.interaction_id),
            -int(candidate.rank_hint if candidate.rank_hint is not None else 1_000_000),
        )

    def _contextualized_topic_supersession_rank_key(
        self,
        candidate: ContextualizedRecallCandidate,
    ) -> tuple[int, int, int, float, int, float, int]:
        """Return the winner key for conflicting contextualized raw recalls."""
        signals = candidate.report.signals
        return (
            1 if signals.is_current_state else 0,
            0 if signals.is_past_state else 1,
            1 if candidate.record.role == "user" else 0,
            float(candidate.record.created_at),
            int(candidate.record.interaction_id),
            float(candidate.similarity_score if candidate.similarity_score is not None else float("-inf")),
            -int(candidate.rank_hint if candidate.rank_hint is not None else 1_000_000),
        )

    def _active_context_topic_rank_key(
        self,
        entry: MemoryEntry,
    ) -> tuple[int, int, int, float, int]:
        """Return the active-memory winner key for one topic."""
        signals = entry.report.signals
        return (
            1 if signals.is_current_state else 0,
            0 if signals.is_past_state else 1,
            1 if entry.provenance.role == "user" else 0,
            float(entry.timestamp),
            int(entry.interaction_id),
        )

    def _is_contextualized_candidate_superseded_by_active_entry(
        self,
        candidate: ContextualizedRecallCandidate,
        active_entry: MemoryEntry,
    ) -> bool:
        """Return whether a newer active entry supersedes one recalled candidate."""
        if active_entry.interaction_id <= candidate.record.interaction_id:
            return False

        candidate_identity = candidate.report.topic_identity
        active_identity = active_entry.report.topic_identity
        if (
            not candidate_identity
            or not active_identity
            or candidate_identity != active_identity
        ):
            return False

        candidate_value = candidate.report.topic_value
        active_value = active_entry.report.topic_value
        if (
            not candidate_value
            or not active_value
            or candidate_value == active_value
        ):
            return False

        older_signals = candidate.report.signals
        newer_signals = active_entry.report.signals

        if older_signals.is_past_state and newer_signals.is_current_state:
            return True
        if older_signals.is_preference and newer_signals.is_preference:
            return True
        if older_signals.is_current_state and newer_signals.is_current_state:
            return True
        return False

    def _suppressed_record_ids_from_entries(
        self,
        entries: deque[MemoryEntry],
        stats: dict[str, int] | None = None,
    ) -> set[str]:
        """Return source-record ids explicitly superseded or invalidated."""
        suppressed: set[str] = set()
        for entry in entries:
            suppressed.update(self._lineage_targets(entry.lineage, stats=stats))
        return suppressed

    def _lineage_targets(
        self,
        lineage: MemoryLineage | None,
        stats: dict[str, int] | None = None,
    ) -> set[str]:
        """Return ids that should be suppressed by explicit lineage edges."""
        if lineage is None:
            return set()

        targets: set[str] = set()
        for field_name in ("supersedes", "corrects", "invalidates"):
            for value in getattr(lineage, field_name):
                if value:
                    targets.add(str(value))
                    if stats is not None:
                        if field_name == "supersedes":
                            stats["superseded"] = stats.get("superseded", 0) + 1
                        elif field_name == "invalidates":
                            stats["invalidated"] = stats.get("invalidated", 0) + 1
                        elif field_name == "corrects":
                            stats["conflict"] = stats.get("conflict", 0) + 1
        return targets

    def _topic_superseded_record_ids_from_entries(
        self,
        entries: deque[MemoryEntry],
        stats: dict[str, int] | None = None,
    ) -> set[str]:
        """Return older entries superseded by newer topic updates in the queue."""
        suppressed: set[str] = set()
        ordered = list(entries)
        for older_idx, older in enumerate(ordered):
            for newer in ordered[older_idx + 1 :]:
                if self._is_topic_superseded(older, newer):
                    suppressed.add(older.record_id)
                    if stats is not None:
                        stats["superseded"] = stats.get("superseded", 0) + 1
                    break
        return suppressed

    def _is_topic_superseded(self, older: MemoryEntry, newer: MemoryEntry) -> bool:
        """Return whether a newer entry supersedes an older one on the same topic."""
        if newer.interaction_id <= older.interaction_id:
            return False

        older_identity = older.report.topic_identity
        newer_identity = newer.report.topic_identity
        if not older_identity or not newer_identity or older_identity != newer_identity:
            return False

        older_value = older.report.topic_value
        newer_value = newer.report.topic_value
        if not older_value or not newer_value or older_value == newer_value:
            return False

        older_signals = older.report.signals
        newer_signals = newer.report.signals

        if older_signals.is_past_state and newer_signals.is_current_state:
            return True
        if older_signals.is_preference and newer_signals.is_preference:
            return True
        if older_signals.is_current_state and newer_signals.is_current_state:
            return True
        return False

    def _reset_context_metrics(self) -> None:
        """Reset lightweight context metrics before rebuilding prompt context."""
        for key in self._context_metrics:
            self._context_metrics[key] = 0

    def _empty_recall_diagnostics(self) -> dict[str, Any]:
        """Return the empty recall-diagnostics structure."""
        return {
            "raw_candidates": [],
            "ranked_candidates": [],
            "final_candidates": [],
            "suppressed": [],
        }

    def _reset_recall_diagnostics(self) -> None:
        """Reset recall diagnostics before rebuilding prompt context."""
        self._recall_diagnostics = self._empty_recall_diagnostics()

    def get_context_metrics(self) -> dict[str, int]:
        """Return a snapshot of the latest context reconstruction metrics.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return dict(self._context_metrics)

    def get_recall_diagnostics(self) -> dict[str, Any]:
        """Return a snapshot of the latest recall diagnostics.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return {
            "raw_candidates": list(self._recall_diagnostics["raw_candidates"]),
            "ranked_candidates": list(self._recall_diagnostics["ranked_candidates"]),
            "final_candidates": list(self._recall_diagnostics["final_candidates"]),
            "suppressed": list(self._recall_diagnostics["suppressed"]),
        }


    def _omega_eff_for(self, entry: MemoryEntry) -> float:
        """Compute the current Ω_eff for a single entry.

        Centralises the Δn arithmetic so all pruning methods use the
        same calculation without duplicating it.
        """
        if self._next_id == 0:
            return entry.omega
        most_recent_id = self._next_id - 1
        delta_n = most_recent_id - entry.interaction_id
        return calculate_effective_score(
            omega=entry.omega,
            delta_n=delta_n,
            lambda_decay=self.config.lambda_decay,
            inertia_strength=self.config.inertia_strength,
        )

    def _retention_bonus_for(self, entry: MemoryEntry) -> float:
        """Return the pruning-only operational retention bonus for an entry."""
        signals = entry.report.signals
        cfg = self._pruning_priority_config
        return (
            (cfg.rho_constraint if signals.is_constraint else 0.0)
            + (cfg.rho_preference if signals.is_preference else 0.0)
            + (cfg.rho_current_state if signals.is_current_state else 0.0)
            + (cfg.rho_correction if signals.is_correction else 0.0)
            + (cfg.rho_replacement if signals.has_replacement else 0.0)
        )

    def _topic_superseded_penalty_for(self, entry: MemoryEntry) -> float:
        """Return the pruning penalty for an entry superseded on the same topic."""
        if entry.record_id not in self._topic_superseded_record_ids_from_entries(self.queue):
            return 0.0
        return self._pruning_priority_config.superseded_past_penalty

    def _topic_superseded_penalty_from_cached_ids(
        self,
        entry: MemoryEntry,
        superseded_record_ids: set[str],
    ) -> float:
        """Return the pruning penalty using one precomputed superseded-id set."""
        if entry.record_id not in superseded_record_ids:
            return 0.0
        return self._pruning_priority_config.superseded_past_penalty

    def _effective_pruning_score_for(self, entry: MemoryEntry) -> float:
        """Return the pressure-pruning score used for eviction ordering."""
        return (
            self._omega_eff_for(entry)
            + self._retention_bonus_for(entry)
            - self._topic_superseded_penalty_for(entry)
        )

    def _get_pruning_candidates(self) -> list[MemoryEntry]:
        """Return entries sorted by eviction priority (highest risk first).

        Only CRITICAL and UNSTABLE entries are pressure candidates.
        HEALTHY entries are protected from budget-pressure eviction.

        Candidate ordering uses ``effective_pruning_score`` ascending, with
        ``interaction_id`` ascending as the tie-breaker. Tier membership
        still depends on raw ``Ω_eff`` only, so pruning bonuses cannot
        promote an entry to HEALTHY or exempt it from decay.

        HEALTHY (Ω_eff > 0.6) messages are **not** in this list.
        They are only removed by ``periodic_cleanup`` once their
        Ω_eff decays below ``hard_kill_threshold`` (≈ 141 turns for
        Ω = 0.85 at default calibration).

        Returns
        -------
        list[MemoryEntry]
            Eviction-priority ordered references into the live queue.
            The queue itself is not modified.
        """
        candidates: list[MemoryEntry] = []
        superseded_record_ids = self._topic_superseded_record_ids_from_entries(self.queue)

        for entry in self.queue:
            omega_eff = self._omega_eff_for(entry)
            if omega_eff <= self.config.critical_threshold:
                candidates.append(entry)
            elif omega_eff <= self.config.healthy_threshold:
                candidates.append(entry)

        candidates.sort(
            key=lambda e: (
                self._omega_eff_for(e)
                + self._retention_bonus_for(e)
                - self._topic_superseded_penalty_from_cached_ids(
                    e,
                    superseded_record_ids,
                ),
                e.interaction_id,
            )
        )
        return candidates


    def prune_to_budget(self) -> list[MemoryEntry]:
        """Evict lowest-priority entries until total tokens ≤ token_budget.
        
                Eviction policy
                ---------------
                1. Compute the full priority list via ``_get_pruning_candidates``.
                2. Remove the head of the list (oldest CRITICAL, else oldest
                   UNSTABLE) from the queue.
                3. Sync ``InteractionMatrix`` by removing the same vector object.
                4. Archive the evicted entry via ``LTMHook.archive``.
                5. Repeat until ``get_total_tokens() ≤ config.token_budget`` or
                   no candidates remain.
        
                If only HEALTHY messages remain and the budget is still exceeded,
                the loop exits without evicting them.  This is intentional —
                HEALTHY messages are protected by design.  The budget deficit is
                logged via the return value for the caller to handle.
        
                Returns
                -------
                list[MemoryEntry]
                    All entries evicted in this call, in eviction order.
        
        Raises:
            None.
        """
        evicted: list[MemoryEntry] = []
        while self.get_total_tokens() > self.config.token_budget:
            candidates = self._get_pruning_candidates()
            if not candidates:
                # Only HEALTHY entries remain; cannot evict by budget pressure.
                break
            victim = candidates[0]
            self._evict(victim)
            evicted.append(victim)
        return evicted


    def periodic_cleanup(self) -> list[MemoryEntry]:
        """Hard-kill sweep: evict any entry with Ω_eff < hard_kill_threshold.
        
                Unlike ``prune_to_budget``, this sweep applies to **all** entries
                including HEALTHY messages that have organically decayed to near-zero.
                It runs every ``config.pruning_frequency`` turns (triggered by
                ``add_interaction``).
        
                A HEALTHY message (Ω = 0.85) reaches the default hard-kill floor
                (0.05) at Δn ≈ 141 turns. The periodic sweep removes these stale
                entries from active memory before they accumulate and inflate the
                token budget.
        
                Returns
                -------
                list[MemoryEntry]
                    All entries evicted in this sweep, in queue order.
        
        Raises:
            None.
        """
        threshold = self.config.hard_kill_threshold
        to_evict = [
            entry for entry in self.queue
            if self._omega_eff_for(entry) < threshold
        ]
        for entry in to_evict:
            self._evict(entry)
        return to_evict


    def _evict(self, entry: MemoryEntry) -> None:
        """Remove entry from queue and matrix, then archive it to LTM.

        This is the single eviction primitive — all removal paths call
        through here to guarantee consistent teardown.

        Steps
        -----
        1. Remove from ``queue`` (O(N) deque scan — acceptable because
           the queue is bounded by the token budget in practice).
        2. Attempt to remove the same vector object from ``InteractionMatrix``
           (O(W), where W = window_size ≤ 10).  No-op if the vector has
           already been scrolled out of the sliding window.
        3. Call ``ltm_hook.archive(entry)`` — fire-and-forget handoff.
        """
        self.queue.remove(entry)
        self.matrix.remove_vector(entry.vector)
        self._ltm_hook.archive(entry)


    @property
    def size(self) -> int:
        """Number of entries currently in the working memory queue.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return len(self.queue)

    @property
    def is_empty(self) -> bool:
        """True when the working memory queue contains no entries.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return len(self.queue) == 0

    @property
    def next_id(self) -> int:
        """The interaction_id that will be assigned to the next insertion.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self._next_id

    @property
    def ltm_hook(self) -> LTMHook:
        """The configured LTM archive hook.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self._ltm_hook
