"""Legacy audit helper primitives for the Deterministic Memory Framework.

`AuditState` is not part of the active runtime contract of the current DMF
pipeline. The core modules expose structured domain results directly
(`AnalysisReport`, `MemoryEntry`, raw recall diagnostics, context metrics)
and do not return `(Result, AuditState)` pairs.

This module is kept as an isolated helper surface until the final cleanup pass.
It should be imported explicitly from `dmf.runtime.base` when needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from dmf.utils.constants import SECONDS_TO_MILLISECONDS


@dataclass
class AuditState:
    """Immutable audit envelope attached to every DMF computation result.

    Captures *how* a result was produced, not just *what* was produced.
    This enables offline benchmarking, hyperparameter tuning, and full
    post-execution inspection without re-running any computation.

    Args:
        module_name: Identifier of the DMF module that produced this state.
        operation: Name of the specific operation performed.
        latency_ms: Wall-clock time elapsed during the operation, in
            milliseconds.
        inputs: Snapshot of the inputs provided to the operation.
        intermediate_weights: Intermediate values computed during the operation.
        logical_decisions: Boolean or categorical flags that drove branching
            logic.
        metadata: Additional context that does not fit the above categories.
        created_at: Unix timestamp in seconds when this audit state was created.

    Returns:
        Dataclass instance containing audit metadata.

    Raises:
        None.
    """

    module_name: str
    operation: str
    latency_ms: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    intermediate_weights: dict[str, Any] = field(default_factory=dict)
    logical_decisions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full audit state to a plain dictionary.

        Returns:
            Dictionary suitable for JSON logging or persistence beside archived
            interactions.

        Raises:
            None.
        """
        return {
            "module_name": self.module_name,
            "operation": self.operation,
            "latency_ms": self.latency_ms,
            "inputs": self.inputs,
            "intermediate_weights": self.intermediate_weights,
            "logical_decisions": self.logical_decisions,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def start_timer(cls, module_name: str, operation: str) -> _AuditTimer:
        """Return a context manager that produces an ``AuditState`` on exit.

        Args:
            module_name: Identifier of the module being timed.
            operation: Operation name being timed.

        Returns:
            Timer object whose ``state`` attribute is populated after the
            context exits.

        Raises:
            None.
        """
        return _AuditTimer(module_name=module_name, operation=operation)

class _AuditTimer:
    """Context-manager that records elapsed time and assembles AuditState.

    Not intended to be instantiated directly; use AuditState.start_timer().
    """

    def __init__(self, module_name: str, operation: str) -> None:
        self.module_name = module_name
        self.operation = operation
        self.inputs: dict[str, Any] = {}
        self.intermediate_weights: dict[str, Any] = {}
        self.logical_decisions: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.state: AuditState | None = None
        self._start: float = 0.0

    def __enter__(self) -> _AuditTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * SECONDS_TO_MILLISECONDS
        self.state = AuditState(
            module_name=self.module_name,
            operation=self.operation,
            latency_ms=round(elapsed_ms, 4),
            inputs=self.inputs,
            intermediate_weights=self.intermediate_weights,
            logical_decisions=self.logical_decisions,
            metadata=self.metadata,
        )
