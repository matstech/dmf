"""Performance measurement utilities for the Deterministic Memory Framework.

ExecutionLatencyTimer is a focused, single-responsibility context manager
that only measures wall-clock time. It is intentionally lower-level than
``runtime.base._AuditTimer``: it carries no AuditState logic, making it
safely composable inside any utility or helper that does not need the full
audit envelope — for example, within the NLP Engine's inner extraction
functions whose latency is then rolled up into an AnalysisReport.
"""

from __future__ import annotations

import time
from typing import Any

from dmf.utils.constants import SECONDS_TO_MILLISECONDS


class ExecutionLatencyTimer:
    """Context manager that measures wall-clock execution time in milliseconds.

    Uses ``time.perf_counter()`` for sub-millisecond precision.
    Safe to reuse: each entry into a 'with' block resets the internal state,
    so the same instance can measure independent operations sequentially.

    Accessing elapsed_ms before the 'with' block has exited raises a
    RuntimeError with an explicit message, preventing silent misuse where
    a caller might read a stale or uninitialized value.

    Args:
        None.

    Returns:
        Context manager instance whose ``elapsed_ms`` property becomes available
        after a completed ``with`` block.

    Raises:
        RuntimeError: If ``elapsed_ms`` is accessed before a measurement has
            completed.
    """

    def __init__(self) -> None:
        self._start_time: float = 0.0
        # A genuine measurement can round to 0.0 ms, so unmeasured state needs
        # a sentinel distinct from any float.
        self._elapsed_ms: float | None = None

    def __enter__(self) -> ExecutionLatencyTimer:
        # Sequential reuse is common in tests and diagnostics, so each block
        # must hide the previous measurement until the new block exits.
        self._elapsed_ms = None
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        raw_elapsed_seconds = time.perf_counter() - self._start_time
        self._elapsed_ms = round(raw_elapsed_seconds * SECONDS_TO_MILLISECONDS, 4)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds for the last completed block.

        Returns:
            Elapsed wall-clock time in milliseconds.

        Raises:
            RuntimeError: If accessed before the current ``with`` block has
                exited or before any measurement has completed.
        """
        if self._elapsed_ms is None:
            raise RuntimeError(
                "elapsed_ms is not available before the 'with' block exits. "
                "Use ExecutionLatencyTimer inside a 'with' statement and "
                "read elapsed_ms only after the block has completed."
            )
        return self._elapsed_ms
