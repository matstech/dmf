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

"""Temporal decay helpers for Effective Survival Score (Ω_eff).

This module attenuates the static Survival Score Ω over interaction turns to
model the natural fading of conversational relevance.

Formula
-------
    μ       = 1 − η · Ω           (inertia multiplier, per-message)
    Ω_eff   = Ω · exp(−λ · μ · Δn)

The decay axis is **interaction count** (Δn), not wall-clock time.
This guarantees deterministic, reproducible memory states: the same
sequence of messages always produces the same pruning decisions,
regardless of execution pacing.

Inertia Effect
--------------
High-quality messages (large Ω) decay slower because their μ is smaller:

    Ω = 0.85 → μ = 0.575 → eff. λ = 0.0201 → half-life ≈ 34.5 turns
    Ω = 0.15 → μ = 0.925 → eff. λ = 0.0324 → half-life ≈ 21.4 turns

The survival gap *widens* over time — entity-rich anchors consistently
outrank phatic cues in any ``argmin(Ω_eff)`` eviction sweep.

CPU-First Guarantee
-------------------
Pure ``math.exp`` arithmetic.  No I/O, no model inference, no NumPy.
Sub-microsecond per call.
"""

from __future__ import annotations

import math


def calculate_effective_score(
    omega: float,
    delta_n: int,
    lambda_decay: float = 0.035,
    inertia_strength: float = 0.5,
) -> float:
    """Compute the Effective Survival Score Ω_eff after Δn turns.
    
        Parameters
        ----------
        omega : float
            Static Survival Score Ω ∈ [0, 1) as produced by
            ``ScoringEngine.calculate_score``.
        delta_n : int
            Number of newer interactions that have arrived since this
            message was created.  Δn = 0 for the most recent message.
        lambda_decay : float
            Base decay rate λ.  Higher → faster forgetting.
        inertia_strength : float
            Inertia coefficient η ∈ [0, 1).  Higher → high-Ω messages
            resist decay more strongly.
    
        Returns
        -------
        float
            Ω_eff ∈ [0, Ω], rounded to 6 decimal places.
    
        Contract
        --------
        - Δn = 0  → returns exactly Ω  (no decay on the latest message).
        - Ω = 0   → returns 0.0        (nothing to save).
        - Ω_eff is monotonically non-increasing in Δn.
        - Ω_eff ≤ Ω always (decay never amplifies).
    
        Performance
        -----------
        One ``math.exp`` call + four multiplications.  No branching in the
        hot path beyond the two early-exit guards.
    
    Args:
        omega: See the function signature and surrounding type hints.
        delta_n: See the function signature and surrounding type hints.
        lambda_decay: See the function signature and surrounding type hints.
        inertia_strength: See the function signature and surrounding type hints.
    
    Raises:
        None.
    """
    # Zero is an absorbing state for survival, so the exponential path would
    # only add work without changing the result.
    if omega == 0.0:
        return 0.0

    # Negative deltas can appear when callers compare against a future anchor;
    # treating them as "not aged yet" keeps decay monotonic.
    if delta_n <= 0:
        return omega

    # See: Ebbinghaus, 1885, "Memory: A Contribution to Experimental
    # Psychology"; this uses the standard exponential forgetting form with
    # score-dependent inertia.
    mu = 1.0 - inertia_strength * omega
    exponent = -lambda_decay * mu * delta_n
    omega_eff = omega * math.exp(exponent)

    return round(omega_eff, 6)


def turns_to_hard_kill(
    omega: float,
    lambda_decay: float = 0.035,
    inertia_strength: float = 0.5,
    hard_kill_threshold: float = 0.05,
) -> int | None:
    """Calculate the exact Δn at which Ω_eff drops below the hard-kill line.
    
        Solves:  Ω · exp(−λ · μ · Δn) < Ω_kill
             →   Δn > −ln(Ω_kill / Ω) / (λ · μ)
    
        Parameters
        ----------
        omega : float
            Static Survival Score.
        lambda_decay : float
            Base decay rate.
        inertia_strength : float
            Inertia coefficient.
        hard_kill_threshold : float
            The Ω_eff value below which the message is evicted.
    
        Returns
        -------
        int | None
            The first Δn at which Ω_eff < hard_kill_threshold (ceiling of
            the continuous solution).  Returns ``None`` if Ω is already at
            or below the threshold (dead on arrival), and ``None`` for Ω = 0
            (cannot decay from zero).
    
    Args:
        omega: See the function signature and surrounding type hints.
        lambda_decay: See the function signature and surrounding type hints.
        inertia_strength: See the function signature and surrounding type hints.
        hard_kill_threshold: See the function signature and surrounding type hints.
    
    Raises:
        None.
    """
    # The caller needs a concrete first kill turn; already-dead records are
    # therefore reported as killable immediately.
    if omega <= hard_kill_threshold:
        return 0

    mu = 1.0 - inertia_strength * omega
    if mu <= 0.0:
        # Theoretically unreachable (η < 1 ⇒ μ > 0 for Ω < 1/η),
        # but guard against degenerate config.
        return None

    denominator = lambda_decay * mu
    if denominator <= 0.0:
        # With no effective decay rate, the hard-kill threshold is unreachable.
        return None

    ratio = hard_kill_threshold / omega
    delta_n = -math.log(ratio) / denominator

    return math.ceil(delta_n)
