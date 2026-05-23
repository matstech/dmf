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

"""Categorical classification of a Survival Score (Ω).

The scoring engine computes a continuous Ω ∈ (0, 1). For human-readable
reporting, pruning decisions, and structured logging, Ω is bucketed into three
tiers using runtime-provided thresholds:

    HEALTHY   Ω > healthy_threshold   — high retention value, safe from pruning.
    UNSTABLE  Ω > critical_threshold  — borderline; at risk under memory pressure.
    CRITICAL  Ω ≤ critical_threshold  — low retention value, first to be pruned.

The enum inherits from ``str`` so that instances serialise cleanly via
``json.dumps`` and ``dataclasses.asdict`` without a custom encoder.
"""

from __future__ import annotations

from enum import Enum


def classify_survival_status(
    omega: float,
    critical_threshold: float,
    healthy_threshold: float,
) -> SurvivalStatus:
    """Classify a Survival Score using caller-provided runtime thresholds.
    
        Boundary rules:
          - HEALTHY:  omega > healthy_threshold
          - UNSTABLE: critical_threshold < omega <= healthy_threshold
          - CRITICAL: omega <= critical_threshold
    
        Parameters
        ----------
        omega : float
            Survival Score to classify.
        critical_threshold : float
            Upper bound for the CRITICAL tier.
        healthy_threshold : float
            Lower bound for the HEALTHY tier.
    
        Returns
        -------
        SurvivalStatus
            Tier derived from the provided thresholds.
    
    Args:
        omega: See the function signature and surrounding type hints.
        critical_threshold: See the function signature and surrounding type hints.
        healthy_threshold: See the function signature and surrounding type hints.
    
    Raises:
        None.
    """
    if omega > healthy_threshold:
        return SurvivalStatus.HEALTHY
    if omega > critical_threshold:
        return SurvivalStatus.UNSTABLE
    return SurvivalStatus.CRITICAL


class SurvivalStatus(str, Enum):
    """Categorical tier derived from the continuous Survival Score Ω.
    
        Boundary rules:
          - HEALTHY:  Ω > 0.6
          - UNSTABLE: 0.3 < Ω ≤ 0.6
          - CRITICAL: Ω ≤ 0.3
    
    Args:
        None.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    HEALTHY = "HEALTHY"
    UNSTABLE = "UNSTABLE"
    CRITICAL = "CRITICAL"
