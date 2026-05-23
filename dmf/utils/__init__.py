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

"""Shared configuration and utility exports for DMF."""

from dmf.utils.config import NLPConfig, VectorConfig, ScoringConfig, DecayConfig
from dmf.utils.config_loader import (
    DMFConfig,
    NLPSettings,
    ScoringWeightsSettings,
    TemporalDecaySettings,
    MemoryTiersSettings,
    CapacitySettings,
    LTMSettings,
    load_dmf_config,
)
from dmf.utils.perf import ExecutionLatencyTimer

__all__ = [
    "NLPConfig",
    "VectorConfig",
    "ScoringConfig",
    "DecayConfig",
    "DMFConfig",
    "NLPSettings",
    "ScoringWeightsSettings",
    "TemporalDecaySettings",
    "MemoryTiersSettings",
    "CapacitySettings",
    "LTMSettings",
    "load_dmf_config",
    "ExecutionLatencyTimer",
]
