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

"""Canonical data contracts for analysis, memory, and recall."""

from dmf.models.analysis import AnalysisReport, InteractionProvenance, InteractionSignals, MemoryLineage
from dmf.models.ltm_hook import CardSearchLTMHook, LTMHook, NullLTMHook
from dmf.models.memory import (
    MemoryEntry,
    MemoryCard,
    MemoryCardProvenance,
    MemoryCardTimeAnchor,
    MemoryCardValidity,
    QueryFrame,
    RetrievedEvidence,
)
from dmf.models.raw_ltm import ContextualizedRecallCandidate, RawLTMRecord, RawRecallHit
from dmf.models.recall_filter import RecallFilter
from dmf.models.status import SurvivalStatus, classify_survival_status

__all__ = [
    "AnalysisReport",
    "InteractionProvenance",
    "InteractionSignals",
    "MemoryLineage",
    "CardSearchLTMHook",
    "LTMHook",
    "NullLTMHook",
    "MemoryEntry",
    "MemoryCard",
    "MemoryCardProvenance",
    "MemoryCardTimeAnchor",
    "MemoryCardValidity",
    "QueryFrame",
    "RetrievedEvidence",
    "ContextualizedRecallCandidate",
    "RawLTMRecord",
    "RawRecallHit",
    "RecallFilter",
    "SurvivalStatus",
    "classify_survival_status",
]
