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
tests/test_config.py
--------------------
Unit tests for dmf/utils/config.py — NLPConfig dataclass.

Coverage:
  - Default field values are correct and match SPEC §3.
  - Custom values are stored exactly as provided.
  - frozen=True enforcement: assignment raises FrozenInstanceError.
  - frozen=True enforcement: deletion raises FrozenInstanceError.
  - Two instances with identical fields compare as equal (dataclass __eq__).
  - Two instances with different fields compare as not equal.
"""

import dataclasses

import pytest

from dmf.utils.config import NLPConfig


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

def test_spacy_model_default_is_en_core_web_sm():
    """spacy_model must default to 'en_core_web_sm' per SPEC §3."""
    config = NLPConfig()
    assert config.spacy_model == "en_core_web_sm"


def test_analyze_system_prompt_default_is_false():
    """analyze_system_prompt must default to False per SPEC §3."""
    config = NLPConfig()
    assert config.analyze_system_prompt is False


# ---------------------------------------------------------------------------
# Custom values
# ---------------------------------------------------------------------------

def test_custom_spacy_model_is_stored_correctly():
    """A non-default spacy_model value must be stored exactly as provided."""
    config = NLPConfig(spacy_model="en_core_web_md")
    assert config.spacy_model == "en_core_web_md"


def test_analyze_system_prompt_true_is_stored_correctly():
    """analyze_system_prompt=True must be stored as the boolean True."""
    config = NLPConfig(analyze_system_prompt=True)
    assert config.analyze_system_prompt is True


def test_both_custom_values_stored_independently():
    """Both fields set to non-default values must be stored independently."""
    config = NLPConfig(spacy_model="en_core_web_lg", analyze_system_prompt=True)
    assert config.spacy_model == "en_core_web_lg"
    assert config.analyze_system_prompt is True


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------

def test_assigning_to_spacy_model_raises_frozen_instance_error():
    """Mutating spacy_model after construction must raise FrozenInstanceError."""
    config = NLPConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.spacy_model = "en_core_web_lg"  # type: ignore[misc]


def test_assigning_to_analyze_system_prompt_raises_frozen_instance_error():
    """Mutating analyze_system_prompt after construction must raise FrozenInstanceError."""
    config = NLPConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.analyze_system_prompt = True  # type: ignore[misc]


def test_deleting_a_field_raises_frozen_instance_error():
    """Deleting any field after construction must raise FrozenInstanceError."""
    config = NLPConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        del config.spacy_model  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

def test_two_default_instances_are_equal():
    """Two NLPConfig instances with identical fields must compare as equal."""
    assert NLPConfig() == NLPConfig()


def test_instances_with_different_fields_are_not_equal():
    """NLPConfig instances with different field values must not be equal."""
    assert NLPConfig(spacy_model="en_core_web_sm") != NLPConfig(spacy_model="en_core_web_md")
