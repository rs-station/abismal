"""The --activation / --pre-activation choices must all actually work.

The list is curated by hand. A hand-written list can drift from what the layer
accepts, and the
failure would land at model-build time in the middle of a run rather than at
argument-parse time, so it is checked here.
"""
import numpy as np
import pytest
import torch

from abismal.command_line.parser.architecture import (
    NAMED_ACTIVATIONS,
    TRANSFORM_CHOICES,
)
from abismal.command_line.parser import parser
from abismal.layers import FeedForward


@pytest.mark.parametrize("name", TRANSFORM_CHOICES)
def test_every_choice_builds_a_working_layer(name):
    """Usable in both slots, not merely resolvable."""
    layer = FeedForward(8, pre_activation=name, activation=name, epsilon=1e-6)
    out = layer(torch.randn((4, 8)))
    assert torch.isfinite(out).all(), f"{name} produced non-finite output"


@pytest.mark.parametrize("name", TRANSFORM_CHOICES)
def test_every_choice_is_accepted_by_the_parser(name):
    for flag in ("--activation", "--pre-activation"):
        ns = parser.parse_args([f"{flag}={name}", "-d", "2.0", "dummy.mtz"])
        assert getattr(ns, flag.lstrip("-").replace("-", "_")) == name


def test_every_normalizer_is_offered():
    """The normalizers are derived from the layer, so adding one reaches the CLI."""
    missing = set(FeedForward.norm_dict) - set(TRANSFORM_CHOICES)
    assert not missing, f"norm_dict entries absent from the CLI choices: {sorted(missing)}"


def test_named_activations_are_not_normalizers():
    """A name in both lists would resolve as a normalizer and shadow the activation."""
    overlap = set(NAMED_ACTIVATIONS) & set(FeedForward.norm_dict)
    assert not overlap, f"names claimed by both: {sorted(overlap)}"


def test_defaults_are_selectable():
    ns = parser.parse_args(["-d", "2.0", "dummy.mtz"])
    assert ns.activation in TRANSFORM_CHOICES
    assert ns.pre_activation in TRANSFORM_CHOICES


def test_an_unlisted_activation_is_rejected():
    """torch knows `mish`; the CLI deliberately does not offer it."""
    with pytest.raises(SystemExit):
        parser.parse_args(["--pre-activation=mish", "-d", "2.0", "dummy.mtz"])
