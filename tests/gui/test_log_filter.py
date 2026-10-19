"""What the log panel shows, and what it hides.

A library that narrates its startup through the same stderr abismal reports on
buries abismal's output: in a fixed log box it is what a user sees first, and on
a short run sometimes all they see. TensorFlow did; PyTorch does not, so nothing
is hidden today. These tests pin the mechanism with stand-in markers, and pin
that a real run's output passes through untouched.
"""
from pathlib import Path

import pytest

from abismal.gui import _log_filter
from abismal.gui._log_filter import NoiseFilter, is_noise, strip_noise


SIGNAL = [
    "Epoch 1/12",
    "1000/1000 - 187s - loss: 10.9483 - NLL: 5.3453 - KL: 2.1794",
    "Freezing intensity standardization layer...",
    "RuntimeWarning: skipping torchref at epoch 5: 1 earlier run(s) still going",
    "UserWarning: Early stopping conditioned on metric `val_wCCpred` which is not available.",
    "Traceback (most recent call last):",
    "ValueError: something abismal actually said",
]


@pytest.fixture
def chatter(monkeypatch):
    """A stand-in library that announces itself, so the mechanism has something to drop."""
    monkeypatch.setattr(_log_filter, "NOISE_MARKERS", ("Registered plugin",))
    return "I0000 00:00:1778869082.207106 plugin.cc:12] Registered plugin cuFoo"


def test_marked_lines_are_dropped(chatter):
    assert is_noise(chatter)


def test_abismal_output_and_warnings_survive(chatter):
    """Matched on message text, never on severity.

    A logging prefix like absl's I0000 is shared with anything logging through
    it, so dropping a severity would take abismal's own warnings with it.
    """
    for line in SIGNAL:
        assert not is_noise(line), line
    assert not is_noise("I0000 00:00:1778869082.2 trainer.py:9] abismal said this")


def test_a_continuation_under_noise_goes_with_it(chatter):
    """Dropping a header and keeping its indented continuations is worse than
    not filtering: the panel fills with orphaned fragments."""
    noise = NoiseFilter()
    assert noise.keep(chatter) is False
    assert noise.keep("\t [[{{node IteratorGetNext}}]]") is False
    assert noise.keep("    while registering cuFoo") is False


def test_a_continuation_under_a_real_error_is_kept(chatter):
    """The same kind of line is the most useful detail in a genuine traceback.

    This is the whole reason the filter is stateful -- deciding per line would
    either keep orphaned continuations or throw away the detail that localises a
    real failure.
    """
    noise = NoiseFilter()
    assert noise.keep("Traceback (most recent call last):") is True
    assert noise.keep('  File "abismal/training/__init__.py", line 9, in fit') is True
    assert noise.keep("ValueError: Incompatible shapes") is True
    assert noise.keep("\t [[{{node IteratorGetNext}}]]") is True


def test_the_filter_resets_between_blocks(chatter):
    noise = NoiseFilter()
    noise.keep(chatter)                                    # noise
    assert noise.keep("Epoch 1/12") is True                # signal again
    assert noise.keep("  warnings.warn(") is True          # continues the signal


def test_nothing_is_hidden_by_default():
    """A PyTorch run says nothing that needs hiding, so the filter must not
    invent work: every line abismal writes reaches the panel."""
    for line in SIGNAL:
        assert not is_noise(line), line


def test_the_recorded_log_comes_through_whole():
    """Against the captured log the replay stub plays, not a hand-written sample."""
    recorded = Path(__file__).parent / "replay" / "console.log"
    raw = recorded.read_text()
    kept = strip_noise(raw)

    assert kept == raw
    assert any(line.startswith("Epoch 1/") for line in kept.splitlines())
    assert any("loss:" in line for line in kept.splitlines())
