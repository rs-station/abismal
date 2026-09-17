"""What the log panel shows, and what it hides.

TensorFlow narrates its startup and its per-epoch dataset exhaustion through the
same stderr abismal reports on. In a terminal that scrolls past; in a fixed log
box it is what a user sees first, and on a short run sometimes all they see.
"""
from pathlib import Path

from abismal.gui._log_filter import NoiseFilter, is_noise, strip_noise


NOISE = [
    "2026-05-15 14:17:48.970679: E external/local_xla/xla/stream_executor/cuda/"
    "cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register",
    "WARNING: All log messages before absl::InitializeLog() is called are written"
    " to STDERR",
    "I0000 00:00:1778869071.448782  603023 gpu_device.cc:2022] Created device "
    "/job:localhost/replica:0/task:0/device:GPU:0 with 46732 MB memory",
    "2026-05-15 14:17:48.995392: I tensorflow/core/platform/cpu_feature_guard.cc:210]"
    " This TensorFlow binary is optimized to use available CPU instructions",
    "I0000 00:00:1778869082.207106 device_compiler.h:188] Compiled cluster using XLA!",
]

SIGNAL = [
    "Epoch 1/12",
    "1000/1000 - 187s - loss: 10.9483 - NLL: 5.3453 - KL: 2.1794",
    "Freezing intensity standardization layer...",
    "RuntimeWarning: skipping torchref at epoch 5: 1 earlier run(s) still going",
    "Traceback (most recent call last):",
    "ValueError: something abismal actually said",
]


def test_tensorflow_narration_is_dropped():
    for line in NOISE:
        assert is_noise(line), line


def test_abismal_output_and_warnings_survive():
    """Matched on message text, never on severity.

    absl's I0000 prefix is shared with anything logging through it, so dropping a
    severity would take abismal's own warnings with it.
    """
    for line in SIGNAL:
        assert not is_noise(line), line


def test_a_graph_frame_under_noise_goes_with_it():
    """End-of-epoch exhaustion trails `[[{{node ...}}]]` lines, once per epoch.

    Dropping the header and keeping its continuations is worse than not
    filtering: the panel fills with orphaned brackets.
    """
    noise = NoiseFilter()
    header = ("2026-05-15 14:18:27.354855: I tensorflow/core/framework/"
              "local_rendezvous.cc:405] Local rendezvous is aborting with "
              "status: OUT_OF_RANGE: End of sequence")
    assert noise.keep(header) is False
    assert noise.keep("\t [[{{node IteratorGetNext}}]]") is False
    assert noise.keep("\t [[Func/variational_merging_model/image_scaler/_11]]") is False


def test_a_graph_frame_under_a_real_error_is_kept():
    """The same line is the most useful detail in a genuine traceback.

    This is the whole reason the filter is stateful -- deciding per line would
    either keep a screenful of end-of-epoch frames or throw away the one detail
    that localises a real failure.
    """
    noise = NoiseFilter()
    assert noise.keep("ValueError: Incompatible shapes in RaggedConcat") is True
    assert noise.keep("\t [[{{node IteratorGetNext}}]]") is True


def test_the_filter_resets_between_blocks():
    noise = NoiseFilter()
    noise.keep("oneDNN custom operations are on")          # noise
    assert noise.keep("Epoch 1/12") is True                # signal again
    assert noise.keep("\t [[{{node Foo}}]]") is True       # continues the signal


def test_the_recorded_log_comes_out_readable():
    """Against the real captured log, not a hand-written sample."""
    recorded = Path(__file__).parent / "replay" / "console.log"
    raw = recorded.read_text()
    kept = strip_noise(raw).splitlines()

    assert len(kept) < len(raw.splitlines()) / 2, "barely filtered anything"
    assert any(line.startswith("Epoch 1/") for line in kept), "lost the epochs"
    assert any("loss:" in line for line in kept), "lost the losses"
    assert not any("cuFFT" in line for line in kept)
    assert not any("IteratorGetNext" in line for line in kept)
    # Whatever survives, nothing should be an orphaned continuation.
    assert not any(line.strip().startswith("[[") for line in kept)
