"""Keeping TensorFlow's startup narration out of the GUI's log panel.

Training shares stderr with everything else abismal has to say, so the console
log on disk is a mixture: a dozen lines about cuFFT registration and oneDNN
before anything happens, then the epochs, the warnings and any traceback. On a
terminal that scrolls past. In a fixed log box it is the first thing a user sees
and, on a short run, sometimes the only thing.

Filtered where the log is *displayed* rather than where it is captured. The
runner already reads console.log line by line to drive the progress bar, so this
is a predicate on strings -- no file descriptors, no reader threads, no risk of
losing a line that mattered. The file on disk keeps everything, which is the
point: this hides noise from the panel without deciding what gets recorded.

The markers come from the same survey as the `tf-quiet` branch's stderr filter.
If that ever lands, the two should share this list.
"""

NOISE_MARKERS = (
    "] Created device /job:",
    "] XLA service ",
    "StreamExecutor device (",
    "] Loaded cuDNN version",
    "] Compiled cluster using XLA!",
    "Unable to register cuFFT factory",
    "Unable to register cuDNN factory",
    "Unable to register cuBLAS factory",
    "All log messages before absl::InitializeLog",
    "oneDNN custom operations are on",
    "This TensorFlow binary is optimized",
    "To enable the following instructions:",
    "disabling MLIR crash reproducer",
    # Dataset exhaustion at the end of every epoch, reported as an abort. One of
    # these blocks per epoch, each trailing a couple of graph-node references.
    "Local rendezvous is aborting with status: OUT_OF_RANGE",
    # recv at :424 and send at :428, both per epoch.
    "Local rendezvous recv item cancelled",
    "Local rendezvous send item cancelled",
    # Keras' one-shot advisory that the callback hooks cost more than a batch.
    # It disables itself after reporting once, and it fires for abismal because
    # every callback here is custom -- the per-batch hooks do nothing, so what it
    # measures is Keras' own bookkeeping. This is the only line dropped here that
    # carries any signal; remove it if you ever want to hear about a genuinely
    # expensive per-batch callback.
    "is slow compared to the batch time",
)

# The continuation of "To enable the following instructions:", which arrives as
# its own line and is meaningless without the line above it.
_CONTINUATIONS = ("To enable them in other operations,",)


def is_noise(line):
    """Is this one of TensorFlow's own status lines rather than abismal's output?

    Matched on message text, never on the severity prefix. absl's `I0000` is
    shared with anything else logging through it, so dropping a severity would
    hide messages worth reading -- and abismal's own warnings come through the
    same channel.
    """
    return any(marker in line for marker in NOISE_MARKERS + _CONTINUATIONS)


def _is_graph_frame(line):
    """An indented `[[{{node ...}}]]` reference, which continues the line above.

    TensorFlow trails its status messages with these, and so do its tracebacks.
    Whether one is noise depends entirely on what it is continuing, which is why
    the filter below has to remember.
    """
    stripped = line.strip()
    return stripped.startswith("[[") and line[:1].isspace()


class NoiseFilter:
    """Drops TensorFlow's status lines, and only their continuations.

    Stateful because a graph-node reference is noise under "Local rendezvous is
    aborting", which happens at the end of every epoch, and is the most useful
    line in the file under a genuine error. Deciding per line cannot tell those
    apart: it would either keep a screenful of end-of-epoch frames or throw away
    the one detail that localises a real failure.
    """

    def __init__(self):
        self._dropped_last = False

    def keep(self, line):
        """Whether this line belongs in the log panel."""
        if _is_graph_frame(line):
            return not self._dropped_last
        self._dropped_last = is_noise(line)
        return not self._dropped_last


def strip_noise(text):
    """The same filter over a block of text, for a log captured before this ran."""
    noise = NoiseFilter()
    return "".join(
        line for line in text.splitlines(keepends=True) if noise.keep(line)
    )
