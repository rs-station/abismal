"""Keeping library startup narration out of the GUI's log panel.

Training shares stderr with everything else in the process, so the console log
on disk is a mixture of abismal's epochs, warnings and tracebacks and whatever
the libraries underneath decide to say. On a terminal that scrolls past. In a
fixed log box it is the first thing a user sees and, on a short run, sometimes
the only thing.

Filtered where the log is *displayed* rather than where it is captured. The
runner already reads console.log line by line to drive the progress bar, so this
is a predicate on strings -- no file descriptors, no reader threads, no risk of
losing a line that mattered. The file on disk keeps everything, which is the
point: this hides noise from the panel without deciding what gets recorded.

Nothing needs hiding at the moment. abismal 1.x ran on TensorFlow, which opened
every run with a dozen lines about cuFFT registration and oneDNN and closed every
epoch with a screenful of dataset-exhaustion notices; this module existed for
those. A PyTorch run writes only abismal's own output (checked on .mtz and
CrystFEL .stream inputs), so the list below is empty. The mechanism stays, so a
library that starts narrating again costs one line here rather than a new filter.
"""

#: Substrings that mark a line as a library's status chatter. Matched on message
#: text, never on a severity prefix -- see `is_noise`.
NOISE_MARKERS = ()


def is_noise(line):
    """Is this one of a library's own status lines rather than abismal's output?

    Matched on message text, never on the severity prefix. A logging prefix such
    as absl's `I0000` is shared with anything else logging through it, so
    dropping a severity would hide messages worth reading -- and abismal's own
    warnings come through the same channel.
    """
    return any(marker in line for marker in NOISE_MARKERS)


def _is_continuation(line):
    """An indented line, which continues the one above it.

    Python's warnings print their source line this way, tracebacks indent their
    frames, and TensorFlow trailed its status messages with indented graph-node
    references. Whether one is noise depends entirely on what it is continuing,
    which is why the filter below has to remember.
    """
    return bool(line.strip()) and line[:1].isspace()


class NoiseFilter:
    """Drops noise lines, and with them only their own continuations.

    Stateful because the same indented line can be noise under a status message
    and the most useful line in the file under a genuine error. Deciding per
    line cannot tell those apart: it would either keep orphaned continuations of
    hidden lines or throw away the detail that localises a real failure.
    """

    def __init__(self):
        self._dropped_last = False

    def keep(self, line):
        """Whether this line belongs in the log panel."""
        if _is_continuation(line):
            return not self._dropped_last
        self._dropped_last = is_noise(line)
        return not self._dropped_last


def strip_noise(text):
    """The same filter over a block of text, for a log captured before this ran."""
    noise = NoiseFilter()
    return "".join(
        line for line in text.splitlines(keepends=True) if noise.keep(line)
    )
