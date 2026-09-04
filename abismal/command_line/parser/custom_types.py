import math
from argparse import ArgumentTypeError
from pathlib import Path


def degrees_of_freedom(value):
    """Degrees of freedom for a Student's t error model, or None for a normal one.

    A t distribution converges to a normal as its degrees of freedom go to
    infinity, so every spelling of "infinite" resolves to None -- which is
    already how the CLI selects `NormalLikelihood`. Accepted: 0, `inf` in any
    capitalization, `math.inf`/`np.inf`, the strings "none"/"None", and None
    itself.

    Takes numbers as well as strings so it can be reused on a value that did
    not come from argparse.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("none", ""):
            return None
        try:
            value = float(text)
        except ValueError:
            raise ArgumentTypeError(
                f"invalid degrees of freedom: {value!r}. Give a positive number, "
                "or one of 0, inf, none for a normal error model."
            )
    value = float(value)
    if value == 0.0 or math.isinf(value):
        return None
    if math.isnan(value) or value < 0.0:
        raise ArgumentTypeError(
            f"degrees of freedom must be positive, got {value!r}. Use 0, inf or "
            "none to ask for a normal error model."
        )
    return value

def list_of_ints(string):
    return [int(i) for i in string.split(',')]

def list_of_floats(string):
    return [float(i) for i in string.split(',')]

def list_of_ops(string):
    return string.split(';')


# Path-valued argument types. The notebook GUI dispatches on these by identity to
# decide which file picker to render (see abismal.gui.components.argparse_gui), so
# an argument naming something on disk should carry one of them rather than `str`.
# Plain `pathlib.Path` covers the single-file case and needs nothing defined here.

def list_of_paths(string):
    """A comma-separated list of paths, as --eff-files and --torchref-pdb accept."""
    return [Path(p) for p in string.split(',')]


def directory(string):
    """A path naming a directory rather than a file."""
    return Path(string)
