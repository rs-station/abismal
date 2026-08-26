from pathlib import Path


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
