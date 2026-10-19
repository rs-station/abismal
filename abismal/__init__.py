## Version number for abismal
def getVersionNumber():
    try:
        from importlib.metadata import version, PackageNotFoundError

        return version("abismal")
    except PackageNotFoundError:
        # Running from a source tree that was never installed
        from os.path import dirname, join

        with open(join(dirname(__file__), "VERSION")) as f:
            return f.read().strip()

__version__ = getVersionNumber()

import pandas as pd #Workaround for scipy 1.15.1 (see https://github.com/rs-station/abismal/issues/12)

# Subpackages load on first use. Importing them all here meant `import abismal`
# paid for torch and the rest even in a process that only wanted the CLI
# parser -- and the GUI imports the parser.
_submodules = (
    "callbacks", "distributions", "io", "layers", "likelihood", "merging",
    "optimizers", "scaling", "surrogate_posterior", "symmetry",
)


def __getattr__(name):
    if name in _submodules:
        import importlib

        return importlib.import_module(f"abismal.{name}")
    raise AttributeError(f"module 'abismal' has no attribute {name!r}")
