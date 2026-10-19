"""Whether the DIALS stack is importable, and what to say when it is not.

`.expt` and `.refl` files are read through `dxtbx` and `dials`, which are the
one dependency abismal cannot declare. Neither is installable from PyPI in any
usable form -- `dxtbx` there is a 2020 source-only 3.0.4.1, and `dials` is a
0.0.1 placeholder -- so there is no `[dials]` extra to write. Declaring one
would make `pip install "abismal[dials]"` appear to succeed while changing
nothing, which is worse than not offering it.

conda-forge has both, current and built for every Python abismal supports. So
the dependency is real, optional, and satisfied by a different package manager:
exactly the thing a preflight is for. Nobody merging `.mtz` or CrystFEL streams
pays anything for it.
"""
from importlib.util import find_spec

INSTALL_HINT = """\
Reading .expt/.refl files needs the DIALS toolbox (dxtbx and dials), which is
not installable from PyPI -- what is published there under those names is a
stale placeholder. Install it from conda-forge instead:

    conda install -c conda-forge dxtbx dials

or, into an environment you already have:

    micromamba install -c conda-forge dxtbx dials

On a machine with only pip -- Colab, say -- abismal can build a prefix beside
this interpreter instead:

    from abismal.io import install_dials_prefix
    install_dials_prefix()

Point ABISMAL_DIALS_PREFIX at an existing one to reuse it.

Merging .mtz files or CrystFEL .stream files needs none of this."""


def dials_is_available():
    """Can the DIALS readers be imported?

    A configured prefix is activated first, so a stack installed beside this
    interpreter counts as available -- that is the whole point of having one.
    Activation is cheap and idempotent: it appends two paths and sets two
    variables, and imports nothing.

    find_spec rather than an import: the stack is heavy and pulls a boost
    extension tree, and the question here is only whether it is present.
    """
    if all(find_spec(name) is not None for name in ("dxtbx", "dials")):
        return True

    from abismal.io._dials_env import activate

    if activate():
        # find_spec consults sys.path, but a path appended after an earlier
        # failed lookup is only seen once the finders are told to look again.
        import importlib

        importlib.invalidate_caches()
        return all(find_spec(name) is not None for name in ("dxtbx", "dials"))
    return False


def require_dials(files=()):
    """Raise a legible error if the DIALS readers are missing.

    Called before the loader rather than inside it. Without this the first sign
    of trouble is a ModuleNotFoundError raised several frames deep, after the
    data manager has already started work -- which is how it surfaced on Colab:
    a traceback ending in `No module named 'dxtbx'` with no indication that the
    fix is a conda install, or that only these file types need it.
    """
    if dials_is_available():
        return

    listed = ""
    if files:
        shown = list(files)[:3]
        listed = "\n\nThe inputs that need it:\n" + "\n".join(
            f"    {f}" for f in shown
        )
        if len(files) > len(shown):
            listed += f"\n    ... and {len(files) - len(shown)} more"

    raise SystemExit(INSTALL_HINT + listed)
