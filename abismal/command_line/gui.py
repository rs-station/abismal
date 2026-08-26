"""Open the abismal GUI notebook in JupyterLab, here.

    abismal.gui [--notebook NAME] [--fresh] [-- <jupyter lab options>]

Copies the packaged notebook into the current directory if it is not already
there, then starts JupyterLab on it. The point is that there is nothing to
locate: the notebook ships inside the package, so `pip install abismal[gui]`
followed by `cd` to your data and `abismal.gui` is the whole story.

Copying rather than opening the packaged file in place is deliberate. Opening it
where it is installed would root the server in site-packages, put that in the
file browser, and make every save an edit to the installed package -- shared by
every project on the machine, and overwritten by the next `pip install`. A copy
in your working directory is yours to annotate and keep.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_NAME = "abismal_gui.ipynb"


def packaged_notebook():
    """The notebook shipped inside abismal.gui."""
    from importlib.resources import files

    # files('abismal') rather than files('abismal.gui'): importing abismal.gui
    # pulls in ipywidgets and the whole form, none of which is needed to copy a
    # file and exec jupyter.
    return Path(str(files("abismal") / "gui" / DEFAULT_NAME))


def place_notebook(destination, fresh=False):
    """Put a copy of the notebook at `destination`. Returns (path, copied).

    An existing notebook is never overwritten without --fresh: by the second run
    it holds the user's own edits and outputs, and silently replacing those
    because they typed the same command again would be indefensible.
    """
    destination = Path(destination)
    if destination.exists() and not fresh:
        return destination, False
    source = packaged_notebook()
    if not source.is_file():
        raise SystemExit(
            f"the packaged notebook is missing: {source}\n"
            "This build did not include it; install abismal from a wheel or an "
            "editable checkout."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination, True


def find_jupyter():
    """The jupyter belonging to this interpreter, or whatever PATH offers.

    The sibling of sys.executable comes first: this script runs on the
    interpreter abismal was installed into, and its environment's jupyter is the
    one with abismal and ipywidgets importable. PATH may not have that
    environment on it at all -- console scripts work by absolute path -- and if
    it has a different one, that jupyter would start kernels that cannot import
    what the notebook needs.
    """
    sibling = Path(sys.executable).parent / "jupyter"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("jupyter")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="abismal.gui",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--notebook", default=DEFAULT_NAME,
        help=f"name to give the copy in this directory (default: {DEFAULT_NAME})",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="overwrite an existing copy with the packaged one, discarding edits",
    )
    parser.add_argument(
        "--print-path", action="store_true",
        help="print the packaged notebook's location and exit",
    )
    args, jupyter_args = parser.parse_known_args(argv)

    if args.print_path:
        print(packaged_notebook())
        return 0

    notebook, copied = place_notebook(args.notebook, fresh=args.fresh)
    print(f"{'copied' if copied else 'using'} {notebook}")
    if not copied:
        print("  (--fresh to replace it with the packaged copy)")

    jupyter = find_jupyter()
    if jupyter is None:
        raise SystemExit(
            "could not find a jupyter to launch. Install the gui extra:\n"
            '  pip install "abismal[gui]"'
        )

    # Launched from the current directory on purpose: that is what the form
    # resolves its paths against, and it is where the user's data is. The
    # server's root_dir follows the notebook argument instead, which nothing
    # reads any more -- see README, "The notebook GUI".
    command = [jupyter, "lab", str(notebook), *jupyter_args]
    print(f"$ {' '.join(command)}")
    sys.stdout.flush()
    # exec rather than spawn: jupyter owns the terminal from here, and Ctrl-C
    # should reach it rather than a wrapper that would have to forward signals.
    os.execv(jupyter, command)


if __name__ == "__main__":
    sys.exit(main())
