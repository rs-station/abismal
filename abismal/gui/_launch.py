"""One call that starts the GUI, and the setup a notebook wants around it.

This exists so the notebook does not have to. Everything here used to live in a
cell: forty-odd lines of install logic, downloads and version checks that every
user met before they met the GUI. Worse than ugly, it was untestable -- Python
embedded in notebook JSON can only be checked by matching strings against its
own source, and guards written that way have already passed while the code was
wrong, and failed by matching their own comments.

As package code it is ordinary: importable, testable, and reviewable.
"""
import os
import subprocess
import sys

from abismal.gui._build import describe_build

# The lysozyme SSAD dataset the abismal benchmarks use: unmerged, anomalous, good
# to 1.7 A. Fetched from the same place data/hewl/download.sh gets it.
EXAMPLE_URL = (
    "https://github.com/rs-station/careless-examples/raw/refs/heads/main/"
    "hewl_ssad/unmerged.mtz"
)
EXAMPLE_PATH = "/content/hewl_unmerged.mtz"

# Its reference data -- the refined model torchref needs, the r-free flags, the
# phenix .eff files. Listed through the API so it tracks the benchmark repo.
REFERENCE_API = (
    "https://api.github.com/repos/rs-station/abismal-benchmarks/contents/"
    "data/hewl/reference_data?ref=main"
)
REFERENCE_DIR = "/content/hewl_reference"


def on_colab():
    """Whether /content exists, which is Colab and is where its file pickers open.

    The directory rather than the google.colab import: what these helpers care
    about is having somewhere to put files that the GUI will look in.
    """
    return os.path.isdir("/content")


def install_torchref():
    """Add torchref where torch already exists, which is cheap and makes refinement work.

    Deliberately not in the `gui` extra: that pulls ~800 MB of torch for everyone
    who only merges. Colab ships torch already and torchref on top is 3.5 MB, so
    where torch is present the trade reverses.

    Without it, --torchref-pdb fails inside a detached worker whose stderr nobody
    reads -- the run trains to completion and silently refines nothing.
    """
    from importlib.util import find_spec

    if find_spec("torch") is None:
        return False
    if find_spec("torchref") is not None:
        return True

    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "torchref>=0.6.0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if done.returncode != 0:
        print(done.stdout)
    return done.returncode == 0


def fetch_example_data(quiet=False):
    """Put the lysozyme dataset where the file picker opens. Best effort."""
    import urllib.request

    if not on_colab():
        return None
    if os.path.exists(EXAMPLE_PATH):
        return EXAMPLE_PATH

    def progress(blocks, block_size, total):
        # 28 MB of silence is indistinguishable from a hang.
        if not quiet and total > 0 and blocks % 200 == 0:
            done = min(blocks * block_size, total)
            print(f"  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", flush=True)

    if not quiet:
        print(f"fetching example data -> {EXAMPLE_PATH}", flush=True)
    try:
        urllib.request.urlretrieve(EXAMPLE_URL, EXAMPLE_PATH, reporthook=progress)
    except Exception as error:
        print(f"could not fetch the example data ({error}); use your own files")
        return None
    return EXAMPLE_PATH


def fetch_reference_data(quiet=False):
    """Mirror hewl's reference_data next to the reflections. Best effort."""
    import json
    import urllib.request

    if not on_colab():
        return None
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    try:
        with urllib.request.urlopen(REFERENCE_API, timeout=30) as response:
            listing = json.load(response)
    except Exception as error:
        # The unauthenticated API allows 60 requests an hour per address, and
        # Colab shares addresses. Not fatal: merging needs only the reflections.
        print(f"could not list the reference data ({error})")
        return None

    for entry in listing:
        if entry.get("type") != "file" or not entry.get("download_url"):
            continue
        destination = os.path.join(REFERENCE_DIR, entry["name"])
        # Size as well as existence, so a half-written file from an interrupted
        # session is replaced rather than trusted.
        if (os.path.exists(destination)
                and os.path.getsize(destination) == entry.get("size")):
            continue
        try:
            urllib.request.urlretrieve(entry["download_url"], destination)
        except Exception as error:
            print(f"  {entry['name']}: {error}")
    return REFERENCE_DIR


def example_data_summary():
    """The paths worth knowing once the example data is in place."""
    return "\n".join([
        f"example data     : {EXAMPLE_PATH}  (anomalous, try dmin 1.7)",
        f"--torchref-pdb   : {REFERENCE_DIR}/RTSAD_HEWL_refine_25.pdb",
        f"--r-free-mtz     : {REFERENCE_DIR}/r-free-flags.mtz",
        f"--eff-files      : {REFERENCE_DIR}/refine.eff",
        "--torchref-wavelength 1.892   (the benchmark .eff carries no wavelength)",
    ])


def no_output_scroll():
    """Let this cell's output grow instead of scrolling inside a fixed box.

    The run panel is tall -- the 3D viewer alone is 600px, and the log sits under
    it -- so Colab caps the output and makes it scrollable. You then land part of
    the way down it, with the progress bar and the training history above the
    fold and no hint they are there. That reads as widgets failing to render, and
    cost a couple of debugging rounds before it turned out to be a scrollbar.

    Colab's own API for this; a no-op anywhere else, and best-effort on Colab
    since a cosmetic call should never be what breaks a launch. Returns whether
    it applied, which is mostly for the tests.
    """
    try:
        from google.colab import output
    except ImportError:
        return False
    try:
        output.no_vertical_scroll()
        return True
    except Exception:
        return False


def stale_modules(names=("ipywidgets", "tensorflow", "numpy")):
    """Loaded modules whose version differs from the one now on disk.

    An install cannot replace a module the interpreter already imported: the old
    object stays in sys.modules and the new files sit unused, so the session
    keeps running code that is no longer installed. ipywidgets is the one that
    bites -- a widget stack half a major version out renders nothing.

    Comparing loaded against installed detects that directly. The earlier version
    of this diffed package versions either side of the install, which needed the
    install to be in the same cell and could only see changes it made itself.
    """
    import importlib.metadata as md

    stale = []
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            continue
        loaded = getattr(module, "__version__", None)
        try:
            on_disk = md.version(name)
        except Exception:
            continue
        if loaded and on_disk and loaded != on_disk:
            stale.append((name, loaded, on_disk))
    return stale


def launch(example_data=False, branch=None, quiet=False):
    """Build the GUI, and return the widget for the notebook to display.

    `example_data=True` also fetches the lysozyme dataset and its reference
    files, and adds torchref where torch already exists -- the reference data is
    there to drive refinement, so it would be no use without it. Off by default:
    someone who brought their own data should not be made to download 31 MB.

    `branch` is only for a deployed debug build, where the install is pinned to a
    commit and there is no branch name recorded to check freshness against.
    """
    stale = stale_modules()
    if stale:
        changed = ", ".join(f"{n} {was} -> {now}" for n, was, now in stale)
        raise RuntimeError(
            f"these are loaded at a different version than is installed: {changed}.\n"
            "The interpreter cannot swap a module it has already imported, so the "
            "session is running code that is no longer on disk.\n"
            "Restart the runtime (Runtime > Restart session), then run this cell "
            "again."
        )

    # Before anything is displayed, so the cell is sized for what follows.
    no_output_scroll()

    if not quiet:
        print(describe_build(branch=branch))

    if example_data:
        install_torchref()
        if fetch_example_data(quiet=quiet):
            fetch_reference_data(quiet=quiet)
            if not quiet:
                print(example_data_summary())

    from abismal.gui.components import ArgparseGUI

    return ArgparseGUI().to_widget()
