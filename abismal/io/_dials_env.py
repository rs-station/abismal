"""Installing and activating the DIALS toolbox beside an existing interpreter.

dxtbx and dials are not installable from PyPI (see _dials_check), so on a
machine that only has pip -- Google Colab, most notably -- they have to come
from conda-forge. That does not require conda: micromamba is a standalone
static binary, so a prefix can be created without touching the interpreter that
will use it.

The prefix is then *appended* to sys.path, never prepended. PYTHONPATH and
prepending put the conda tree ahead of everything, which is not what we want:
the prefix carries its own numpy, ml_dtypes, h5py and scipy, and shadowing the
host's broke TensorFlow outright -- conda's older ml_dtypes made jax raise on
import, which is the same failure as the old `tensorflow==2.18` pin. Appended,
the prefix supplies only what the host genuinely lacks, which is dxtbx and
dials and their private dependencies.

One version does have to line up. dxtbx ships compiled extensions built against
a particular numpy, and numpy's ABI is forward- but not backward-compatible, so
the host's numpy must be at least the prefix's. The solve is constrained to
`numpy<2.4` for that reason: it lands on numpy 2.3.x, which TensorFlow accepts
(>=1.26, no ceiling) and which stays under torchref's `numpy<2.4.0` -- so the
resulting stack holds together with refinement still working. Left
unconstrained, conda picks numpy 2.5 and torchref becomes uninstallable.
"""
import os
import subprocess
import sys
from pathlib import Path

MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"

# Keeps torchref installable; see the module docstring.
NUMPY_CEILING = "numpy<2.4"

PACKAGES = ("dxtbx", "dials")

#: Where the prefix lives unless told otherwise. /content is Colab's disk.
DEFAULT_PREFIX = "/content/dials-env" if os.path.isdir("/content") else None

#: Overrides everything, for a prefix made by hand or by a cluster module.
PREFIX_ENV_VAR = "ABISMAL_DIALS_PREFIX"


def configured_prefix():
    """The prefix to use, or None if there is nothing to look at."""
    from_env = os.environ.get(PREFIX_ENV_VAR)
    if from_env:
        return Path(from_env)
    if DEFAULT_PREFIX and os.path.isdir(DEFAULT_PREFIX):
        return Path(DEFAULT_PREFIX)
    return None


def _site_packages(prefix):
    """The two directories a conda prefix keeps importable code in.

    lib-dynload as well as site-packages: the compiled boost extensions live
    there, and without it `import dials` fails on `boost_python_meta_ext` long
    after dxtbx itself has imported cleanly.
    """
    version = "%d.%d" % sys.version_info[:2]
    lib = Path(prefix) / "lib" / f"python{version}"
    return lib / "site-packages", lib / "lib-dynload"


def activate(prefix=None):
    """Make a prefix usable, by this process and by anything it launches.

    Appends rather than prepends, and sets what libtbx and the dynamic linker
    need. Returns True if the paths were added.

    Two audiences, and the second is the important one. This process gets the
    import paths and a best-effort preload of the prefix's C++ runtime, which
    works if nothing has loaded the system one yet. Subprocesses get
    LD_LIBRARY_PATH in their environment, which the linker reads at startup and
    which therefore always works -- and since the merge itself runs as a
    subprocess, that is the path that carries the real workload.
    """
    prefix = Path(prefix) if prefix else configured_prefix()
    if prefix is None or not prefix.is_dir():
        return False

    site_packages, dynload = _site_packages(prefix)
    if not site_packages.is_dir():
        return False

    for path in (site_packages, dynload):
        entry = str(path)
        if entry not in sys.path:
            sys.path.append(entry)

    # libtbx unpickles a build path at import and looks for it relative to
    # sys.prefix, which is the host interpreter and therefore wrong. Without
    # this it fails hunting for /usr/share/cctbx.
    os.environ.setdefault("LIBTBX_BUILD", str(Path(prefix) / "share" / "cctbx"))

    _preload_runtime(prefix)

    # For anything this process launches. The linker reads LD_LIBRARY_PATH when
    # a process starts, so setting it here does nothing for *us* -- that is what
    # the preload above is for -- but a child inherits it, which is how the
    # merge subprocess the GUI spawns picks up the same libraries.
    lib = str(Path(prefix) / "lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in existing.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = f"{existing}{os.pathsep}{lib}" if existing else lib

    return True


def _preload_runtime(prefix):
    """Load the prefix's C++ runtime before anything needs a symbol from it.

    conda builds against a newer toolchain than an old host provides, and the
    compiled extensions want a CXXABI the system libstdc++ does not export --
    `version CXXABI_1.3.15 not found` from boost_python_meta_ext, several
    imports after dxtbx appeared to load cleanly.

    Setting LD_LIBRARY_PATH cannot fix this from inside a running process: the
    linker reads it at startup. Loading the library explicitly with RTLD_GLOBAL
    does, because the symbols are then already resolved when the extension is
    opened.

    Only effective if nothing has already loaded the system libstdc++. Once
    TensorFlow is imported it has, and the preload cannot displace symbols that
    are already resolved -- so this helps a clean process and not a live kernel
    that has been training. That is why `activate` also exports
    LD_LIBRARY_PATH: the merge runs in a subprocess, which starts fresh and
    reads the variable the way the linker expects. The in-process path is a
    convenience; the subprocess path is the one that has to work.

    Best effort. Where the host is new enough there is nothing to fix, and a
    failure here should surface as the import error it causes rather than here.
    """
    import ctypes

    for name in ("libstdc++.so.6", "libgcc_s.so.1"):
        candidate = Path(prefix) / "lib" / name
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def prefix_numpy_version(prefix):
    """The numpy the prefix was built against, read from its own metadata."""
    site_packages, _ = _site_packages(prefix)
    for entry in site_packages.glob("numpy-*.dist-info"):
        return entry.name.split("-")[1]
    return None


def _micromamba(workdir):
    """Fetch the standalone micromamba binary. No conda, no Python needed."""
    import urllib.request

    binary = Path(workdir) / "bin" / "micromamba"
    if binary.is_file():
        return binary

    Path(workdir).mkdir(parents=True, exist_ok=True)
    archive = Path(workdir) / "micromamba.tar.bz2"
    urllib.request.urlretrieve(MICROMAMBA_URL, archive)
    # -f matters: without it tar reads stdin, which is how this first failed --
    # the shell form this was translated from piped curl into tar, and the file
    # download needs to be named explicitly.
    subprocess.run(
        ["tar", "-xj", "-f", str(archive), "-C", str(workdir), "bin/micromamba"],
        check=True,
    )
    archive.unlink(missing_ok=True)
    binary.chmod(0o755)
    return binary


def install(prefix=None, workdir="/tmp/abismal-micromamba", quiet=False):
    """Create a prefix holding dxtbx and dials, and make it importable.

    Returns the prefix, or None if it could not be built. Several GB and
    several minutes; the caller decides whether that is wanted.
    """
    prefix = Path(prefix or DEFAULT_PREFIX or "./dials-env")

    if activate(prefix):
        if not quiet:
            print(f"dials prefix already present at {prefix}")
        return prefix

    if not quiet:
        print(
            f"creating a dials prefix at {prefix}\n"
            "  dxtbx and dials are not on PyPI, so this comes from conda-forge.\n"
            "  Several GB, a few minutes, once per runtime.",
            flush=True,
        )

    try:
        binary = _micromamba(workdir)
    except Exception as error:
        print(f"could not fetch micromamba ({error})")
        return None

    version = "%d.%d" % sys.version_info[:2]
    command = [
        str(binary), "create", "--yes", "--quiet",
        "--prefix", str(prefix),
        "-c", "conda-forge",
        f"python={version}", *PACKAGES, NUMPY_CEILING,
    ]
    environment = {**os.environ, "MAMBA_ROOT_PREFIX": str(Path(workdir) / "root")}
    done = subprocess.run(command, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:])
        print("creating the dials prefix failed")
        return None

    if not activate(prefix):
        print(f"the prefix at {prefix} was created but could not be activated")
        return None
    return prefix
