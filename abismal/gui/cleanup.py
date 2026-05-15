import os
import glob
import shutil


# Patterns that abismal (or its GUI wrapper) is known to create under out_dir.
# Anything not matching these is left alone — better to be conservative and
# leave stray user files than to delete them.
_FILE_PATTERNS = (
    'abismal.log',
    'console.log',
    'abismal.pid',
    'datamanager.yml',
    'history.csv',
    'epoch_*.keras',
    'asu_*.mtz',
)
_DIR_PATTERNS = (
    'eff_*',
    'diffmaps_*',
)


def find_abismal_outputs(out_dir):
    """Return sorted list of paths under out_dir matching known abismal outputs."""
    if not os.path.isdir(out_dir):
        return []
    found = set()
    for pat in _FILE_PATTERNS:
        for p in glob.glob(os.path.join(out_dir, pat)):
            if os.path.isfile(p):
                found.add(p)
    for pat in _DIR_PATTERNS:
        for p in glob.glob(os.path.join(out_dir, pat)):
            if os.path.isdir(p):
                found.add(p)
    return sorted(found)


def cleanup_abismal_outputs(out_dir):
    """Remove known abismal outputs from out_dir. Returns list of removed paths."""
    removed = []
    for path in find_abismal_outputs(out_dir):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed.append(path)
        except OSError:
            pass
    return removed
