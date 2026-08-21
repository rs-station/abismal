# Releasing abismal

## Build from a clean clone, not from your working tree

```bash
git clone --branch <release-branch> . /tmp/abismal-release
cd /tmp/abismal-release
python -m build --wheel
```

setuptools packages a directory by **what is on disk**, not by what git tracks.
`[tool.setuptools.packages.find] include = ["abismal*"]` selects the package, and
then every `.py` inside it is shipped -- including untracked scratch files.

This is not hypothetical. Building 0.0.8 in a working tree that had five
uncommitted experimental modules produced a wheel containing all of them:

```
abismal/bg_scaler.py
abismal/test.py
abismal/scaling/{bg,bup,scaling_with_amortized_gb}.py
```

`git rm --cached` does not help -- the files are still on disk. A clean clone
of the release branch is the only thing that guarantees the artifact matches the
commit.

## Check the artifact before publishing

```bash
python - <<'PY'
import zipfile, glob
w = sorted(glob.glob('dist/*.whl'))[-1]
names = zipfile.ZipFile(w).namelist()
print(w)
print('unexpected:', [n for n in names
                      if n.startswith(('notes', 'jobs', 'build'))
                      or 'gui' in n.lower()])
PY
```

The GUI lives on the `gui` branch and must not appear in a library release.

## Version

`abismal/VERSION` is the single source (`[tool.setuptools.dynamic]`). Bump it,
commit, then tag to match -- existing tags are `v0.0.2` .. `v0.0.8`.

## Before tagging

- `pytest tests/` from a clean clone. Failures for `dxtbx` are expected when
  DIALS is not installed; anything else is not.
- Confirm the wheel's `Version:` matches `abismal/VERSION`.
