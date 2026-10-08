#!/usr/bin/env python
"""Publish a debug build of the GUI notebook and print the Colab URL for it.

For iterating on work that is not committed yet. The committed notebooks are
served straight off their own branch --

    https://colab.research.google.com/github/rs-station/abismal/blob/gui/abismal/gui/abismal_gui.ipynb
    https://colab.research.google.com/github/rs-station/abismal/blob/gui/devtools/colab_minimal.ipynb

-- and those are what a user opens, so those are what should normally be
tested: a deployed copy is rewritten (pinned to a commit, diagnostics cell
appended) and therefore is not quite the notebook that ships.

What this adds is a build from the *working tree*, without committing: useful
when a fix needs a Colab tab before it deserves a commit, and it keeps a
debugging session from putting thirty commits on a real branch.

Colab opens a notebook straight out of a public GitHub repo:

    https://colab.research.google.com/github/<owner>/<repo>/blob/<ref>/<path>.ipynb

so there is nothing to host and nothing to upload -- "deploying" is pushing a
commit, and the same URL always serves whatever is newest on that ref. This
script builds that commit from the current working tree and force-pushes it to a
scratch branch, so a debug cycle is: run this, then reload the same bookmark.

Two things it is careful about.

**It never touches your index or working tree.** This repo always holds
uncommitted WIP -- untracked scratch files live inside `abismal/` itself -- and
sweeping that into a commit has happened here before. So the commit is assembled
in a scratch index via GIT_INDEX_FILE, and the only staging command is
`git add -u`, which updates *tracked* files and cannot pick up an untracked one.
Run `git status` before and after: it will be identical.

**The notebook and the package it installs are the same revision.** The commit is
built in two steps: C1 holds the code, and C2 holds C1 plus a notebook rewritten
to install `@C1` exactly. Pinning to a sha rather than a branch name is what makes
a reload deterministic -- otherwise pip or the browser can serve something stale
and a cached install is indistinguishable from a bug.

C2 also carries a diagnostics cell that prints the environment as JSON, so one
copy-paste comes back instead of a description.

    python devtools/colab_deploy.py            # build, push, print the URL
    python devtools/colab_deploy.py --dry-run  # build and inspect, push nothing
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = "abismal/gui/abismal_gui.ipynb"
PROBE = "devtools/collect_colab_debug.py"
DEFAULT_BRANCH = "colab-debug"

# Branches this must never write to. The scratch branch is force-pushed every
# cycle, which is only safe because nothing else lives there.
PROTECTED = {"main", "master", "dev", "gui"}


def git(*args, **kwargs):
    """Run git in the repo, returning stdout. `env` extras are merged, not replaced."""
    env = {**os.environ, **kwargs.pop("env_extra", {})}
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, env=env, **kwargs,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"git {' '.join(args)} failed ({result.returncode}):\n{result.stderr}"
        )
    return result.stdout.strip()


def repo_slug():
    """`owner/repo`, from whichever URL form the remote uses."""
    url = git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot read a github owner/repo out of {url!r}")
    return match.group("slug")


def build_tree(index_file):
    """A tree of HEAD plus every tracked modification in the working tree.

    `git add -u` is the whole safety story: it stages changes to files git already
    knows about and has no way to reach an untracked one, so scratch files sitting
    in the source tree cannot ride along.
    """
    env_extra = {"GIT_INDEX_FILE": index_file}
    git("read-tree", "HEAD", env_extra=env_extra)
    git("add", "-u", env_extra=env_extra)

    # Plus anything staged in the real index. `add -u` works against a scratch
    # index seeded from HEAD, so a file that is new -- even one just `git add`ed
    # -- is invisible to it, and the package ships unable to import itself.
    # Staging is the deliberate act that separates a new source file from the
    # scratch files sitting untracked in this tree; honour it, and nothing else.
    staged = [
        path for path in
        git("diff", "--cached", "--name-only", "HEAD").splitlines()
        if path and (REPO / path).is_file()
    ]
    if staged:
        git("add", "--", *staged, env_extra=env_extra)

    return git("write-tree", env_extra=env_extra), env_extra


def debug_notebook(source_text, slug, code_sha, notebook_path, branch):
    """The notebook as deployed: pinned install, plus a diagnostics cell.

    Returns the new notebook JSON as text. The committed notebook is untouched --
    this edits a copy on its way into the commit.
    """
    notebook = json.loads(source_text)

    pinned = f"git+https://github.com/{slug}@{code_sha}"
    rewritten = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "git+https://github.com/" not in source:
            continue
        source = re.sub(
            r"git\+https://github\.com/[^'\"\s]+", pinned, source
        )
        cell["source"] = source.splitlines(keepends=True)
        rewritten += 1

    if not rewritten:
        raise SystemExit(
            f"no cell in {notebook_path} references a git+https:// spec to pin -- "
            "the install cell must have changed shape"
        )

    # Debug builds want the example data and a branch to check freshness
    # against; the shipped notebook should carry neither. Rewriting the call
    # here keeps `launch()` in the committed notebook as the thing a user reads.
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if "\nlaunch()" not in source:
            continue
        cell["source"] = source.replace(
            "\nlaunch()",
            f"\nlaunch(example_data=True, branch={branch!r})",
        ).splitlines(keepends=True)

    probe_url = f"https://raw.githubusercontent.com/{slug}/{code_sha}/{PROBE}"
    diagnostics = f'''# Diagnostics -- added by devtools/colab_deploy.py, not in the committed notebook.
# Run this and paste the output back. It reports the environment as JSON: which
# packages the install actually changed, where the file pickers resolve to, and
# whether torch was already loaded when the install ran.
import json, urllib.request

_src = urllib.request.urlopen("{probe_url}").read()
_ns = {{}}
exec(_src, _ns)
print(json.dumps(_ns["probe"](), indent=2, sort_keys=True, default=str))'''

    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": diagnostics.splitlines(keepends=True),
    })
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def missing_modules(sha):
    """Modules the committed tree imports from itself but does not contain.

    `git add -u` stages tracked files only -- that is what stops untracked WIP
    from being swept into a deploy -- but it also means a genuinely new source
    file silently does not ship, and the package arrives on Colab unable to
    import itself. That happened: components/_compat.py was new, was left out,
    and the notebook died on ModuleNotFoundError after a full install.

    So: read every abismal module in the built tree, resolve its intra-package
    imports back to paths, and report any that are not in the tree.
    """
    import ast

    files = [
        f for f in git("ls-tree", "-r", "--name-only", sha).splitlines()
        if f.startswith("abismal/") and f.endswith(".py")
    ]
    present = set(files)
    missing = set()
    for path in files:
        try:
            tree = ast.parse(git("show", f"{sha}:{path}"))
        except SyntaxError:
            continue
        wanted = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                wanted.add(node.module)
            elif isinstance(node, ast.Import):
                wanted.update(alias.name for alias in node.names)
        for name in wanted:
            if not name.startswith("abismal."):
                continue
            base = name.replace(".", "/")
            # A module, a regular package, or a namespace package -- abismal
            # has one of the last (abismal/ragged/ carries no __init__.py), so
            # requiring __init__ would call a module that ships fine missing.
            if (
                f"{base}.py" in present
                or f"{base}/__init__.py" in present
                or any(f.startswith(f"{base}/") for f in present)
            ):
                continue
            missing.add((path, name))
    return sorted(missing)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                        help=f"scratch branch to force-push (default: {DEFAULT_BRANCH})")
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK,
                        help="repo-relative notebook to publish")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the commits and report, but push nothing")
    args = parser.parse_args(argv)

    if args.branch in PROTECTED:
        raise SystemExit(
            f"refusing to deploy to {args.branch!r}: this branch is force-pushed "
            f"every cycle. Protected: {', '.join(sorted(PROTECTED))}"
        )

    slug = repo_slug()
    before = git("status", "--porcelain")

    with tempfile.TemporaryDirectory() as scratch:
        index_file = str(Path(scratch) / "index")

        # C1 -- the code, exactly as the working tree has it.
        tree, env_extra = build_tree(index_file)
        head = git("rev-parse", "HEAD")
        code_sha = git(
            "commit-tree", tree, "-p", head,
            "-m", f"colab debug build (code) from {head[:12]}",
        )

        # C2 -- the same tree with the notebook rewritten to install C1.
        source_text = (REPO / args.notebook).read_text()
        deployed = debug_notebook(source_text, slug, code_sha, args.notebook,
                                  args.branch)
        blob = subprocess.run(
            ["git", "-C", str(REPO), "hash-object", "-w", "--stdin"],
            input=deployed, capture_output=True, text=True, check=True,
        ).stdout.strip()
        git("update-index", "--cacheinfo", f"100644,{blob},{args.notebook}",
            env_extra=env_extra)
        notebook_tree = git("write-tree", env_extra=env_extra)
        deploy_sha = git(
            "commit-tree", notebook_tree, "-p", code_sha,
            "-m", f"colab debug build (notebook pinned to {code_sha[:12]})",
        )

    after = git("status", "--porcelain")
    if before != after:
        raise SystemExit(
            "the working tree changed during the build -- refusing to push.\n"
            "This script must be a no-op on your checkout; please report it."
        )

    absent = missing_modules(deploy_sha)
    if absent:
        lines = "\n".join(f"    {who} imports {what}" for who, what in absent)
        raise SystemExit(
            "the build is missing modules it imports:\n" + lines
            + "\n\nMost likely a new file that is still untracked -- only tracked\n"
            "files are staged, deliberately, so untracked WIP cannot ride along.\n"
            "Add the ones that belong in the package by explicit path:\n"
            "    git add <file>"
        )

    files = git("ls-tree", "-r", "--name-only", deploy_sha).splitlines()
    url = (
        f"https://colab.research.google.com/github/{slug}/blob/"
        f"{args.branch}/{args.notebook}"
    )

    print(f"  code commit     {code_sha}")
    print(f"  deploy commit   {deploy_sha}")
    print(f"  files in tree   {len(files)}")
    print(f"  working tree    unchanged ({len(before.splitlines())} entries, as before)")

    if args.dry_run:
        print("\n  --dry-run: nothing pushed. Would push to "
              f"{args.branch} and print:\n\n     {url}\n")
        return 0

    git("push", "--force", "origin", f"{deploy_sha}:refs/heads/{args.branch}")

    print(
        "\n" + "=" * 70
        + "\nDeployed. Open this, every time -- the URL never changes:\n"
        f"\n     {url}\n"
        "\nThen, and this part matters:"
        "\n  Runtime -> Disconnect and delete runtime     (the install mutates a"
        "\n                                                live kernel, so a reused"
        "\n                                                runtime is not a clean test)"
        "\n  Runtime -> Run all"
        "\n\nThe last cell prints the environment as JSON. Paste that back.\n"
        + "=" * 70 + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
