"""Which commit is actually running, and whether it is the one you meant.

Two ways to end up looking at code you are not running, both of which have
happened here:

  * the notebook installs only when abismal is *absent*, so a session that
    already has one keeps whatever it had -- while still printing the commit it
    meant to install;
  * GitHub serves a cached copy of a branch file for a few minutes, so a reload
    right after a push can hand you the previous notebook.

The first is a stale *install*, the second a stale *notebook*, and they want
opposite fixes -- delete the runtime, or reload the page. Telling them apart
needs three facts: what is installed, what the notebook asked for, and what the
branch actually points at.
"""
import json
import urllib.request

REPO = "rs-station/abismal"


def installed_commit():
    """The commit pip installed, or None if abismal did not come from git.

    pip records this in direct_url.json (PEP 610) for any VCS install, which is
    the only reliable answer -- the version number does not move between
    commits, so it cannot distinguish two builds of the same release.
    """
    try:
        import importlib.metadata as md

        raw = md.distribution("abismal").read_text("direct_url.json")
        if not raw:
            return None
        return json.loads(raw).get("vcs_info", {}).get("commit_id")
    except Exception:
        return None


def requested_revision():
    """What was asked for at install time -- a branch name, a tag, or a commit.

    pip records this next to the resolved commit, so a build installed from
    `@gui` remembers that, which is what makes an unattended freshness check
    possible: there is something to compare the branch against.
    """
    try:
        import importlib.metadata as md

        raw = md.distribution("abismal").read_text("direct_url.json")
        if not raw:
            return None
        return json.loads(raw).get("vcs_info", {}).get("requested_revision")
    except Exception:
        return None


def branch_tip(branch, repo=REPO, timeout=15):
    """The head of `branch` and its parents, or None if it cannot be asked.

    Both are needed because devtools/colab_deploy.py pushes *two* commits: one
    holding the code, and one on top of it holding the notebook rewritten to
    install that code. The notebook therefore pins the head's parent, never the
    head, so comparing an install against the head alone reports every healthy
    deploy as stale -- which is what the first version of this did.

    Unauthenticated, so it can fail for reasons unrelated to you: shared
    addresses hit the hourly limit. None means "unknown", never "stale".
    """
    url = f"https://api.github.com/repos/{repo}/commits/{branch}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            commit = json.load(response)
        return {
            "sha": commit.get("sha"),
            "parents": [p.get("sha") for p in commit.get("parents", [])],
        }
    except Exception:
        return None


def describe_build(expected=None, branch=None, repo=REPO):
    """One line on whether the running code is current, and what to do if not.

    `expected` is the commit the notebook pinned; `branch` the branch it was
    deployed from. Either may be omitted, and whatever can be checked is.
    """
    running = installed_commit()
    if running is None:
        return "build: not installed from git, so its commit cannot be determined"

    short = running[:8]
    # Fall back to whatever was asked for at install time, so a plain launch()
    # can still say whether it is current without being told the branch. A
    # resolved commit is not a branch, so there is nothing to compare it to.
    if branch is None:
        asked = requested_revision()
        if asked and asked != running:
            branch = asked
    if expected and running != expected:
        return (
            f"build: {short} STALE -- this notebook pinned {expected[:8]}.\n"
            "       The install only runs when abismal is absent, so a session\n"
            "       that already had one kept it. Runtime > Disconnect and\n"
            "       delete runtime, then run again."
        )

    tip = branch_tip(branch, repo) if branch else None
    if tip:
        # Current means the install is either the head or the commit the head
        # sits on -- see branch_tip for why the notebook pins the parent.
        current = {tip["sha"], *tip["parents"]}
        if running not in current:
            return (
                f"build: {short} matches this notebook, but {branch} has moved to\n"
                f"       {tip['sha'][:8]}. The notebook itself is stale -- GitHub\n"
                "       caches branch files briefly. Reload the page and run again."
            )
        return f"build: {short} current -- matches {branch}"
    if branch:
        return f"build: {short} matches this notebook ({branch} head unknown)"
    return f"build: {short}"
