#!/usr/bin/env python
"""Drive the official Colab runtime image locally.

The container reproduces Colab's *runtime* exactly -- the same Python, the same
preinstalled TensorFlow and numpy, the same process tree and filesystem layout --
which is the half of Colab that can be automated. It does not reproduce Colab's
frontend: the widget manager is proprietary and output areas are sandboxed onto a
separate origin, so anything about whether pixels appear still needs `serve` plus
a real Colab tab.

What that division buys: the install/dependency class of bug, the /proc and
JPY_PARENT_PID behaviour that decides where the file pickers open, PATH, and
whether a GPU runtime has a GPU TensorFlow can see -- all without a human.

Subcommands:

    pull    fetch the upstream image (~23 GB, once)
    build   install the working tree into it and commit, so later runs are fast
    probe   dump collect_colab_debug --json from inside the container
    run     run an arbitrary command inside the built image
    serve   start the runtime with its port published, for Colab's
            "connect to a local runtime" flow
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

UPSTREAM = "us-docker.pkg.dev/colab-images/public/runtime"
LOCAL = "localhost/abismal-colab:latest"

REPO = Path(__file__).resolve().parent.parent
# Colab's own working directory. Mounting the checkout here rather than somewhere
# neutral keeps paths the same as a user's session, which matters because the file
# pickers resolve against /content.
MOUNT = "/content/abismal"
# setuptools needs a writable source tree; the mount is read-only.
SRC = "/tmp/abismal-src"
ARTIFACTS = REPO / "devtools" / "artifacts" / "colab"


def _run(args, **kwargs):
    print("+", " ".join(str(a) for a in args), file=sys.stderr)
    return subprocess.run(args, **kwargs)


def _podman():
    from shutil import which

    for engine in ("podman", "docker"):
        if which(engine):
            return engine
    raise SystemExit("neither podman nor docker is on PATH")


def _upstream_entrypoint(engine):
    """The upstream image's ENTRYPOINT/CMD, as --change arguments for commit.

    `podman commit` carries over the *container's* config, and the build container
    runs with `--entrypoint /bin/bash` so that /datalab/run.sh does not start a
    jupyter server we do not want during an install. Committing it unchanged bakes
    that override into the image: `serve` then re-runs the pip install as its
    entrypoint and exits, having never started the server. Read the real values
    back rather than hardcoding them, so an upstream change does not silently
    reintroduce this.
    """
    import json

    out = subprocess.run(
        [engine, "image", "inspect", UPSTREAM,
         "--format", "{{json .Config.Entrypoint}}|{{json .Config.Cmd}}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    entrypoint, cmd = (json.loads(part or "null") for part in out.split("|", 1))
    changes = []
    if entrypoint is not None:
        changes += ["--change", f"ENTRYPOINT {json.dumps(entrypoint)}"]
    changes += ["--change", f"CMD {json.dumps(cmd or [])}"]
    return changes


def _image_exists(engine, name):
    got = subprocess.run(
        [engine, "image", "exists", name], capture_output=True
    ).returncode
    return got == 0


def cmd_pull(args):
    return _run([_podman(), "pull", UPSTREAM]).returncode


def cmd_build(args):
    """Install the working tree into the image once and commit the result.

    Without this every probe re-downloads TensorFlow, which dwarfs the thing being
    tested. The commit is the cache.
    """
    engine = _podman()
    if not _image_exists(engine, UPSTREAM):
        raise SystemExit(f"{UPSTREAM} not present; run `pull` first")

    container = "abismal-colab-build"
    _run([engine, "rm", "-f", container], capture_output=True)

    # --entrypoint bypasses /datalab/run.sh, which would start the jupyter server.
    #
    # Installed from a copy, not from the mount: setuptools writes abismal.egg-info
    # into the source directory, so a read-only mount fails with "Cannot update
    # time stamp of directory". Copying also keeps the host checkout clean of
    # build artifacts owned by a container uid.
    install = (
        f"rm -rf {SRC} && cp -a {MOUNT} {SRC} && rm -rf {SRC}/abismal.egg-info && "
        f"python -m pip install --no-cache-dir '{SRC}[gui]'"
    )
    got = _run(
        [
            engine, "run", "--name", container,
            "-v", f"{REPO}:{MOUNT}:ro",
            "--entrypoint", "/bin/bash",
            UPSTREAM, "-lc", install,
        ]
    )
    if got.returncode != 0:
        return got.returncode

    # Commit before verifying. The install takes minutes and the import check is
    # the thing most likely to fail -- on Colab it already does, over a
    # jax/ml_dtypes conflict -- and an image you cannot poke at is no way to
    # investigate why. A failed check is a finding, not a reason to throw the
    # artifact away.
    _run([engine, "commit", *_upstream_entrypoint(engine), container, LOCAL])
    _run([engine, "rm", "-f", container], capture_output=True)

    check = _run(
        [
            engine, "run", "--rm", "--entrypoint", "/bin/bash", LOCAL, "-lc",
            "python -c 'import abismal.gui; print(\"abismal.gui import OK\")'",
        ]
    )
    if check.returncode != 0:
        print(
            f"\nimage committed as {LOCAL}, but importing abismal.gui in it failed.\n"
            f"Investigate with: {Path(__file__).name} run python -c 'import abismal'",
            file=sys.stderr,
        )
    return check.returncode


def cmd_run(args):
    engine = _podman()
    image = LOCAL if _image_exists(engine, LOCAL) else UPSTREAM
    return _run(
        [
            engine, "run", "--rm",
            "-v", f"{REPO}:{MOUNT}:ro",
            "--entrypoint", "/bin/bash",
            image, "-lc", " ".join(args.command),
        ]
    ).returncode


def cmd_probe(args):
    engine = _podman()
    image = LOCAL if _image_exists(engine, LOCAL) else UPSTREAM
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / args.name

    got = _run(
        [
            engine, "run", "--rm",
            "-v", f"{REPO}:{MOUNT}:ro",
            "--entrypoint", "/bin/bash",
            image, "-lc",
            f"python {MOUNT}/devtools/collect_colab_debug.py --json",
        ],
        capture_output=True,
        text=True,
    )
    if got.returncode != 0:
        sys.stderr.write(got.stderr)
        return got.returncode

    out.write_text(got.stdout)
    print(f"wrote {out}")
    return 0


def _port_is_free(port):
    import socket

    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cmd_serve(args):
    """Start the real runtime for Colab's local-runtime flow.

    The image already allows Colab's origin (COLAB_JUPYTER_ALLOW_ORIGIN_PAT),
    which is what makes this supported rather than a hack -- verified: a request
    carrying `Origin: https://colab.research.google.com` gets a 200. The working
    tree is mounted read-write so an editable install stays live across edits.

    Inside the container jupyter listens on 9000 and Colab's own proxy on 8080;
    8080 is the one to publish, and the token jupyter prints has to be rewritten
    onto the host port before it is any use to anyone.
    """
    engine = _podman()
    image = LOCAL if _image_exists(engine, LOCAL) else UPSTREAM

    if not _port_is_free(args.port):
        raise SystemExit(
            f"port {args.port} is already in use -- a jupyter of your own, most "
            f"likely, since {args.port} is the conventional one.\n"
            f"  ss -ltnp | grep :{args.port}     # what holds it\n"
            f"  {Path(__file__).name} serve --port {args.port + 1}"
        )

    if not _port_is_free(args.ui_port):
        raise SystemExit(
            f"port {args.ui_port} (the browsable jupyter) is already in use.\n"
            f"  {Path(__file__).name} serve --ui-port {args.ui_port + 1}"
        )

    name = "abismal-colab-serve"
    _run([engine, "rm", "-f", name], capture_output=True)
    command = [
        engine, "run", "--rm", "--name", name,
        # 8080 is Colab's proxy: it answers /api/* and nothing else -- no /, no
        # /tree, no /lab. That is all Colab's frontend consumes, but it means the
        # connect URL 404s in a browser, which looks broken and is not.
        "-p", f"127.0.0.1:{args.port}:8080",
        # 9000 is the actual jupyter server, UI and all -- but it binds to the
        # container's own loopback, which published ports cannot reach. A socat
        # forwarder on 9001 is started once the server is up (see below), so the
        # same runtime can be opened directly and compared against what Colab
        # makes of it.
        "-p", f"127.0.0.1:{args.ui_port}:9001",
        "-v", f"{REPO}:{MOUNT}",
        image,
    ]
    print("+", " ".join(command), file=sys.stderr)

    token = re.compile(r"/\?token=([0-9a-f]+)")
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    announced = False
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if announced:
                continue
            found = token.search(line)
            if found:
                announced = True
                secret = found.group(1)
                url = f"http://127.0.0.1:{args.port}/?token={secret}"

                # jupyter is on the container's loopback, so bridge it to an
                # address a published port can actually reach. Done here rather
                # than at startup because the server has to exist first, and the
                # token line is the signal that it does.
                bridged = _run(
                    [engine, "exec", "-d", name, "socat",
                     "TCP-LISTEN:9001,fork,reuseaddr", "TCP:127.0.0.1:9000"],
                    capture_output=True,
                ).returncode == 0
                ui = (
                    # /tree, not /lab: the Colab image ships nbclassic and has
                    # no jupyterlab at all.
                    f"http://127.0.0.1:{args.ui_port}/tree?token={secret}"
                    if bridged else None
                )
                print(
                    "\n" + "=" * 70
                    + "\nColab local runtime is up. To attach a real Colab frontend:"
                    "\n  1. open https://colab.research.google.com"
                    "\n  2. Connect (top right) -> Connect to a local runtime"
                    "\n  3. paste this URL:\n"
                    f"\n     {url}\n"
                    "\nDo not open that one in a browser -- it is Colab's proxy, which"
                    "\nanswers /api/* and 404s everything else. A 404 there is correct."
                    + (
                        "\n\nThe same runtime, browsable, with jupyter's own UI:\n"
                        f"\n     {ui}\n"
                        if ui else
                        "\n\n(could not bridge jupyter's own UI out of the container;"
                        "\nthe Colab connect URL above is unaffected.)\n"
                    ) +
                    "\nThe checkout is mounted at " + MOUNT + ", so edits here are live."
                    "\nCtrl-C stops the runtime.\n" + "=" * 70 + "\n",
                    flush=True,
                )
        return proc.wait()
    except KeyboardInterrupt:
        _run([engine, "stop", "-t", "5", name], capture_output=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pull").set_defaults(func=cmd_pull)
    sub.add_parser("build").set_defaults(func=cmd_build)

    probe = sub.add_parser("probe")
    probe.add_argument("--name", default="probe.json", help="artifact filename")
    probe.set_defaults(func=cmd_probe)

    run = sub.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8888,
                       help="host port for Colab's proxy, the one Colab connects to")
    serve.add_argument("--ui-port", type=int, default=8890,
                       help="host port for jupyter's own browsable UI")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
