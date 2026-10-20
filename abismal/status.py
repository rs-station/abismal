"""
One-line progress messages for people and for the GUI.

`status("Reading the data")` prints `[status] Reading the data`. The GUI's runner
tails the run's output and shows the latest such line beside the progress bar, so
these name what a run is doing at the moments when nothing else visibly moves.
`refinement(event, name)` prints `[refinement] <event> <name>` as refinement
workers start and finish, which the GUI uses to list what is running.
"""

STATUS_PREFIX = "[status] "
REFINEMENT_PREFIX = "[refinement] "


def status(message):
    print(STATUS_PREFIX + message, flush=True)


def refinement(event, name, detail=""):
    """event: 'started', 'finished', 'failed' or 'skipped'; name: the result directory's name."""
    print(f"{REFINEMENT_PREFIX}{event} {name}" + (f" {detail}" if detail else ""), flush=True)
