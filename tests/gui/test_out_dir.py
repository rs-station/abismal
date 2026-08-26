"""out_dir's save-as row: parent + name, and the status line under it.

Kept apart from test_paths.py because this is the one picker that names
something that need not exist. The others are open dialogs, where a listing of
what is there is the complete set of valid answers; this one is a save target,
so browsing alone can never express it.
"""
import re

import pytest

from abismal.gui.components.argparse_gui import ArgparseGUIBase
from abismal.gui.components.file_selector import PathSelector

import gui_harness as H


def widget_for(form, dest):
    return next(w for a, w in form._all_args.items() if a.dest == dest)


def save_selector(parent, name=""):
    return PathSelector(
        description="out_dir", mode="save",
        value=str(parent), name=name, initial_directory=parent,
    )


def status_text(selector):
    """The status line with its markup stripped."""
    return re.sub(r"<[^>]+>", "", selector._status.value).replace("&mdash;", "--").strip()


# ---------------------------------------------------------------------------
# the value is the join
# ---------------------------------------------------------------------------

def test_the_value_joins_the_parent_and_the_name(tmp_path):
    selector = save_selector(tmp_path, "run_003")

    assert selector.value == f"{tmp_path}/run_003"


def test_an_empty_name_leaves_the_parent_itself(tmp_path):
    """Clearing the name is a legitimate way to say "write straight into here"."""
    selector = save_selector(tmp_path, "")

    assert selector.value == str(tmp_path)


def test_a_name_with_slashes_still_joins_cleanly(tmp_path):
    selector = save_selector(tmp_path, "/run_003/")

    assert selector.value == f"{tmp_path}/run_003"


def test_browsing_sets_the_parent_and_keeps_the_name(tmp_path):
    """The name is what you are creating; navigating is about where to put it."""
    (tmp_path / "hewl").mkdir()
    selector = save_selector(tmp_path, "run_003")

    selector._browse_button.click()
    selector._dir_list.value = "hewl"
    selector._select_button.click()

    assert selector.value == f"{tmp_path}/hewl/run_003"


def test_the_browser_offers_no_file_list(tmp_path):
    """There is no file to pick: the target does not exist yet."""
    selector = save_selector(tmp_path)
    selector._browse_button.click()

    assert selector._file_list is None


# ---------------------------------------------------------------------------
# the status line
# ---------------------------------------------------------------------------

def test_a_new_directory_says_it_will_be_created(tmp_path):
    selector = save_selector(tmp_path, "run_003")

    assert status_text(selector) == "will be created"


def test_a_missing_parent_is_called_out(tmp_path):
    """The case worth catching. Today a mistyped parent passes the form and only
    fails inside the child, where the message lands in console.log."""
    selector = save_selector(tmp_path / "hwel", "run_003")

    assert "does not exist" in status_text(selector)


def test_an_existing_empty_directory_is_not_flagged(tmp_path):
    (tmp_path / "run_003").mkdir()
    selector = save_selector(tmp_path, "run_003")

    assert status_text(selector) == "exists, and holds no abismal output"


def test_an_existing_run_warns_about_the_overwrite(tmp_path):
    """This is the collision that raises the confirmation dialog on Run. Saying
    so here means finding out before launching rather than after."""
    target = tmp_path / "run_003"
    target.mkdir()
    (target / "history.csv").write_text("Epoch\n1\n")
    (target / "abismal.log").write_text("")
    (target / "eff_0_asu_0_epoch_1").mkdir()

    selector = save_selector(tmp_path, "run_003")

    text = status_text(selector)
    assert "3 abismal output(s)" in text
    assert "overwrite" in text


def test_a_file_where_the_directory_should_go(tmp_path):
    (tmp_path / "run_003").write_text("")
    selector = save_selector(tmp_path, "run_003")

    assert "is a file" in status_text(selector)


def test_the_status_follows_what_you_type(tmp_path):
    """It is wired to both fields, so editing either one re-evaluates."""
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "history.csv").write_text("Epoch\n1\n")
    selector = save_selector(tmp_path, "fresh")
    assert status_text(selector) == "will be created"

    selector._name_input.value = "taken"
    assert "overwrite" in status_text(selector)

    selector._input.value = str(tmp_path / "nowhere")
    assert "does not exist" in status_text(selector)


def test_an_empty_target_says_nothing(tmp_path):
    selector = save_selector("", "")

    assert status_text(selector) == ""


# ---------------------------------------------------------------------------
# how the form sets it up
# ---------------------------------------------------------------------------

@pytest.fixture
def real_form():
    return H.build_form()


def test_out_dir_is_the_only_save_row(real_form):
    modes = {
        a.dest: w.mode for a, w in real_form._all_args.items()
        if isinstance(w, PathSelector)
    }
    assert modes["out_dir"] == "save"
    assert [d for d, m in modes.items() if m == "save"] == ["out_dir"]


def test_the_offered_name_is_timestamped(real_form):
    """So a second run does not land on the first and raise the overwrite dialog
    as a matter of routine."""
    widget = widget_for(real_form, "out_dir")

    assert re.fullmatch(r"abismal_\d{4}-\d{2}-\d{2}_\d{4}", widget._name_input.value)


def test_the_parent_is_the_directory_jupyter_was_launched_from(real_form):
    from abismal.gui.components.file_selector import default_directory

    assert widget_for(real_form, "out_dir")._input.value == default_directory()


def test_the_form_emits_the_joined_path(real_form):
    widget = widget_for(real_form, "out_dir")
    widget._input.value = "/data/hewl"
    widget._name_input.value = "run_003"

    args = real_form.to_args()

    assert args[args.index("-o") + 1] == "/data/hewl/run_003"


def test_the_joined_path_survives_the_parser(real_form):
    from pathlib import Path

    widget = widget_for(real_form, "out_dir")
    widget._input.value = "/data/hewl"
    widget._name_input.value = "run_003"

    parsed = real_form.parser.parse_args(real_form.to_args() + ["-d", "1.8", "in.mtz"])

    assert parsed.out_dir == Path("/data/hewl/run_003")


def test_the_extra_rows_line_up_under_the_label(real_form):
    """_set_label_widths sizes one label per control; this one has three rows and
    the two without a label still have to align with the one that has."""
    widget = widget_for(real_form, "out_dir")

    width = widget._label.layout.width
    assert width
    assert all(spacer.layout.width == width for spacer in widget._spacers)


def test_new_run_name_is_overridable():
    """It is a classmethod so a caller can name runs their own way."""
    class Fixed(ArgparseGUIBase):
        @staticmethod
        def new_run_name():
            return "always_this"

    assert Fixed.new_run_name() == "always_this"
