"""`--mtz-metadata` must reach the MTZ reader.

The option was defined, echoed into the run log, and read by nothing: a commit
that added metadata auto-detection removed the plumbing but left the flag, so a
user naming columns silently got the auto-detected set instead. These tests pin
the chain from the parsed option through DataManager to MTZLoader.
"""
import pytest

from abismal.io.manager import DataManager, _split_metadata_keys
from abismal.io.mtz import MTZLoader, MTZ_METADATA_KEYS


def test_auto_detection_is_the_default(conventional_mtz):
    """No option means the canonical list, filtered to what the file carries."""
    loader = MTZLoader(conventional_mtz)
    loader.get_dataset()
    assert loader.metadata_keys == [k for k in MTZ_METADATA_KEYS if k in
                                    __import__("reciprocalspaceship").read_mtz(conventional_mtz)]
    assert loader.explicit_metadata_keys is False


def test_explicit_keys_are_honoured(conventional_mtz):
    """Named columns are used verbatim, in the order given -- not the canonical order."""
    keys = ["YDET", "XDET"]
    loader = MTZLoader(conventional_mtz, metadata_keys=keys)
    loader.get_dataset()
    assert loader.metadata_keys == keys
    assert loader.metadata_length == 2


def test_explicit_keys_change_the_metadata_width(conventional_mtz):
    """The width the scale model sees must follow the request.

    This is what the broken wiring silently got wrong: the model was built
    against the auto-detected width no matter what the user asked for.
    """
    auto = MTZLoader(conventional_mtz)
    auto.get_dataset()
    picked = MTZLoader(conventional_mtz, metadata_keys=["XDET", "YDET"])
    picked.get_dataset()
    assert picked.metadata_length == 2
    assert auto.metadata_length > picked.metadata_length


def test_missing_explicit_key_is_an_error(conventional_mtz):
    """A named column that is absent must fail, not be dropped.

    Filtering is right for auto-detection and wrong here: quietly using fewer
    columns than asked for changes the model's input width with no indication.
    """
    loader = MTZLoader(conventional_mtz, metadata_keys=["XDET", "NOSUCHCOLUMN"])
    with pytest.raises(ValueError, match="NOSUCHCOLUMN"):
        loader.get_dataset()


def test_missing_key_is_tolerated_when_auto_detecting(conventional_mtz):
    """The canonical list names columns many files lack; that must stay fine."""
    loader = MTZLoader(conventional_mtz)
    loader.get_dataset()          # must not raise
    assert len(loader.metadata_keys) > 0


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ("XDET,YDET", ["XDET", "YDET"]),
    (" XDET , YDET ", ["XDET", "YDET"]),      # whitespace around commas
    ("XDET,,YDET", ["XDET", "YDET"]),         # empty field
])
def test_cli_string_parsing(value, expected):
    assert _split_metadata_keys(value) == expected


def test_manager_forwards_to_the_loader(conventional_mtz, monkeypatch):
    """The link that was missing: DataManager -> MTZLoader.

    Asserted on the argv the loader is constructed with rather than by building
    a dataset, so the test needs no ray cluster and cannot pass by accident if
    the keys are dropped further downstream.
    """
    seen = {}
    import abismal.io as io
    real = io.MTZLoader

    class Spy(real):
        def __init__(self, *args, **kwargs):
            seen["metadata_keys"] = kwargs.get("metadata_keys")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(io, "MTZLoader", Spy)
    manager = DataManager(inputs=[conventional_mtz], dmin=None,
                          mtz_metadata=["XDET", "YDET"])
    assert manager.mtz_metadata == ["XDET", "YDET"]
    manager.get_dataset()
    assert seen["metadata_keys"] == ["XDET", "YDET"], (
        "DataManager did not forward mtz_metadata to MTZLoader"
    )


def test_manager_config_round_trips(tmp_path, conventional_mtz):
    """datamanager.yml is reloaded to resume a run, so the keys must persist.

    `get_config` reads `self.cell.parameters`, so the cell is supplied here
    rather than discovered by reading the file.
    """
    import gemmi

    manager = DataManager(
        inputs=[conventional_mtz], dmin=None,
        cell=gemmi.UnitCell(79.0, 79.0, 38.0, 90.0, 90.0, 90.0),
        spacegroup=gemmi.SpaceGroup("P 43 21 2"),
        mtz_metadata=["XDET", "YDET"],
    )
    assert manager.get_config()["mtz_metadata"] == ["XDET", "YDET"]

    path = tmp_path / "datamanager.yml"
    manager.to_file(str(path))
    assert DataManager.from_file(str(path)).mtz_metadata == ["XDET", "YDET"]
