"""GemmiMolViewer: map column selection and the standalone HTML document.

`html` is a complete document that loads 3Dmol and gemmi from a CDN, so it can be
opened in a browser with no Jupyter involved -- which is what the browser tier does.
Everything except its rendering is assertable here.
"""
import pytest

import gui_harness as H
from abismal.gui.components.gemmimol import GemmiMolViewer


@pytest.fixture(scope="module")
def torchref_files(tmp_path_factory):
    """A refined.pdb + refined.mtz with torchref's column names."""
    return H.make_results_template(tmp_path_factory.mktemp("torchref"))


@pytest.fixture
def phenix_mtz(tmp_path):
    """The same reflections under phenix's column names."""
    import numpy as np
    import reciprocalspaceship as rs

    hkl = [(h, k, l) for h in range(1, 4) for k in range(1, 4) for l in range(1, 4)]
    rng = np.random.default_rng(0)
    n = len(hkl)
    ds = rs.DataSet(
        {
            "H": np.array([i[0] for i in hkl], "int32"),
            "K": np.array([i[1] for i in hkl], "int32"),
            "L": np.array([i[2] for i in hkl], "int32"),
            "2FOFCWT": rng.uniform(10, 100, n).astype("float32"),
            "PH2FOFCWT": rng.uniform(-180, 180, n).astype("float32"),
            "ANOM": rng.uniform(0, 5, n).astype("float32"),
            "PANOM": rng.uniform(-180, 180, n).astype("float32"),
        },
        cell=[30.0, 30.0, 30.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)
    path = tmp_path / "refine_001.mtz"
    ds.write_mtz(str(path))
    return path


# ---------------------------------------------------------------------------
# map_keys
# ---------------------------------------------------------------------------

def test_torchref_columns_give_a_2fofc_map(torchref_files):
    """Regression: only phenix's names were listed, so torchref runs got no 2Fo-Fc map.

    The failure was silent -- a missing column is filtered out, so the viewer simply
    showed the anomalous difference map alone with nothing reported anywhere.
    """
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )
    assert viewer.map_keys == ["FWT", "PHWT", "ANOM", "PANOM"]


def test_phenix_columns_still_work(torchref_files, phenix_mtz):
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"), mtz_file=str(phenix_mtz)
    )
    assert viewer.map_keys == ["2FOFCWT", "PH2FOFCWT", "ANOM", "PANOM"]


def test_keys_come_in_pairs(torchref_files, phenix_mtz):
    """gemmimol reads the flat list pairwise, so an odd length mis-assigns phases."""
    for mtz in (torchref_files / "refined.mtz", phenix_mtz):
        viewer = GemmiMolViewer(
            pdb_file=str(torchref_files / "refined.pdb"), mtz_file=str(mtz)
        )
        assert len(viewer.map_keys) % 2 == 0


def test_a_half_present_pair_is_dropped_whole(tmp_path, torchref_files):
    """An amplitude with no phase must not be emitted alone.

    Emitting it would shift every later pair by one and hand each map the phases of
    the map before it -- silently, since the list stays plausible.
    """
    import numpy as np
    import reciprocalspaceship as rs

    hkl = [(h, k, l) for h in range(1, 4) for k in range(1, 4) for l in range(1, 4)]
    n = len(hkl)
    rng = np.random.default_rng(0)
    ds = rs.DataSet(
        {
            "H": np.array([i[0] for i in hkl], "int32"),
            "K": np.array([i[1] for i in hkl], "int32"),
            "L": np.array([i[2] for i in hkl], "int32"),
            "FWT": rng.uniform(10, 100, n).astype("float32"),   # amplitude, no PHWT
            "ANOM": rng.uniform(0, 5, n).astype("float32"),
            "PANOM": rng.uniform(-180, 180, n).astype("float32"),
        },
        cell=[30.0, 30.0, 30.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
    ).infer_mtz_dtypes()
    ds.set_index(["H", "K", "L"], inplace=True)
    path = tmp_path / "partial.mtz"
    ds.write_mtz(str(path))

    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"), mtz_file=str(path)
    )
    assert viewer.map_keys == ["ANOM", "PANOM"]


def test_no_mtz_returns_none_rather_than_raising(torchref_files):
    """The guard tested pdb_file and then opened mtz_file until 2026-08-25."""
    viewer = GemmiMolViewer(pdb_file=str(torchref_files / "refined.pdb"), mtz_file=None)
    assert viewer.map_keys is None


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------

def test_html_is_a_complete_document(torchref_files):
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )
    html = viewer.html
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "</html>" in html


def test_every_template_placeholder_is_substituted(torchref_files):
    """A leftover $name would reach the browser as literal text."""
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )
    assert "$" not in viewer.html


def test_html_carries_the_viewer_id_and_map_keys(torchref_files):
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
        viewer_id="test-viewer-id",
    )
    html = viewer.html
    assert "test-viewer-id" in html
    for key in viewer.map_keys:
        assert key in html


def test_the_files_travel_inside_the_document(torchref_files):
    """No URL is involved any more. The only URLs a notebook can offer are /files/
    ones, which the jupyter server serves solely from under its root_dir -- so
    results written anywhere else had none, and Colab has no such endpoint at all.
    """
    import base64

    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )
    html = viewer.html

    for name in ("refined.pdb", "refined.mtz"):
        encoded = base64.b64encode((torchref_files / name).read_bytes()).decode()
        assert encoded in html
    assert "/files/" not in html.replace("/files/ ", "")   # only the comment remains
    assert str(torchref_files) not in html                 # nor any kernel-side path


def test_the_reload_payload_carries_the_files_too(torchref_files):
    """Re-embedding would rebuild the iframe and reset the camera, so later epochs
    arrive by postMessage instead."""
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )

    payload = viewer.reload_payload

    assert payload["type"] == "reload"
    assert payload["map_keys"] == ["FWT", "PHWT", "ANOM", "PANOM"]
    assert payload["pdb_b64"] and payload["mtz_b64"]
    # The parent broadcasts to every iframe, so the id has to ride along for the
    # viewer to tell its own reload from another viewer's.
    assert payload["viewer_id"] == viewer.viewer_id


def test_base64_needs_no_escaping_in_the_document(torchref_files):
    """It is dropped straight into a javascript string literal."""
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )

    for encoded in (viewer.pdb_b64, viewer.mtz_b64):
        assert not (set(encoded) - set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        ))


def test_a_viewer_with_no_files_encodes_to_nothing(torchref_files):
    assert GemmiMolViewer().pdb_b64 == ""


def test_the_viewer_is_sealed_in_an_iframe(torchref_files):
    """Regression: the viewer's CSS escaped and took the notebook with it.

    Its stylesheet opens `* { margin: 0 ... }` and puts `background-color:
    black; height: 600px; overflow: hidden` on html and body -- right for a page
    the viewer owns, ruinous anywhere else. Jupyter's `isolated: True` metadata
    is meant to keep it in its own frame; JupyterLab honours that and Colab does
    not, so on Colab the rules applied to the notebook and everything below the
    viewer went black and got clipped. Reproduced in a browser: the unsealed
    document sets overflow:hidden on its host, the sealed one does not.

    srcdoc makes the isolation structural rather than a request, so it does not
    depend on a frontend choosing to cooperate.
    """
    viewer = GemmiMolViewer(
        pdb_file=str(torchref_files / "refined.pdb"),
        mtz_file=str(torchref_files / "refined.mtz"),
    )
    sealed = viewer.iframe_html

    assert sealed.startswith("<iframe "), sealed[:40]
    assert "srcdoc=" in sealed
    # Nothing the host page could interpret as its own markup or styling.
    for raw in ("<!doctype html>", "<style>", "<html", "<body"):
        assert raw not in sealed, f"{raw!r} reached the host page unescaped"
    assert "&lt;!doctype html&gt;" in sealed, "the document should be escaped, not dropped"


def test_the_runner_renders_the_sealed_viewer(torchref_files):
    """The isolation is worthless if the runner still emits the bare document."""
    from pathlib import Path

    import abismal.gui.runner as runner_module

    source = Path(runner_module.__file__).read_text()
    assert "viewer.iframe_html" in source
    assert "'text/html': viewer.html" not in source
