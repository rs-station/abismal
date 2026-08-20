"""Tests for the torchref worker's map-geometry and peak-finding helpers.

The worker is deliberately not importable as ``abismal.callbacks._torchref_worker``
-- it is spawned by file path so that it never imports the ``abismal`` package
and never initialises a second TensorFlow/CUDA context. These tests load it the
same way a spawned worker would, by path, so they exercise the module that
actually runs.

Everything here is a pure function over a numpy array or a gemmi grid: no
refinement, no GPU, no reference data.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

gemmi = pytest.importorskip("gemmi")
pytest.importorskip("skimage")

WORKER = (Path(__file__).resolve().parents[2]
          / "abismal" / "callbacks" / "_torchref_worker.py")


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location("_torchref_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid(shape=(24, 24, 24), cell=(24.0, 24.0, 24.0, 90, 90, 90), spacegroup="P 1"):
    g = gemmi.FloatGrid(*shape)
    g.unit_cell = gemmi.UnitCell(*cell)
    g.spacegroup = gemmi.SpaceGroup(spacegroup)
    return g


# --------------------------------------------------------------------------
# grid geometry
# --------------------------------------------------------------------------

def test_map_grid_size_is_fine_enough(worker):
    """The requested voxel size is an upper bound, on every axis."""
    cell = gemmi.UnitCell(93.99, 93.99, 130.87, 90, 90, 120)
    size = worker.map_grid_size(cell, voxel_size=0.3)
    for length, n in zip((cell.a, cell.b, cell.c), size):
        assert length / n <= 0.3


def test_map_grid_size_scales_with_cell_not_resolution(worker):
    """Grid size depends only on the cell.

    This is the property that makes peak z-scores comparable between datasets;
    it is what the old gemmi ``sample_rate`` (d_min/spacing) did not give.
    """
    small = worker.map_grid_size(gemmi.UnitCell(30, 30, 30, 90, 90, 90), 0.3)
    large = worker.map_grid_size(gemmi.UnitCell(60, 60, 60, 90, 90, 90), 0.3)
    assert [2 * n for n in small] == large


def test_peak_min_distance_uses_the_coarsest_axis(worker):
    """A voxel count must not overshoot the requested separation anywhere.

    peak_local_max measures min_distance isotropically in index space, so on an
    anisotropic grid the coarsest axis is the binding one -- overshooting there
    is what would merge two real atoms.
    """
    cell = gemmi.UnitCell(10.0, 10.0, 40.0, 90, 90, 90)
    shape = (100, 100, 100)          # 0.1, 0.1, 0.4 A spacing
    n = worker.peak_min_distance(cell, shape, separation=1.0)
    assert n * 0.4 <= 1.0            # coarsest axis stays within the request
    assert n >= 1


def test_peak_min_distance_never_zero(worker):
    """A grid coarser than the requested separation still separates by 1 voxel."""
    cell = gemmi.UnitCell(100.0, 100.0, 100.0, 90, 90, 90)
    assert worker.peak_min_distance(cell, (10, 10, 10), separation=1.0) == 1


# --------------------------------------------------------------------------
# periodic peak finding
# --------------------------------------------------------------------------

def test_periodic_peaks_finds_an_interior_maximum(worker):
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[10, 10, 10] = 5.0
    idx = worker.periodic_peak_indices(arr, min_distance=2, threshold=1.0)
    assert [tuple(i) for i in idx] == [(10, 10, 10)]


def test_periodic_peaks_finds_a_maximum_on_the_boundary(worker):
    """A peak at index 0 is interior to a periodic map, not on an edge.

    skimage's default ``exclude_border=True`` drops exactly this peak, which is
    why the implementation must pass ``exclude_border=False`` and pad.
    """
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[0, 0, 0] = 5.0
    idx = worker.periodic_peak_indices(arr, min_distance=2, threshold=1.0)
    assert [tuple(i) for i in idx] == [(0, 0, 0)]


def test_periodic_peaks_are_roll_invariant(worker):
    """Translating a periodic map must not change the set of peak heights.

    The regression this guards: folding padding-region detections back with a
    modulo instead of cropping them produced duplicate peaks that appeared and
    disappeared depending on where the map was cut.
    """
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(24, 24, 24)).astype(np.float32)
    arr[5, 7, 11] = 20.0
    arr[0, 0, 23] = 18.0             # straddles the seam
    arr[12, 12, 12] = 16.0

    def heights(a):
        idx = worker.periodic_peak_indices(a, min_distance=3, threshold=8.0)
        return sorted(np.round(a[tuple(idx.T)], 5))

    base = heights(arr)
    assert base == [16.0, 18.0, 20.0]
    for shift in [(0, 0, 1), (7, 0, 0), (12, 12, 12), (23, 23, 23), (5, 13, 19)]:
        assert heights(np.roll(arr, shift, axis=(0, 1, 2))) == base, shift


def test_periodic_peaks_merge_within_min_distance(worker):
    """Two maxima closer than min_distance collapse to the stronger one."""
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[10, 10, 10] = 5.0
    arr[10, 10, 12] = 4.0            # 2 voxels away
    idx = worker.periodic_peak_indices(arr, min_distance=4, threshold=1.0)
    assert [tuple(i) for i in idx] == [(10, 10, 10)]

    idx = worker.periodic_peak_indices(arr, min_distance=1, threshold=1.0)
    assert len(idx) == 2


def test_periodic_peaks_empty_when_nothing_clears_threshold(worker):
    arr = np.zeros((16, 16, 16), dtype=np.float32)
    idx = worker.periodic_peak_indices(arr, min_distance=2, threshold=1.0)
    assert len(idx) == 0


# --------------------------------------------------------------------------
# symmetry deduplication
# --------------------------------------------------------------------------

def test_dedup_symmetry_is_identity_in_p1(worker):
    """P1 has one operator, so nothing may be removed."""
    idx = np.array([[1, 2, 3], [10, 11, 12]])
    kept = worker.dedup_symmetry(idx, (24, 24, 24), gemmi.SpaceGroup("P 1"))
    assert kept.tolist() == idx.tolist()


def test_dedup_symmetry_collapses_equivalent_copies(worker):
    """All symmetry images of one site collapse to a single representative."""
    sg = gemmi.SpaceGroup("P 4")
    shape = (24, 24, 24)
    frac = np.array([0.1, 0.2, 0.3])
    idx = np.array([
        np.mod(np.round(np.mod(op.apply_to_xyz(list(frac)), 1.0) * shape).astype(int), shape)
        for op in sg.operations()
    ])
    assert len(idx) == 4
    kept = worker.dedup_symmetry(idx, shape, sg)
    assert len(kept) == 1


def test_dedup_symmetry_keeps_distinct_sites(worker):
    sg = gemmi.SpaceGroup("P 1")
    idx = np.array([[1, 1, 1], [8, 8, 8], [15, 15, 15]])
    kept = worker.dedup_symmetry(idx, (24, 24, 24), sg)
    assert len(kept) == 3


def test_dedup_symmetry_keeps_the_first_row(worker):
    """Callers sort by descending height, so the survivor must be the first."""
    sg = gemmi.SpaceGroup("P 4")
    shape = (24, 24, 24)
    frac = np.array([0.1, 0.2, 0.3])
    idx = np.array([
        np.mod(np.round(np.mod(op.apply_to_xyz(list(frac)), 1.0) * shape).astype(int), shape)
        for op in sg.operations()
    ])
    kept = worker.dedup_symmetry(idx, shape, sg)
    assert kept.tolist() == [idx[0].tolist()]


def test_dedup_symmetry_handles_empty_input(worker):
    kept = worker.dedup_symmetry(np.zeros((0, 3), dtype=int), (24, 24, 24),
                                 gemmi.SpaceGroup("P 1"))
    assert len(kept) == 0


# --------------------------------------------------------------------------
# ADP restraint propagation
# --------------------------------------------------------------------------

def test_adp_restraint_spec_empty_for_isotropic(worker):
    """Isotropic runs keep torchref's defaults on every channel."""
    assert worker.adp_restraint_spec("isotropic", 0.2) == {}
    assert worker.adp_restraint_spec("anisotropic", None) == {}


def test_adp_restraint_spec_targets_the_anisotropy_channel(worker):
    spec = worker.adp_restraint_spec("anisotropic", 0.2)
    assert spec == {"simu": {"simu_sigma_aniso": 0.2}}


def test_apply_adp_restraints_raises_when_a_value_does_not_stick(worker):
    """The read-back assert is the point of the helper.

    Without it this would fail the same silent way the upstream bug does: a
    target rebuild resets the sigma and refinement proceeds at a restraint
    weight nobody chose.
    """
    class Swallow:
        simu_sigma_aniso = 1.0

        def __setattr__(self, key, value):
            pass                      # accept the write, discard it

    class Ref:
        adp_target = {"simu": Swallow()}

    with pytest.raises(RuntimeError, match="did not stick"):
        worker.apply_adp_restraints(Ref(), {"simu": {"simu_sigma_aniso": 0.2}}, "test")


def test_apply_adp_restraints_noop_on_empty_spec(worker):
    """An empty spec must not touch the refinement at all."""
    class Boom:
        @property
        def adp_target(self):
            raise AssertionError("empty spec must not reach adp_target")

    worker.apply_adp_restraints(Boom(), {}, "test")
