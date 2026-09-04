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
    g.set_unit_cell(gemmi.UnitCell(*cell))
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


CELLS = [
    ((60.0, 70.0, 80.0, 90, 90, 90), "P 21 21 21"),   # orthorhombic
    ((93.99, 93.99, 130.87, 90, 90, 120), "P 61 2 2"),  # hexagonal
    ((40.0, 45.0, 50.0, 50, 55, 60), "P 1"),          # strongly oblique
]


def _sized_grid(worker, cell, spacegroup, voxel_size=0.3):
    unit_cell = gemmi.UnitCell(*cell)
    grid = gemmi.FloatGrid(*worker.map_grid_size(unit_cell, voxel_size=voxel_size))
    # set_unit_cell(), not `grid.unit_cell = ...`: gemmi recomputes derived
    # geometry only in the setter, so assignment leaves it on the default cell.
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.SpaceGroup(spacegroup)
    return unit_cell, grid


def _voxel_step_real_space(unit_cell, shape, axis):
    """Real-space distance covered by one voxel step along `axis`.

    Measured through orthogonalized coordinates rather than computed, so the
    test cannot restate the implementation's own arithmetic.
    """
    origin = unit_cell.orthogonalize(gemmi.Fractional(0, 0, 0))
    frac = [0.0, 0.0, 0.0]
    frac[axis] = 1.0 / shape[axis]
    return unit_cell.orthogonalize(gemmi.Fractional(*frac)).dist(origin)


@pytest.mark.parametrize("cell,spacegroup", CELLS)
def test_requested_voxel_size_holds_in_real_space(worker, cell, spacegroup):
    """One voxel must span at most the requested size, on every axis.

    Asserted against orthogonalized coordinates. Comparing against
    ``cell.a / nu`` instead would just restate ``map_grid_size``'s own ceil and
    pass for any cell.
    """
    unit_cell, grid = _sized_grid(worker, cell, spacegroup)
    for axis in range(3):
        step = _voxel_step_real_space(unit_cell, grid.shape, axis)
        assert step <= 0.3 + 1e-9, f"axis {axis} spans {step:.4f} A"


@pytest.mark.parametrize("cell,spacegroup", CELLS)
def test_peak_separation_bounds_the_axial_extent(worker, cell, spacegroup):
    """min_distance voxels must span at most the request ALONG AN AXIS.

    Only the axial extent is bounded. skimage suppresses within a Chebyshev
    cube, so the diagonal reaches sqrt(3) times further -- see
    `test_peak_separation_actually_resolves_two_maxima` for what that costs.

    The regression: sizing against gemmi's `Grid.spacing` (the perpendicular
    inter-plane distance, always <= the step length) inflates the voxel count.
    On the oblique cell that gives 4 voxels spanning 1.19 A for a requested 1.0.
    """
    unit_cell, grid = _sized_grid(worker, cell, spacegroup)
    n = worker.peak_min_distance(grid, separation=1.0)
    assert n >= 1
    for axis in range(3):
        span = n * _voxel_step_real_space(unit_cell, grid.shape, axis)
        assert span <= 1.0 + 1e-9, (
            f"{n} voxels along axis {axis} span {span:.4f} A, requested 1.0"
        )


def test_peak_separation_actually_resolves_two_maxima(worker):
    """Two maxima an S-S bond apart must come back as two peaks.

    Asserted by running the finder on a synthetic map rather than by dividing
    the requested separation by the quantity the implementation divides by --
    that is a floor identity and passes for any step-based implementation.

    Pins the real quantity: hewl's disulfide sulfurs are 2.02-2.05 A apart and
    each is its own anomalous scatterer, so merging them halves the site count.
    The margin is thin by design (the exclusion cube's diagonal is ~2.04 A),
    which is precisely why it is worth a test.
    """
    unit_cell = gemmi.UnitCell(79.34, 79.34, 37.81, 90, 90, 90)   # hewl
    shape = worker.map_grid_size(unit_cell, voxel_size=0.3)
    grid = gemmi.FloatGrid(*shape)
    grid.set_unit_cell(unit_cell)
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    n = worker.peak_min_distance(grid)

    arr = np.zeros(shape, dtype=np.float32)
    step = _voxel_step_real_space(unit_cell, shape, 0)
    offset = int(round(2.02 / step))          # an S-S bond along the a axis
    arr[20, 20, 20] = 10.0
    arr[20 + offset, 20, 20] = 9.0

    found = worker.periodic_peak_indices(arr, n, threshold=1.0)
    assert len(found) == 2, (
        f"min_distance={n} ({n * step:.2f} A axial) merged two maxima "
        f"{offset * step:.2f} A apart"
    )


def test_peak_min_distance_never_zero(worker):
    """A grid coarser than the requested separation still separates by 1 voxel."""
    grid = gemmi.FloatGrid(10, 10, 10)
    grid.set_unit_cell(gemmi.UnitCell(100.0, 100.0, 100.0, 90, 90, 90))
    grid.spacegroup = gemmi.SpaceGroup("P 1")
    assert worker.peak_min_distance(grid, separation=1.0) == 1


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


# --------------------------------------------------------------------------
# peaks.csv frames -- both halves of the property, on a non-P1 spacegroup
# --------------------------------------------------------------------------

def _one_atom_structure(spacegroup, cell, pos):
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*cell)
    st.spacegroup_hm = spacegroup
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name = "ZN"
    residue.seqid = gemmi.SeqId(1, " ")
    atom = gemmi.Atom()
    atom.name = "ZN"
    atom.element = gemmi.Element("Zn")
    atom.pos = gemmi.Position(*pos)
    atom.occ = 1.0
    atom.b_iso = 20.0
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    # Populate cell.images from the spacegroup; without them NeighborSearch
    # only sees the model's own copy and a peak on a symmetry image matches
    # nothing.
    st.setup_cell_images()
    return st


def test_peaks_csv_frames_agree(worker):
    """`|cen - coord| == dist` AND `coord*` is a modelled atom position.

    Both halves, together, on a non-P1 spacegroup with the density placed on a
    SYMMETRY IMAGE of the atom rather than on the atom itself -- which is what
    `dedup_symmetry` routinely leaves behind, since it keeps whichever copy was
    strongest wherever it sits.

    Each half alone is satisfiable by a wrong implementation, and both were
    gotten wrong in turn:
      - reporting the unmoved centroid satisfies neither;
      - moving the ATOM onto the peak satisfies the invariant while making
        `coord*` a position present in no file, breaking any consumer that
        joins peaks.csv back to the model by coordinate.
    """
    cell = (60.0, 60.0, 90.0, 90, 90, 90)
    st = _one_atom_structure("P 43 21 2", cell, (12.0, 18.0, 25.0))

    grid = gemmi.FloatGrid(*worker.map_grid_size(st.cell, voxel_size=0.5))
    grid.set_unit_cell(st.cell)
    grid.spacegroup = gemmi.SpaceGroup("P 43 21 2")

    # Put a blob on a symmetry image of the atom, not on the atom.
    op = list(gemmi.SpaceGroup("P 43 21 2").operations())[3]
    frac = st.cell.fractionalize(st[0][0][0][0].pos)
    image = np.mod(op.apply_to_xyz([frac.x, frac.y, frac.z]), 1.0)
    centre = np.round(image * np.array(grid.shape)).astype(int)
    arr = np.array(grid, copy=False)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            for dw in (-1, 0, 1):
                idx = tuple(np.mod(centre + [du, dv, dw], grid.shape))
                arr[idx] = 10.0 - abs(du) - abs(dv) - abs(dw)

    report = worker.peaks_by_local_max(st, grid, z_score_cutoff=3.0)
    assert len(report) >= 1, "the planted blob was not found"

    modelled = [a.pos for ch in st[0] for r in ch for a in r]
    for _, row in report.iterrows():
        cen = np.array([row["cenx"], row["ceny"], row["cenz"]])
        crd = np.array([row["coordx"], row["coordy"], row["coordz"]])
        assert abs(np.linalg.norm(cen - crd) - row["dist"]) < 1e-6, (
            "cen* and coord* are in different symmetry frames"
        )
        assert any(p.dist(gemmi.Position(*crd)) < 1e-6 for p in modelled), (
            "coord* is not a modelled atom position"
        )


def test_peaks_by_local_max_empty_keeps_the_schema(worker):
    """No peaks must still yield the full column set, not an empty frame.

    peaks.csv is read by abismal-benchmarks; a frame missing its columns turns
    a quiet epoch into a KeyError downstream.
    """
    cell = (40.0, 40.0, 40.0, 90, 90, 90)
    st = _one_atom_structure("P 1", cell, (10.0, 10.0, 10.0))
    grid = gemmi.FloatGrid(*worker.map_grid_size(st.cell, voxel_size=0.5))
    grid.set_unit_cell(st.cell)
    grid.spacegroup = gemmi.SpaceGroup("P 1")

    report = worker.peaks_by_local_max(st, grid, z_score_cutoff=5.0)
    assert len(report) == 0
    assert list(report.columns) == worker.PEAK_COLUMNS


# --------------------------------------------------------------------------
# absolute scaling of the refined mtz
#
# abismal merges on an arbitrary scale and torchref inherits it, so every
# amplitude it writes lands a couple of hundred times below absolute. The maps
# are fine -- rmsd-normalised quantities do not move -- but a viewer showing an
# absolute contour level shows ~0.003, and coot's initial 1.5-rmsd level then
# reads as 0.00.
# --------------------------------------------------------------------------

class FakeRefinement:
    """Stands in for the torchref refiner: the two structure factor sets only.

    `F_calc` is the model on an absolute scale, `F_calc_scaled` is that pushed
    through the scaler onto the observed scale, so their ratio is the factor.
    """

    def __init__(self, f_scaled, scale):
        torch = pytest.importorskip("torch")
        self._scaled = torch.tensor(f_scaled, dtype=torch.float64)
        self._abs = self._scaled * scale

    def get_F_calc(self):
        return self._abs

    def get_F_calc_scaled(self):
        return self._scaled


@pytest.fixture
def refined_mtz(tmp_path):
    """An mtz on an arbitrary scale, with PANOM unwrapped as torchref writes it."""
    n = 400
    rng = np.random.default_rng(0)
    hkls = np.array([(h, k, l)
                     for h in range(1, 9) for k in range(1, 9) for l in range(1, 9)],
                    dtype=float)[:n]
    amplitude = rng.uniform(0.05, 4.0, size=n)
    phase = np.linspace(-179.0, 179.0, n)

    mtz = gemmi.Mtz(with_base=True)
    mtz.cell = gemmi.UnitCell(30.0, 30.0, 30.0, 90, 90, 90)
    mtz.spacegroup = gemmi.SpaceGroup("P 1")
    mtz.add_dataset("refined")
    for label, ctype in (
        ("F-model", "F"), ("FWT", "F"), ("ANOM", "F"), ("SIGF-obs", "Q"),
        ("PHWT", "P"), ("PANOM", "P"), ("R-free-flags", "I"),
    ):
        mtz.add_column(label, ctype)
    mtz.set_data(np.column_stack([
        hkls,
        amplitude,              # F-model
        amplitude * 0.9,        # FWT
        amplitude * 0.01,       # ANOM
        amplitude * 0.05,       # SIGF-obs
        phase,                  # PHWT, already in range
        phase - 90.0,           # PANOM, unwrapped exactly as torchref writes it
        np.zeros(n),            # R-free-flags
    ]))
    path = tmp_path / "refined.mtz"
    mtz.write_to_file(str(path))
    return path, amplitude


def _column(path, label):
    mtz = gemmi.read_mtz_file(str(path))
    return np.array(mtz.column_with_label(label), copy=False).copy()


def test_scale_comes_from_the_refiner(worker, refined_mtz):
    """The factor is |F_calc| / |scaler(F_calc)|, taken from the refiner itself."""
    path, amplitude = refined_mtz
    K = 250.0
    before = _column(path, "F-model")

    scale = worker.rescale_mtz_to_absolute(path, FakeRefinement(amplitude, K))

    assert scale == pytest.approx(K, rel=1e-6)
    assert np.allclose(_column(path, "F-model"), before * K, rtol=1e-5)


def test_every_amplitude_column_is_scaled(worker, refined_mtz):
    """FWT, ANOM and the sigmas must move with F-model, or the maps disagree."""
    path, amplitude = refined_mtz
    before = {c: _column(path, c) for c in ("FWT", "ANOM", "SIGF-obs")}

    scale = worker.rescale_mtz_to_absolute(path, FakeRefinement(amplitude, 250.0))

    for label, original in before.items():
        assert np.allclose(_column(path, label), original * scale, rtol=1e-5), (
            f"{label} was not scaled"
        )


def test_phases_and_flags_are_left_alone(worker, refined_mtz):
    """Scaling a phase or an R-free flag would be a corruption, not a rescale."""
    path, amplitude = refined_mtz
    phwt_before = _column(path, "PHWT")
    flags_before = _column(path, "R-free-flags")

    worker.rescale_mtz_to_absolute(path, FakeRefinement(amplitude, 250.0))

    assert np.allclose(_column(path, "PHWT"), phwt_before)
    assert np.allclose(_column(path, "R-free-flags"), flags_before)


def test_panom_is_wrapped_into_range(worker, refined_mtz):
    """torchref writes `phase - 90` unwrapped, so PANOM arrives spanning [-450, 90]."""
    path, amplitude = refined_mtz
    before = _column(path, "PANOM")
    assert before.min() < -180.0, "fixture should start out of range"

    worker.rescale_mtz_to_absolute(path, FakeRefinement(amplitude, 250.0))

    after = _column(path, "PANOM")
    assert after.min() >= -180.0 and after.max() <= 180.0
    # Wrapping must be a no-op on the map: same angle, different representation.
    assert np.allclose(np.cos(np.deg2rad(after)), np.cos(np.deg2rad(before)), atol=1e-4)
    assert np.allclose(np.sin(np.deg2rad(after)), np.sin(np.deg2rad(before)), atol=1e-4)


def test_a_refiner_without_structure_factors_is_survivable(worker, refined_mtz):
    """No scale available must leave amplitudes untouched, not zero or nan them.

    PANOM is still wrapped -- that correction needs nothing from the refiner.
    """
    path, _ = refined_mtz
    fmodel_before = _column(path, "F-model")
    panom_before = _column(path, "PANOM")

    scale = worker.rescale_mtz_to_absolute(path, object())

    assert scale is None
    assert np.allclose(_column(path, "F-model"), fmodel_before)
    assert _column(path, "PANOM").min() >= -180.0
    assert panom_before.min() < -180.0


# --------------------------------------------------------------------------
# cell handling: the merged data are stamped with the model cell
# --------------------------------------------------------------------------

def _mtz(tmp_path, name, cell, spacegroup="P 21 21 21", column="F", flag=False):
    """A tiny merged-style MTZ on a handful of Miller indices."""
    import reciprocalspaceship as rs

    h = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [1, 1, 1]])
    ds = rs.DataSet(
        {
            "H": rs.DataSeries(h[:, 0], dtype="H"),
            "K": rs.DataSeries(h[:, 1], dtype="H"),
            "L": rs.DataSeries(h[:, 2], dtype="H"),
        },
        cell=gemmi.UnitCell(*cell),
        spacegroup=gemmi.SpaceGroup(spacegroup),
        merged=True,
    ).set_index(["H", "K", "L"])
    if flag:
        ds["R-free-flags"] = rs.DataSeries([0, 1, 1, 1, 1], index=ds.index, dtype="MTZInt")
    else:
        ds[column] = rs.DataSeries(np.arange(len(ds), dtype="float32") + 1.0,
                                   index=ds.index, dtype="F")
        ds["SIG" + column] = rs.DataSeries(np.ones(len(ds), "float32"),
                                           index=ds.index, dtype="Q")
    path = tmp_path / name
    ds.write_mtz(str(path))
    return str(path)


@pytest.mark.parametrize("pct", [0.0, 1.0, 5.0, 20.0])
def test_r_free_join_survives_a_disagreeing_cell(worker, tmp_path, pct):
    """Flags match on Miller index, so cell agreement is not a precondition.

    rs.DataSet.join rejects operands whose cells differ by more than 0.5% in any
    parameter. A merged MTZ carrying a nominal cell from a stream can easily sit
    outside that, and refusing the join there is a false negative.
    """
    import reciprocalspaceship as rs

    ref = (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
    off = (30.0 * (1 + pct / 100.0),) + ref[1:]
    flags = _mtz(tmp_path, f"flags{pct}.mtz", ref, flag=True)
    ds = rs.read_mtz(_mtz(tmp_path, f"data{pct}.mtz", off))

    out = worker.join_r_free_flags(ds, flags)
    assert "R-free-flags" in out.columns
    # every reflection is present in both files, so all of them match
    assert np.isfinite(out["R-free-flags"].to_numpy("float64")).all()


def test_r_free_join_still_rejects_a_spacegroup_mismatch(worker, tmp_path):
    """The half of the isomorphism check that does bear on matching hkl."""
    import reciprocalspaceship as rs

    cell = (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
    flags = _mtz(tmp_path, "sg_flags.mtz", cell, flag=True)
    ds = rs.read_mtz(_mtz(tmp_path, "sg_data.mtz", cell))
    ds.spacegroup = gemmi.SpaceGroup("P 1")

    with pytest.raises(ValueError, match="spacegroup"):
        worker.join_r_free_flags(ds, flags)


def test_prepare_data_stamps_the_model_cell(worker, tmp_path):
    """The written input_data.mtz carries the model's cell, not the merge's.

    Regression test for ordering: stamping the cell after the R-free join left
    the join to compare the merge cell against the flag file's, which fails on
    serial data whose nominal cell drifts from the model.
    """
    import reciprocalspaceship as rs

    model_cell = (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
    merged_cell = (33.0, 40.0, 47.0, 90.0, 90.0, 90.0)  # ~10% and ~6% off

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*model_cell)
    st.spacegroup_hm = "P 21 21 21"
    pdb = tmp_path / "model.pdb"
    st.write_pdb(str(pdb))

    data = _mtz(tmp_path, "merged.mtz", merged_cell)
    flags = _mtz(tmp_path, "rfree.mtz", model_cell, flag=True)

    out_path, _anomalous = worker.prepare_data(
        data, str(pdb), str(tmp_path), r_free_mtz=flags
    )
    written = rs.read_mtz(out_path)
    assert written.cell.parameters == pytest.approx(model_cell, abs=1e-3)
    assert "R-free-flags" in written.columns
