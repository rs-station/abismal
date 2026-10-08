import os

# Hide the GPU from every test process, before anything imports TensorFlow.
#
# addopts carries `-n auto`, so pytest-xdist starts one worker per core -- 16 on
# this machine -- and each one imports TensorFlow and takes the card's entire
# default allocation. set_memory_growth is called by abismal.io.set_gpu on the
# CLI path and never in a test, so nothing shares. The first worker wins the
# card and the rest die on `cudaSetDevice() ... out of memory`, which reads as a
# real failure in code that has nothing to do with devices: 64 tests pass under
# `-n 0` and 7 fail under `-n auto`, same card, same tests.
#
# A worker that hangs while holding the card is worse. One ran for 46 hours,
# keeping 46 GB of GPU and -- through file handles on already-unlinked ray spill
# files -- 264 GB of disk that `du` could not see, until the filesystem hit 100%.
#
# Set here rather than per-file because it has to beat the first `import
# tensorflow` in any worker, and conftest is imported before the test modules.
# test_cli, test_cchalf and test_transform_choices each call
# set_visible_devices([], 'GPU') already, but `--gpu-id` defaults to 0 and
# set_gpu turns the device back on inside main(), so whether a CLI test touched
# the GPU depended on what had run before it in the same process.
#
# ABISMAL_TEST_GPU=1 opts back in, for deliberately testing device placement.
# Use it with `-n 0`; with xdist it reproduces exactly the problem above.
if not os.environ.get("ABISMAL_TEST_GPU"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
from os.path import dirname, abspath, join



@pytest.fixture
def conventional_mtz():
    datapath = "data/conventional.mtz"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_refl():
    datapath = "data/stills.refl"
    filename = abspath(join(dirname(__file__), datapath))
    return filename

@pytest.fixture
def stills_expt():
    datapath = "data/stills.expt"
    filename = abspath(join(dirname(__file__), datapath))
    return filename


@pytest.fixture
def stills_stream():
    datapath = "data/crystfel.stream"
    filename = abspath(join(dirname(__file__), datapath))
    return filename



def build_small_model(metadata_length=3, dmin=4.0, posterior="foldednormal", prior="wilson", standardize=True, reindexing_ops=None):
    """A small but complete VariationalMergingModel, for tests that need a real one."""
    import gemmi
    from abismal.symmetry import ReciprocalASU, ReciprocalASUGraph
    from abismal.merging import VariationalMergingModel
    from abismal.scaling import ImageScaler
    from abismal.likelihood import NormalLikelihood

    cell = gemmi.UnitCell(34., 45., 98., 90., 90., 90.)
    sg = gemmi.SpaceGroup("P 21 21 21")
    rac = ReciprocalASUGraph(ReciprocalASU(cell, sg, dmin, anomalous=True))
    if prior == "wilson":
        from abismal.prior.structure_factor.wilson import WilsonPrior
        prior = WilsonPrior(rac)
    else:
        from abismal.prior.normal import NormalPrior
        prior = NormalPrior(rac)
    loc_init = prior.flat_distribution().mean
    scale_init = 0.1 * loc_init
    if posterior == "foldednormal":
        from abismal.surrogate_posterior.structure_factor import FoldedNormalPosterior as Posterior
    elif posterior == "rice":
        from abismal.surrogate_posterior.structure_factor import RicePosterior as Posterior
    q = Posterior(rac, loc_init=loc_init, scale_init=scale_init)
    scale_model = ImageScaler(metadata_length, mlp_width=8, mlp_depth=2, prior_name="lognormal")
    return VariationalMergingModel(
        scale_model, q, prior, NormalLikelihood(), mc_samples=4,
        reindexing_ops=reindexing_ops, standardize=standardize,
    )


def synthetic_batch(model, num_images=3, refls_per_image=20, seed=0):
    """A Batch of random observations of reflections in `model`'s ASU."""
    import numpy as np
    from abismal.io import Batch

    rng = np.random.default_rng(seed)
    rasu = model.surrogate_posterior.rac.reciprocal_asus[0]
    m = model.scale_model.metadata_length
    images = []
    for _ in range(num_images):
        H = rasu.Hunique[rng.choice(rasu.asu_size, refls_per_image)]
        n = len(H)
        I = rng.gamma(1., 10., size=(n, 1)).astype(np.float32)
        images.append((
            np.zeros((n, 1), np.int32),
            H.astype(np.int32),
            np.ones((n, 1), np.float32),
            np.ones((n, 1), np.float32),
            rng.normal(size=(n, m)).astype(np.float32),
            I,
            np.sqrt(I + 1.).astype(np.float32),
        ))
    return Batch.collate(images)
