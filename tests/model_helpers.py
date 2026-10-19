"""Small real models and batches for tests that need them."""


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


SMALL_FACTS = {
    "cell": [34., 45., 98., 90., 90., 90.],
    "spacegroup": "P 21 21 21",
    "dmin": 4.0,
    "num_asus": 1,
    "metadata_length": 3,
}


def build_cli_model(*flags, facts=None):
    """A small model built the way the CLI builds one, so it can be saved and reloaded."""
    from abismal.build import build_model
    from abismal.command_line.parser import parser

    args = parser.parse_args(["-o", "unused", "-d", "4", "unused.mtz", "--d-model", "8", "--layers", "2", *flags])
    return build_model(args, dict(SMALL_FACTS if facts is None else facts)), args
