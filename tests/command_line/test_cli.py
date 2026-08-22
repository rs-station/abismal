"""
End to end tests for abismal using a limited feature set
"""
import gemmi #this is necessary for some baffling dependency reason
import reciprocalspaceship as rs
import numpy as np
import tf_keras as tfk
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from abismal.command_line.abismal import main as abismal_main
from abismal.command_line.cchalf import main as cchalf_main
import pytest
from os.path import exists
from os import chdir
from glob import glob
from tempfile import TemporaryDirectory



base_flags = (
    "--dmin=5.0",
    "--epochs=2",
    "--batch-size=3",
    "--d-model=4",
    "--layers=2",
    "--run-eagerly",
    "--shuffle-buffer-size=10",
    "--steps-per-epoch=10",
    "--num-cpus=1",
    "--debug",
)

def run_abismal(flags, files, additional_asserts=()):
    with TemporaryDirectory() as output_dir:
        chdir(output_dir)
        flags = ' ' + ' '.join(flags) + ' ' 
        args = flags + f' -o ./ '
        args += ' '.join(files)
        for file in files:
            args += f' {file} '
        abismal_main(args.split())

        assert exists('asu_0_epoch_1.mtz')
        assert exists('asu_0_epoch_2.mtz')
        assert exists('datamanager.yml')
        assert exists('epoch_0.keras')
        assert exists('epoch_1.keras')
        assert exists('epoch_2.keras')
        assert exists('history.csv')
        for add in additional_asserts:
            assert add()

        tfk.saving.load_model("epoch_0.keras")
        tfk.saving.load_model("epoch_1.keras")
        tfk.saving.load_model("epoch_2.keras")

        rs.read_mtz("asu_0_epoch_1.mtz")
        rs.read_mtz("asu_0_epoch_2.mtz")

        args = " datamanager.yml epoch_2.keras --run-eagerly --sf-init epoch_0.keras "
        cchalf_main(args.split())
        assert exists('abismal_xval.mtz')
        assert_cchalf_optimized()


def assert_cchalf_optimized():
    """
    The half-dataset structure factors must actually move during abismal.cchalf.

    cchalf freezes everything but the surrogate posterior. Doing that by way of
    `model.trainable = False` silently zeroes out `model.trainable_variables`
    -- tf_keras gates a layer's trainable weights on the layer itself -- and
    `train_step` then applies gradients to nothing. cchalf still ran to
    completion and still wrote every mtz, so only the *values* betray it: the
    per-epoch files came out byte-identical. Compare them.
    """
    def read(path):
        # An asu with nothing `seen` in this half writes a zero-row mtz, which
        # gemmi refuses to read back. At test scale that is routine (--separate
        # gives six asus ten steps to be observed in), and it is not the failure
        # this assertion is looking for, so pass over those.
        try:
            ds = rs.read_mtz(path)
        except RuntimeError:
            return None
        return ds if len(ds) > 0 else None

    compared = 0
    for first_path in sorted(glob('half_*/asu_*_epoch_1.mtz')):
        first = read(first_path)
        last = read(first_path.replace('_epoch_1.mtz', '_epoch_2.mtz'))
        if first is None or last is None:
            continue

        # `seen` only ever grows, so epoch 2 covers at least epoch 1's reflections.
        shared = first.index.intersection(last.index)
        # Posterior type and anomalous flag decide whether the columns are F, I or
        # F(+)/F(-), so take every float column rather than naming one.
        keys = [k for k in first.columns if k in last.columns
                and np.issubdtype(first[k].to_numpy().dtype, np.floating)]
        if len(shared) == 0 or len(keys) == 0:
            continue

        compared += 1
        before = np.concatenate([first.loc[shared][k].to_numpy('float32') for k in keys])
        after = np.concatenate([last.loc[shared][k].to_numpy('float32') for k in keys])
        assert not np.array_equal(before, after), (
            f"{first_path}: structure factors are unchanged after an epoch of "
            f"cchalf; the surrogate posterior is not being optimized"
        )

    assert compared > 0, "no half-dataset mtz pair had reflections to compare"


def test_mtz(conventional_mtz):
    flags = base_flags
    files = [
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )

def test_dials(stills_expt, stills_refl):
    flags = base_flags
    files = [
        stills_expt,
        stills_refl,
    ]
    run_abismal(
        flags,
        files,
    )

def test_stream(stills_stream):
    flags = base_flags
    files = [
        stills_stream,
    ]
    run_abismal(
        flags,
        files,
    )


@pytest.mark.parametrize(
    ('kind', 'distribution'), (
        ('structure_factor', 'normal'),
        ('structure_factor', 'foldednormal'),
        ('structure_factor', 'rice'),
        ('structure_factor', 'nakagami'),
        pytest.param('structure_factor', 'gamma', marks=pytest.mark.xfail(reason='Not implemented')),
        ('structure_factor', 'truncatednormal'),
        ('intensity', 'normal'),
        ('intensity', 'foldednormal'),
        pytest.param('intensity', 'rice', marks=pytest.mark.xfail(reason='Not implemented')),
        ('intensity', 'gamma'),
        pytest.param('intensity', 'nakagami', marks=pytest.mark.xfail(reason='Not implemented')),
        pytest.param('intensity', 'truncatednormal', marks=pytest.mark.xfail(reason='Not implemented')),
    )
)
def test_posteriors(conventional_mtz, kind, distribution):
    flags = base_flags  + (
        f"--posterior-type={kind}",
        f"--posterior-distribution={distribution}"
    )
    files = [
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )


def test_multivariate_normal_posterior(conventional_mtz):
    flags = base_flags  + (
        f"--posterior-type=structure_factor",
        f"--posterior-distribution=normal",
        f"--posterior-rank=3",
        f"--prior-distribution=normal",
    )
    files = [
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )

def test_separate(conventional_mtz):
    flags = base_flags + ('--separate',)
    files = [
        conventional_mtz,
        conventional_mtz,
        conventional_mtz,
    ]
    additional_asserts = (
        lambda : exists('asu_1_epoch_1.mtz'),
        lambda : exists('asu_2_epoch_1.mtz'),
    )
    run_abismal(
        flags,
        files,
        additional_asserts
    )


def test_glu(conventional_mtz):
    flags = base_flags + ('--gated',)
    files = [
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )


def test_multivariate_wilson_prior(conventional_mtz):
    flags = base_flags  + (
        f"--parents=0,0",
        f"-r 0.0,0.99",
        f"--prior-distribution=wilson",
    )
    files = [
        conventional_mtz,
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )

