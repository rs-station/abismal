"""
End to end tests for abismal using a limited feature set
"""
import gemmi #this is necessary for some baffling dependency reason
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from abismal.command_line.abismal import main as abismal_main
from abismal.command_line.cchalf import main as cchalf_main
import pytest
from os.path import exists
from os import chdir
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
    #"--embed",
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

        args = " datamanager.yml epoch_2.keras --run-eagerly --sf-init epoch_0.keras "
        cchalf_main(args.split())
        assert exists('abismal_xval.mtz')

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
        pytest.param('structure_factor', 'gamma', marks=pytest.mark.xfail(reason='Not implemented')),
        ('structure_factor', 'truncatednormal'), 
        ('intensity', 'normal'),
        ('intensity', 'foldednormal'),
        pytest.param('intensity', 'rice', marks=pytest.mark.xfail(reason='Not implemented')),
        ('intensity', 'gamma'),
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


def test_ev11(conventional_mtz):
    flags = base_flags + ('--refine-uncertainties',)
    files = [
        conventional_mtz,
    ]
    run_abismal(
        flags,
        files,
    )
