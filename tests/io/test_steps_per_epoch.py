"""The --steps-per-epoch contract.

An epoch is 1000 gradient steps by default, with the dataset repeated to fill
it, so epoch length -- and therefore how often the expensive per-epoch
callbacks run -- does not scale with dataset size. Passing 0 asks for one pass
over the data instead, which is what the flag's absence used to mean.
"""

import pytest

from abismal.io.manager import DataManager


def _manager(mtz, **kwargs):
    kwargs.setdefault("dmin", 4.0)
    kwargs.setdefault("num_cpus", 1)
    return DataManager([mtz], **kwargs)


@pytest.mark.parametrize(
    "steps_per_epoch,expected,expected_repeat",
    [
        (0, None, False),
        (None, None, False),
        (1, 1, True),
        (1000, 1000, True),
    ],
)
def test_zero_means_one_pass(
    conventional_mtz, steps_per_epoch, expected, expected_repeat
):
    """0 and None are the same thing internally, and neither repeats the data."""
    dm = _manager(conventional_mtz, steps_per_epoch=steps_per_epoch)
    assert dm.steps_per_epoch is expected or dm.steps_per_epoch == expected
    assert dm.repeat_train is expected_repeat


@pytest.mark.parametrize("steps_per_epoch", [0, None, 250])
def test_survives_a_config_round_trip(conventional_mtz, steps_per_epoch):
    """A datamanager.yml written with 0 must reload as one-pass, not as 0.

    This also covers files written before the default changed, which recorded
    null for what is now spelled 0.
    """
    dm = _manager(
        conventional_mtz,
        cell=(10, 10, 10, 90, 90, 90),
        spacegroup=1,
        steps_per_epoch=steps_per_epoch,
    )
    reloaded = DataManager.from_config(dm.get_config())
    assert reloaded.steps_per_epoch == dm.steps_per_epoch
    assert reloaded.repeat_train == dm.repeat_train


def test_cli_default_is_1000(conventional_mtz):
    from abismal.command_line.parser import parser

    args = parser.parse_args(["-d", "4.0", "-o", "unused", conventional_mtz])
    assert args.steps_per_epoch == 1000
    assert DataManager.from_parser(args).steps_per_epoch == 1000


def test_cli_zero_reaches_the_manager_as_one_pass(conventional_mtz):
    from abismal.command_line.parser import parser

    args = parser.parse_args(
        ["-d", "4.0", "-o", "unused", "--steps-per-epoch", "0", conventional_mtz]
    )
    assert args.steps_per_epoch == 0
    dm = DataManager.from_parser(args)
    assert dm.steps_per_epoch is None
    assert dm.repeat_train is False


def test_repeated_dataset_outlasts_one_pass(conventional_mtz):
    """The point of a fixed epoch: it can exceed the amount of data available.

    Without the repeat, asking for more steps than the data holds would end the
    epoch early.
    """
    import itertools

    dm = _manager(conventional_mtz, batch_size=1, steps_per_epoch=64)
    train, _ = dm.get_train_test_splits()
    assert len(list(itertools.islice(train, 64))) == 64

    one_pass = _manager(conventional_mtz, batch_size=1, steps_per_epoch=0)
    train_once, _ = one_pass.get_train_test_splits()
    assert len(list(train_once)) < 64
