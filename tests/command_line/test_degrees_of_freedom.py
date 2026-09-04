"""The --studentt-dof contract.

The error model is Student's t with 32 degrees of freedom by default. Because a
t distribution converges to a normal as its degrees of freedom go to infinity,
every spelling of "infinite" resolves to None, which is what selects
NormalLikelihood in abismal/command_line/abismal.py.
"""

import math
from argparse import ArgumentTypeError

import numpy as np
import pytest

from abismal.command_line.parser.custom_types import degrees_of_freedom


@pytest.mark.parametrize(
    "value",
    [0, "0", 0.0, math.inf, np.inf, float("inf"), "inf", "Inf", "INF", "infinity",
     None, "None", "none", " none ", ""],
)
def test_infinite_spellings_mean_normal(value):
    assert degrees_of_freedom(value) is None


@pytest.mark.parametrize(
    "value,expected", [(32, 32.0), ("32", 32.0), (32.0, 32.0), ("8.5", 8.5), (1, 1.0)]
)
def test_finite_values_pass_through_as_float(value, expected):
    result = degrees_of_freedom(value)
    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize("value", ["-1", -5.0, "nan", float("nan"), "abc", "1,2"])
def test_invalid_values_are_rejected(value):
    """Caught at parse time rather than surfacing later as a NaN loss."""
    with pytest.raises(ArgumentTypeError):
        degrees_of_freedom(value)


def test_cli_default_is_32():
    from abismal.command_line.parser import parser

    args = parser.parse_args(["-d", "4.0", "-o", "unused", "unused.mtz"])
    assert args.studentt_dof == 32.0


@pytest.mark.parametrize("spelling", ["0", "inf", "none", "None"])
def test_cli_selects_the_normal_likelihood(spelling):
    """None is the value abismal.py tests to choose NormalLikelihood."""
    from abismal.command_line.parser import parser

    args = parser.parse_args(
        ["-d", "4.0", "-o", "unused", "--studentt-dof", spelling, "unused.mtz"]
    )
    assert args.studentt_dof is None


def test_cli_short_flag_still_works():
    from abismal.command_line.parser import parser

    args = parser.parse_args(["-d", "4.0", "-o", "unused", "-t", "8", "unused.mtz"])
    assert args.studentt_dof == 8.0
