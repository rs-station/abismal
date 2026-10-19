import pytest
import torch

from abismal.command_line.parser import parser
from abismal.device import resolve_compile, resolve_precision, supports_bfloat16

ARGS = ["-d", "2.0", "dummy.mtz"]


@pytest.mark.parametrize("flags,expected", [
    ([], "auto"),
    (["--compile"], "always"),
    (["--jit-compile"], "always"),
    (["--compile", "never"], "never"),
    (["--compile=always"], "always"),
])
def test_compile_flag_spellings(flags, expected):
    assert parser.parse_args(flags + ARGS).compile == expected


def test_precision_defaults_to_auto():
    assert parser.parse_args(ARGS).precision == "auto"


def test_auto_compiles_on_cuda_only():
    assert resolve_compile("auto", torch.device("cuda")) is True
    assert resolve_compile("auto", torch.device("cpu")) is False
    assert resolve_compile("always", torch.device("cpu")) is True
    assert resolve_compile("never", torch.device("cuda")) is False


def test_auto_precision_is_float32_off_gpu():
    assert resolve_precision("auto", torch.device("cpu")) == "float32"
    assert resolve_precision("bfloat16", torch.device("cpu")) == "bfloat16"
    assert resolve_precision("float32", torch.device("cuda")) == "float32"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_auto_precision_follows_native_bfloat16_support():
    device = torch.device("cuda")
    expected = "bfloat16" if supports_bfloat16(device) else "float32"
    assert resolve_precision("auto", device) == expected
