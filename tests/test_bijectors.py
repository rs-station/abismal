import numpy as np
import pytest
import torch

from abismal.bijectors import squareplus, squareplus_inverse, BIJECTORS
from abismal.surrogate_posterior.surrogate_posterior import PositiveParameter


def test_squareplus_matches_its_definition_and_inverts():
    x = torch.linspace(-50., 50., 1001, dtype=torch.float64)
    torch.testing.assert_close(squareplus(x), 0.5 * (x + torch.sqrt(x * x + 4.)))
    torch.testing.assert_close(squareplus_inverse(squareplus(x)), x)
    assert squareplus(torch.tensor(0.)) == 1.


def test_squareplus_tail_is_algebraic_and_free_of_cancellation():
    x = torch.tensor([-1e2, -1e4, -1e6, -1e8], dtype=torch.float32, requires_grad=True)
    y = squareplus(x)
    np.testing.assert_allclose(y.detach().numpy(), -1. / x.detach().numpy(), rtol=1e-3)
    y.sum().backward()
    assert (x.grad > 0).all()   # never exactly zero, unlike exp or softplus out here
    assert torch.exp(torch.tensor(-1e4)) == 0.


@pytest.mark.parametrize("bijector", sorted(BIJECTORS))
def test_positive_parameter_starts_at_its_init(bijector):
    init = torch.tensor([1e-6, 0.3, 1., 7.])
    p = PositiveParameter(init, shift=1e-12, bijector=bijector)
    torch.testing.assert_close(p(), init, rtol=1e-5, atol=1e-10)
    p.set_values(torch.tensor([2., 2., 2., 2.]))
    torch.testing.assert_close(p(), torch.full((4,), 2.))


def test_unknown_bijector():
    with pytest.raises(ValueError):
        PositiveParameter(torch.ones(2), bijector="softplus")


@pytest.mark.parametrize("posterior", ["foldednormal", "rice"])
def test_posterior_round_trip(tmp_path, posterior):
    from model_helpers import build_small_model
    from abismal.saving import load_model
    m = build_small_model(posterior=posterior)
    q = m.surrogate_posterior
    q2 = type(q)(q.rac, loc_init=q.loc(), scale_init=q.scale(), bijector="squareplus")
    assert q2.get_config()["bijector"] == "squareplus"
    torch.testing.assert_close(q2.loc(), q.loc(), rtol=1e-5, atol=1e-6)
    m.surrogate_posterior = q2
    from abismal.saving import save_model
    save_model(m, str(tmp_path / "m.pt"))
    r = load_model(str(tmp_path / "m.pt"))
    assert r.surrogate_posterior.bijector == "squareplus"
    torch.testing.assert_close(r.surrogate_posterior.loc(), q2.loc())
    torch.testing.assert_close(r.surrogate_posterior.scale(), q2.scale())


def test_old_configs_default_to_exp():
    from model_helpers import build_small_model
    q = build_small_model().surrogate_posterior
    config = q.get_config()
    config.pop("bijector")
    assert type(q).from_config(config).bijector == "exp"
