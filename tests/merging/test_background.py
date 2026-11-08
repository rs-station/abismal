import pytest
import torch
from abismal import tracking
from abismal.scaling import ImageScaler
from model_helpers import build_small_model, synthetic_batch


def _with_background(kind, **kwargs):
    model = build_small_model()
    k = {'offset': 1, 'latent': 2}[kind]
    model.scale_model = ImageScaler(model.scale_model.metadata_length, mlp_width=8, mlp_depth=2, prior_name="lognormal",
                                    background_outputs=k)
    model.background = kind
    model.background_prior_width = kwargs.get('width', 1.)
    model.background_sigma_column = kwargs.get('column', None)
    return model


def test_a_background_needs_its_sigma_column_and_a_known_kind():
    from abismal.merging import VariationalMergingModel
    m = build_small_model()
    args = (m.scale_model, m.surrogate_posterior, m.prior, m.likelihood)
    with pytest.raises(ValueError):
        VariationalMergingModel(*args, background='latent')
    with pytest.raises(ValueError):
        VariationalMergingModel(*args, background='flat')


@pytest.mark.parametrize('kind,column', [('offset', None), ('latent', 0)])
def test_the_background_trains_and_adds_its_prior_term(kind, column):
    model = _with_background(kind, column=column)
    batch = synthetic_batch(model).map(torch.as_tensor)
    if kind == 'latent':  # a positive background uncertainty in the raw metadata column
        batch = batch._replace(metadata=batch.metadata.abs() + 1.)
    with tracking.track() as t:
        model(batch, training=True)
    t.total_loss().backward()
    w = model.scale_model.output_dense.weight
    assert w.grad is not None and w.grad[model.scale_model.scale_outputs:].abs().sum() > 0


def test_an_offset_moves_the_prediction_by_sigma_times_beta():
    model = _with_background('offset')
    batch = synthetic_batch(model).map(torch.as_tensor)
    with tracking.track():
        model(batch, training=True)  # the standardization's first moments
    model.eval()
    def predict():
        torch.manual_seed(0)
        with tracking.track(), torch.no_grad():
            return model(batch, training=False)
    before = predict()
    with torch.no_grad():
        model.scale_model.output_dense.bias[model.scale_model.scale_outputs] += 2.
    after = predict()
    sig = model.standardize_inputs(batch, training=False).sigiobs.reshape(-1)
    shift = (after - before).reshape(-1)
    assert torch.allclose(shift, 2. * sig, rtol=1e-3, atol=1e-4)
