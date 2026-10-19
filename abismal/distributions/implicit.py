"""Implicit reparameterization gradients (Figurnov et al. 2018, arXiv:1805.08498).

For a sample ``z`` drawn by a non-differentiable procedure, the reparameterization
gradient is ``dz/dtheta = -(dF/dtheta) / f(z)`` with ``F`` the CDF and ``f`` the
density. The samplers here compute ``z`` and those derivatives without a graph and
splice them in with the "detach trick",

    out = z + sum_i g_i * (p_i - p_i.detach()),

whose value is ``z`` and whose gradient with respect to ``p_i`` is ``g_i`` (summed
by autograd over whatever dimensions ``p_i`` was broadcast along, including the
leading sample dimensions). Unlike a ``torch.autograd.Function`` this is ordinary
tensor arithmetic, so torch.compile traces straight through it.

The ``g_i`` must be finite: ``inf * 0`` would turn the *value* into NaN. Second
derivatives are not supported (``g_i`` is a constant to autograd), as with TFP.
"""
import torch


def implicit_reparameterization(z, *params_and_grads):
    """Attach the gradients ``dz/dparam`` to the sample ``z``.

    Args:
        z: The sample, ``[*sample_shape, *batch_shape]``, without grad history.
        *params_and_grads: ``(param, dz_dparam)`` pairs. ``param`` must broadcast
            against ``z``; ``dz_dparam`` has the shape of ``z`` and no grad history.

    Returns:
        A tensor equal to ``z`` that carries the implicit gradients.
    """
    out = z
    for param, grad in params_and_grads:
        # A static check, so it does not break torch.compile graphs; it spares
        # the arithmetic in plain .sample() calls and for constant parameters.
        if torch.is_grad_enabled() and param.requires_grad:
            out = out + grad * (param - param.detach())
    return out
