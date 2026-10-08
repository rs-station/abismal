import math
import torch


class AbismalOptimizer(torch.optim.Optimizer):
    """
    Adam and its relatives with *lazy* updates for the surrogate posterior.

    Every surrogate posterior parameter belongs to one miller index, and a
    batch only observes a fraction of them. An ordinary Adam step would still
    move the unobserved ones -- their momenta decay toward zero and keep
    pushing the parameter -- at a rate set by how rarely they are observed.
    A parameter group with ``lazy=True`` instead updates, decays and advances
    the moments of only those entries whose gradient is nonzero this step.
    This is a dense, masked update rather than a sparse one, which keeps it
    compatible with clipping, gradient norms and torch.compile.

    The update rules, the clipping options and the decoupled weight decay
    follow the tf_keras optimizers abismal used before, including keras' order
    of operations: gradients are clipped, then weight decay is applied to the
    pre-update values, then the step is taken. The bias correction uses one
    global step count for every parameter, as keras' `iterations` did.

    Parameters
    ----------
    params : iterable
        Parameters or parameter groups. A group may set ``lazy=True``.
    learning_rate : float or callable
        A constant, or a function of the (zero based) step count.
    beta_1, beta_2, epsilon : float
    weight_decay : float (optional)
        Decoupled weight decay. Off by default except in AdamW.
    clipnorm : float (optional)
        Clip each parameter's gradient to this norm.
    global_clipnorm : float (optional)
        Clip all gradients jointly to this global norm.
    clipvalue : float (optional)
        Clip every gradient element to [-clipvalue, clipvalue].
    """
    default_weight_decay = None

    def __init__(
        self,
        params,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.9,
        epsilon=1e-12,
        weight_decay="default",
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
    ):
        super().__init__(params, dict(lazy=False))
        if weight_decay == "default":
            weight_decay = self.default_weight_decay
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue
        self.global_clipnorm = global_clipnorm
        self.iterations = 0

    def current_learning_rate(self):
        if callable(self.learning_rate):
            return self.learning_rate(self.iterations)
        return self.learning_rate

    def _clip(self, grads):
        if self.clipnorm is not None and self.clipnorm > 0:
            grads = [
                g * (self.clipnorm / torch.linalg.vector_norm(g).clamp_min(self.clipnorm))
                for g in grads
            ]
        if self.global_clipnorm is not None and self.global_clipnorm > 0:
            norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in grads])
            ) if grads else None
            if norm is not None:
                grads = [g * (self.global_clipnorm / norm.clamp_min(self.global_clipnorm)) for g in grads]
        if self.clipvalue is not None:
            grads = [g.clamp(-self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def _use_weight_decay(self, group):
        """Whether the dense, every-step decay of the base class applies."""
        return self.weight_decay is not None and self.weight_decay > 0

    def _lazy_weight_decay(self, group):
        """Whether to decay inside the masked update instead (AdamW only)."""
        return False

    def update(self, param, grad, m, v, alpha, lazy, lr, group):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params, groups = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
        grads = self._clip([p.grad for p in params])

        lr = self.current_learning_rate()
        t = self.iterations + 1
        alpha = lr * math.sqrt(1. - self.beta_2 ** t) / (1. - self.beta_1 ** t)

        for p, g, group in zip(params, grads, groups):
            lazy = group['lazy']
            if self._use_weight_decay(group):
                p.sub_(p * self.weight_decay * lr)
            state = self.state[p]
            if len(state) == 0:
                state['m'] = torch.zeros_like(p)
                state['v'] = torch.zeros_like(p)
            self.update(p, g, state['m'], state['v'], alpha, lazy, lr, group)

        self.iterations += 1
        return loss

    def state_dict(self):
        state = super().state_dict()
        state['iterations'] = self.iterations
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.iterations = state_dict.pop('iterations', 0)
        super().load_state_dict(state_dict)


class Adam(AbismalOptimizer):
    def update(self, p, g, m, v, alpha, lazy, lr, group):
        """Update step given gradient and the associated model variable."""
        b1, b2 = self.beta_1, self.beta_2
        if lazy:
            nonzero = g != 0.
            m.add_(torch.where(nonzero, (g - m) * (1 - b1), 0.))
            v.add_(torch.where(nonzero, (g * g - v) * (1 - b2), 0.))
            update = (m * alpha) / (torch.sqrt(v) + self.epsilon)
            if self._lazy_weight_decay(group):
                # Decoupled: uses the pre-update variable value, matching the
                # order the dense decay is applied relative to the step.
                update = update + p * self.weight_decay * lr
            p.sub_(torch.where(nonzero, update, 0.))
        else:
            m.add_((g - m) * (1 - b1))
            v.add_((g * g - v) * (1 - b2))
            p.sub_((m * alpha) / (torch.sqrt(v) + self.epsilon))


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, ICLR 2019).

    For dense parameters, the base class's every-step decoupled decay already
    makes `Adam(weight_decay=...)` AdamW, and this class adds nothing.

    What the base class gets wrong is the *lazy* parameters. It decays every
    parameter on every step, including entries whose gradient was zero only
    because the corresponding reflection was absent from the batch. Those
    entries would be pulled toward zero at a rate governed by how rarely they
    are observed rather than by the objective -- precisely the coupling
    between update frequency and parameter value that the lazy update exists
    to avoid, and worst for the least-observed parameters.

    So lazy groups opt out of the every-step decay and are instead decayed
    inside the update, masked by the same `nonzero` condition that gates the
    moment updates. A lazy parameter is therefore decayed once per *update*
    rather than once per *step*, which keeps decay and adaptation on the same
    clock. Note the consequence: a parameter updated on one step in ten decays
    ten times more slowly in wall-clock terms than a dense one at the same
    `weight_decay`.

    The `weight_decay` default of 0.004 matches `tf_keras.optimizers.AdamW`.
    Setting it to 0.0 or None recovers plain `Adam` exactly.
    """
    default_weight_decay = 0.004

    def _use_weight_decay(self, group):
        if group['lazy']:
            return False
        return super()._use_weight_decay(group)

    def _lazy_weight_decay(self, group):
        return self.weight_decay is not None and self.weight_decay > 0


class AdaBelief(AbismalOptimizer):
    def update(self, p, g, m, v, alpha, lazy, lr, group):
        """Update step given gradient and the associated model variable."""
        b1, b2 = self.beta_1, self.beta_2
        delta = g - m
        if lazy:
            nonzero = g != 0.
            m.add_(torch.where(nonzero, (1 - b1) * delta, 0.))
            v.add_(torch.where(nonzero, (b2 - 1.) * v + (1. - b2) * delta * delta, 0.))
            p.sub_(torch.where(nonzero, (m * alpha) / (torch.sqrt(v) + self.epsilon), 0.))
        else:
            m.add_((1 - b1) * delta)
            v.add_((b2 - 1.) * v + (1. - b2) * delta * delta)
            p.sub_((m * alpha) / (torch.sqrt(v) + self.epsilon))
