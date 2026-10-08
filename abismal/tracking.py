"""
Losses and metrics reported from inside a forward pass.

This replaces Keras' `add_loss` and `add_metric`. Deeply nested modules -- the
scale model's KL, the standardization layers' running moments, the likelihood's
correlation coefficients -- report what they computed as a side effect, and the
trainer collects it around the call:

```
with tracking.track() as tracker:
    model(batch, training=True)
loss = tracker.total_loss()
```

Outside a `track()` block the calls are no-ops, so a module can be used on its
own (in a test, say) without any setup.
"""
from contextlib import contextmanager


class Tracker:
    def __init__(self):
        self.losses = []
        self.metrics = {}

    def total_loss(self):
        return sum(self.losses)


_stack = []


@contextmanager
def track():
    tracker = Tracker()
    _stack.append(tracker)
    try:
        yield tracker
    finally:
        _stack.pop()


def add_loss(value):
    """Add `value` to the objective being minimized."""
    if _stack:
        _stack[-1].losses.append(value)


def add_metric(value, name):
    """Report `value` under `name`. It is detached; metrics never carry gradients."""
    if _stack:
        if hasattr(value, "detach"):
            value = value.detach()
        _stack[-1].metrics[name] = value
