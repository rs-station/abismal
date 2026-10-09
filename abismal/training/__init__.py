"""
The training loop that replaces keras' `Model.fit`.

It keeps the parts of keras' behavior the rest of abismal depends on:
`Epoch i/N` lines on stdout (the notebook GUI drives its progress bar from
them), per-epoch metric logs named the way `history.csv` has always named them,
`val_` prefixed validation metrics, and callbacks with keras' hook names.
"""
import functools
import math
import sys
import time
import warnings
import torch
from abismal import tracking
from abismal.callbacks.base import Callback


def parameter_groups(model):
    """
    The optimizer's parameter groups: the surrogate posterior, which is updated
    lazily (see abismal.optimizers), and everything else.
    """
    groups = model.parameter_groups()
    lazy = groups.pop("q") + groups.pop("q_half", [])
    dense = [p for params in groups.values() for p in params]
    out = []
    if dense:
        out.append({"params": dense, "lazy": False})
    if lazy:
        out.append({"params": lazy, "lazy": True})
    return out


def grad_rms(grads):
    """
    Root-mean-square gradient magnitude over a group of parameters.

    Returns 0 for an empty group rather than nan: a frozen component
    (--freeze-scales, --freeze-posterior, abismal.cchalf) has no trainable
    parameters and would otherwise report a nan gradient norm.
    """
    if len(grads) == 0:
        return torch.zeros(())
    return torch.sqrt(torch.stack([g.square().mean() for g in grads]).mean())


def piecewise_constant(boundary, before, after):
    """keras' PiecewiseConstantDecay with one boundary, as a function of step."""
    def learning_rate(step):
        return before if step <= boundary else after
    return learning_rate


class EarlyStopping(Callback):
    """Stop training when `monitor` has not improved for `patience` epochs."""
    def __init__(self, monitor="val_loss", patience=0, mode="min", min_delta=0.):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)
        self.best = None
        self.wait = 0
        self._warned = False

    def _improved(self, value):
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor not in logs:
            if not self._warned:
                warnings.warn(
                    f"Early stopping conditioned on metric `{self.monitor}` which is "
                    f"not available. Available metrics are: {','.join(logs)}"
                )
                self._warned = True
            return
        value = logs[self.monitor]
        if self._improved(value):
            self.best = value
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.trainer.stop_training = True


class _RunningMeans:
    """
    Per-epoch means of metric tensors, kept on the device until read.

    Each batch counts in proportion to its `weight`, the number of images in
    it. Keras averaged its metrics over batches unweighted, which let a short
    final batch count as much as a full one: with three batches of 100
    validation images and a remainder of 2, the reported validation NLL of a
    fixed model moved by +-1.2 from one reshuffle to the next, while the
    image-weighted mean did not move at all.
    """
    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(self, metrics, weight=1):
        for k, v in metrics.items():
            v = v.detach().float() * weight
            self.sums[k] = self.sums[k] + v if k in self.sums else v
        self.count += weight

    def result(self, prefix=""):
        if self.count == 0:
            return {}
        values = torch.stack([v.reshape(()) for v in self.sums.values()]).cpu() / self.count
        return {prefix + k: float(v) for k, v in zip(self.sums, values)}


def _loss_and_metrics(model, batch, training):
    """The objective for one batch, and the metrics reported while computing it."""
    with tracking.track() as tracker:
        model(batch, training=training)
    return tracker.total_loss(), dict(tracker.metrics)


def _format_logs(logs):
    return " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())


class Trainer:
    """
    Parameters
    ----------
    model : VariationalMergingModel
    optimizer : abismal.optimizers.AbismalOptimizer
    device : torch.device or str (optional)
    callbacks : list (optional)
    verbose : int (optional)
        0 is silent, 1 shows a progress bar, 2 prints one line per epoch.
    compile : dict (optional)
        Keyword arguments for `torch.compile`, applied to the loss computation
        of the training and validation steps. None (the default) runs eagerly.
    """
    def __init__(self, model, optimizer, device=None, callbacks=None, verbose=1, compile=None):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.callbacks = list(callbacks) if callbacks is not None else []
        self.verbose = verbose
        self._train_loss = functools.partial(_loss_and_metrics, model, training=True)
        self._test_loss = functools.partial(_loss_and_metrics, model, training=False)
        if compile is not None:
            # What is compiled returns only the loss and the (detached) metrics.
            # Compiling the model itself also made its unused prediction an
            # output of the graph, and AOTAutograd's backward then failed on
            # that output's missing gradient.
            self._train_loss = torch.compile(self._train_loss, **compile)
            self._test_loss = torch.compile(self._test_loss, **compile)
        self.history = {}
        self.stop_training = False
        for callback in self.callbacks:
            callback.set_trainer(self)

    def transfer(self, batch):
        """Move a numpy `Batch` to the training device. Runs on the prefetch thread."""
        pin = self.device.type == "cuda"
        def move(a):
            t = torch.as_tensor(a)
            if pin:
                t = t.pin_memory()
            return t.to(self.device, non_blocking=pin)
        return batch.map(move)

    def _batches(self, stream, persistent, attr):
        """
        An iterator over `stream`. A repeating stream keeps one iterator across
        epochs, so its shuffle buffer spans passes as tf.data's did; otherwise
        each epoch makes a fresh pass.
        """
        if persistent:
            it = getattr(self, attr, None)
            if it is None:
                it = stream.iterate(transfer=self.transfer, prefetch=2)
                setattr(self, attr, it)
            return it
        return stream.iterate(transfer=self.transfer, prefetch=2)

    def train_step(self, batch):
        self.model.train()
        loss, tracked = self._train_loss(batch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        metrics = {"loss": loss.detach()}
        metrics.update(tracked)
        labels = {"scale": "|∇s|", "q": "|∇q|", "ll": "|∇ll|", "p": "|∇p|", "q_half": "|∇q½|"}
        for name, params in self.model.parameter_groups().items():
            grads = [p.grad for p in params if p.grad is not None]
            # Like keras, the likelihood and prior only report a gradient norm
            # when they have something to train.
            if name in ("scale", "q") or len(grads) > 0:
                metrics[labels[name]] = grad_rms(grads).to(loss.device)
            for g in grads:
                torch.nan_to_num_(g, nan=0., posinf=0., neginf=0.)

        self.optimizer.step()
        return metrics

    @torch.no_grad()
    def test_step(self, batch):
        self.model.eval()
        loss, tracked = self._test_loss(batch)
        metrics = {"loss": loss.detach()}
        metrics.update(tracked)
        return metrics

    def _run(self, iterator, steps, step_fn, progress):
        means = _RunningMeans()
        i = 0
        last_print = 0.
        for batch in iterator:
            means.update(step_fn(batch), weight=batch.num_images)
            i += 1
            if progress is not None and (time.time() - last_print > 0.5 or i == steps):
                last_print = time.time()
                progress(i, means)
            if steps is not None and i >= steps:
                break
        return means, i

    def _progress(self, steps, start):
        if self.verbose != 1:
            return None
        def show(i, means):
            total = steps if steps is not None else "?"
            logs = means.result()
            elapsed = time.time() - start
            sys.stdout.write(f"\r{i}/{total} - {elapsed:.0f}s - {_format_logs(logs)}")
            sys.stdout.flush()
        return show

    def fit(self, train, epochs=1, steps_per_epoch=None, validation_data=None, validation_steps=None):
        """
        Parameters
        ----------
        train : abismal.io.pipeline.BatchStream
        epochs : int
        steps_per_epoch : int (optional)
            Required when `train` repeats; otherwise each epoch is one pass.
        validation_data : abismal.io.pipeline.BatchStream (optional)
        validation_steps : int (optional)

        Returns
        -------
        history : dict
            Per-epoch values of each logged metric.
        """
        if train.repeat and steps_per_epoch is None:
            raise ValueError("A repeating training stream needs steps_per_epoch")
        self.stop_training = False
        for callback in self.callbacks:
            callback.on_train_begin()

        for epoch in range(epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{epochs}", flush=True)

            start = time.time()
            iterator = self._batches(train, train.repeat, "_train_iter")
            means, steps = self._run(
                iterator, steps_per_epoch, self.train_step,
                self._progress(steps_per_epoch, start),
            )
            if not train.repeat:
                iterator.close()
            logs = means.result()

            if validation_data is not None:
                val_iter = self._batches(validation_data, validation_data.repeat, "_val_iter")
                val_means, _ = self._run(val_iter, validation_steps, self.test_step, None)
                if not validation_data.repeat:
                    val_iter.close()
                logs.update(val_means.result(prefix="val_"))

            if self.verbose == 1:
                sys.stdout.write(f"\r{steps}/{steps} - {time.time() - start:.0f}s - {_format_logs(logs)}\n")
                sys.stdout.flush()
            elif self.verbose == 2:
                print(f"{steps}/{steps} - {time.time() - start:.0f}s - {_format_logs(logs)}", flush=True)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
            if self.stop_training:
                break

        for callback in self.callbacks:
            callback.on_train_end()
        self.close()
        return self.history

    def close(self):
        for attr in ("_train_iter", "_val_iter"):
            it = getattr(self, attr, None)
            if it is not None:
                it.close()
                setattr(self, attr, None)
