"""Choosing the PyTorch device abismal trains on."""
import torch


def list_devices():
    devices = ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def select_device(gpu_id=0, device=None):
    """
    The device named by `device` if given, else GPU `gpu_id` when CUDA (or
    Apple's MPS) is available, else the CPU.
    """
    if device is not None:
        return torch.device(device)
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu-id {gpu_id} requested but only {torch.cuda.device_count()} GPU(s) are visible"
            )
        return torch.device(f"cuda:{gpu_id}")
    if gpu_id is not None and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure(device):
    """
    Settings that make PyTorch behave like the TensorFlow abismal ran on before.

    TensorFlow ran float32 matmuls in TF32 on Ampere and newer GPUs by default;
    PyTorch does not, and without it the scale network runs several times
    slower for no accuracy abismal can use.
    """
    torch.set_float32_matmul_precision("high")
