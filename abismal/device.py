"""Choosing the PyTorch device abismal trains on."""
import os
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

    On CUDA it also switches the caching allocator to expandable segments,
    unless PYTORCH_CUDA_ALLOC_CONF already says otherwise. Every batch holds a
    different number of reflections, so blocks freed by one step rarely fit
    the next and the default allocator fragments: on cxidb_81 a run reserving
    10 GB was using 1.6 GB. Expandable segments grow and reuse one mapping
    instead. Must run before the first allocation on the device.
    """
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda" and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        # Private, so tolerate a torch that has moved it; the default allocator
        # only costs memory, not correctness.
        setter = getattr(torch.cuda.memory, "_set_allocator_settings", None)
        if setter is not None:
            setter("expandable_segments:True")


def resolve_compile(choice, device):
    """Whether to torch.compile for the --compile setting `choice` on `device`."""
    if choice in (True, "always"):
        return True
    if choice in (False, None, "never"):
        return False
    return device.type == "cuda"


def supports_bfloat16(device):
    """Whether `device` runs bfloat16 natively, rather than by emulation."""
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older torch has no including_emulation, and reports emulation too
        return torch.cuda.get_device_capability(device)[0] >= 8


def resolve_precision(choice, device):
    """The precision for the --precision setting `choice` on `device`."""
    if choice == "auto":
        return "bfloat16" if supports_bfloat16(device) else "float32"
    return choice
