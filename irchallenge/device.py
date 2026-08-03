"""Torch device selection, with a CPU fallback if MPS isn't usable."""


def get_device(preferred="mps"):
    import torch
    if preferred == "mps" and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    return "cpu"
