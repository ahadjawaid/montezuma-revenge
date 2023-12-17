import torch


def get_device(device: str = None):
    return torch.device(
        device if device else "cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )
