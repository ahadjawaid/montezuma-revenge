import torch
from pathlib import Path

def get_device(device: str = None):
    return torch.device(
        device if device else "cuda" if torch.cuda.is_available() else "mps"
        if torch.backends.mps.is_available() else "cpu"
    )

def get_sorted_checkpoints(checkpoint_dir: str):
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir_path.glob("*.pt"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    return checkpoints