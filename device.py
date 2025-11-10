import torch
from contextlib import nullcontext


def detect_device() -> torch.device:
    """Detect the best available device. 
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


DEVICE = detect_device()


def select_amp(device: torch.device):
    """Return a dict with autocast device_type, dtype, and whether GradScaler is needed.

    Policy:
    - CUDA: prefer bfloat16 if supported; otherwise use float16 with GradScaler.
    - MPS: try bfloat16, fallback to float16.
    - CPU: no autocast.
    """
    if device.type == 'cuda':
        try:
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                return {'device_type': 'cuda', 'dtype': torch.bfloat16, 'use_scaler': False}
        except Exception:
            pass
        return {'device_type': 'cuda', 'dtype': torch.float16, 'use_scaler': True}

    if device.type == 'mps':
        # Probe bfloat16 support in autocast; fallback to float16
        try:
            with torch.autocast('mps', dtype=torch.bfloat16):
                _ = (torch.ones(1, device=device) * 1.0)
            return {'device_type': 'mps', 'dtype': torch.bfloat16, 'use_scaler': False}
        except Exception:
            return {'device_type': 'mps', 'dtype': torch.float16, 'use_scaler': False}

    return {'device_type': 'cpu', 'dtype': None, 'use_scaler': False}


def get_autocast_cm(amp_conf: dict):
    """Return an autocast context manager based on amp_conf dict."""
    dtype = amp_conf.get('dtype')
    if dtype is None:
        return nullcontext()
    return torch.autocast(amp_conf['device_type'], dtype=dtype)


