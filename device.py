import warnings
from contextlib import nullcontext

import torch


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
    - CUDA: prefer natively supported bfloat16; otherwise use float16 with GradScaler.
    - MPS: bfloat16 only when a probe verifies it; otherwise no autocast (fp32).
    - CPU: no autocast.
    """
    if device.type == 'cuda':
        try:
            if hasattr(torch.cuda, 'is_bf16_supported'):
                try:
                    # torch >= 2.3 defaults to including_emulation=True, which
                    # reports True on every CUDA GPU. Emulated bf16 on pre-Ampere
                    # hardware is several times slower than fp32, so ask for
                    # native support only.
                    bf16_ok = torch.cuda.is_bf16_supported(including_emulation=False)
                except TypeError:
                    # Older torch has no such kwarg and never counted emulation.
                    bf16_ok = torch.cuda.is_bf16_supported()
                if bf16_ok:
                    return {'device_type': 'cuda', 'dtype': torch.bfloat16, 'use_scaler': False}
        except Exception:
            pass
        return {'device_type': 'cuda', 'dtype': torch.float16, 'use_scaler': True}

    if device.type == 'mps':
        # torch.autocast with an unsupported dtype only warns and disables itself,
        # and elementwise ops are never autocast-eligible, so an exception probe
        # cannot fail: run a matmul and check the dtype it actually produced.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with torch.autocast('mps', dtype=torch.bfloat16):
                    out = torch.mm(torch.ones(1, 1, device=device), torch.ones(1, 1, device=device))
            if out.dtype == torch.bfloat16:
                return {'device_type': 'mps', 'dtype': torch.bfloat16, 'use_scaler': False}
        except Exception:
            pass
        # No verified bf16: run full fp32. fp16 without a GradScaler would risk
        # silent under/overflow, so it is not offered as a fallback.
        return {'device_type': 'mps', 'dtype': None, 'use_scaler': False}

    return {'device_type': 'cpu', 'dtype': None, 'use_scaler': False}


def get_autocast_cm(amp_conf: dict):
    """Return an autocast context manager based on amp_conf dict."""
    dtype = amp_conf.get('dtype')
    if dtype is None:
        return nullcontext()
    return torch.autocast(amp_conf['device_type'], dtype=dtype)


