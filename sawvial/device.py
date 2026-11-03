"""GPU selection. SAWVIAL is meant to run on the NVIDIA GPU: by default (SAWVIAL_DEVICE=cuda) it refuses to start
on CPU, because the most common cause is a CPU-only PyTorch build installed by a plain `pip install torch`, which
would silently make every answer many times slower."""
from __future__ import annotations

import logging
from typing import Any, Dict

from sawvial import settings
from sawvial.model_config import ModelConfigError

log = logging.getLogger("sawvial.device")
CUDA_INSTALL = (r".\.venv\Scripts\python.exe -m pip install --force-reinstall torch "
                r"--index-url https://download.pytorch.org/whl/cu128")


def pick_device(torch_module=None) -> Dict[str, Any]:
    """-> {"device": "cuda"|"cpu", "name", "vram_mb", "torch", "torch_cuda"}. Raises ModelConfigError when a GPU is
    required (default) but PyTorch cannot use one."""
    torch = torch_module
    if torch is None:
        try:
            import torch
        except ImportError as exc:
            raise ModelConfigError("PyTorch is not installed. Run the setup snippet in the README.") from exc
    want = settings.DEVICE
    info = {"device": "cpu", "name": "CPU", "vram_mb": None, "torch": getattr(torch, "__version__", "?"),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None)}
    if want == "cpu":
        return info
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(device="cuda", name=props.name, vram_mb=int(props.total_memory // (1024 * 1024)))
        log.info("Using GPU: %s (%d MB VRAM), torch %s, CUDA %s", info["name"], info["vram_mb"], info["torch"], info["torch_cuda"])
        return info
    if info["torch_cuda"] is None:
        why = f"this PyTorch ({info['torch']}) is the CPU-only build"
    else:
        why = (f"PyTorch {info['torch']} (CUDA {info['torch_cuda']}) cannot see a GPU - update the NVIDIA driver "
               f"and check `nvidia-smi` works")
    if want == "auto":
        log.warning("No usable GPU (%s); running on CPU (slow).", why)
        return info
    raise ModelConfigError(
        f"No usable GPU: {why}. Install the CUDA build of PyTorch:\n  {CUDA_INSTALL}\n"
        f"(set SAWVIAL_DEVICE=auto only if you really want to run on the CPU)")
