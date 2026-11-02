"""Device management and memory handling for Apple Silicon MPS, CUDA, and CPU."""

from __future__ import annotations
import gc
import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


def validate_cuda_device(device: torch.device, raise_on_error: bool = False) -> bool:
    """Validate that CUDA kernels can execute on the target device.
    
    Catches CUDA compute capability mismatches ('CUDA error: no kernel image is available
    for execution on the device') before model weights are loaded.
    """
    if device.type != "cuda":
        return True

    try:
        # Test minimal kernel execution
        t = torch.zeros(1, device=device)
        _ = (t + 1).item()
        return True
    except RuntimeError as e:
        err_msg = str(e)
        if "no kernel image is available" in err_msg:
            dev_idx = device.index if device.index is not None else 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            cap = torch.cuda.get_device_capability(dev_idx)
            archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
            arch_str = ", ".join(archs) if archs else "unknown"
            diagnostic = (
                f"CUDA error: no kernel image is available for execution on the device.\n"
                f"CUDA kernel execution failed on '{dev_name}' (Compute Capability: {cap[0]}.{cap[1]} / sm_{cap[0]}{cap[1]}).\n"
                f"The installed PyTorch binary ({torch.__version__}) was compiled for architectures: [{arch_str}], "
                f"which does not include sm_{cap[0]}{cap[1]}.\n\n"
                "To resolve this, reinstall PyTorch matching your CUDA version:\n"
                "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
                "  # or for CUDA 12.4:\n"
                "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
                "  # or for older CUDA 11.8:\n"
                "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
                "Alternatively, run on CPU with: --device cpu"
            )
            logger.error(diagnostic)
            if raise_on_error:
                raise RuntimeError(diagnostic) from e
            return False

        if raise_on_error:
            raise
        logger.warning(f"CUDA device validation failed: {err_msg}")
        return False


def get_optimal_device(preferred: Optional[str] = None) -> torch.device:
    """Detect and return the optimal available PyTorch device.
    
    Priority:
    1. User preferred device (if valid and available)
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Silicon GPU)
    4. CPU fallback
    """
    if preferred:
        pref = preferred.lower().strip()
        if pref.startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(pref)
            validate_cuda_device(dev, raise_on_error=True)
            return dev
        elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif pref == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(
                f"Requested device '{preferred}' is not available. Falling back to auto-detection."
            )

    if torch.cuda.is_available():
        cuda_dev = torch.device("cuda")
        if validate_cuda_device(cuda_dev, raise_on_error=False):
            logger.info(f"Using NVIDIA CUDA device: {torch.cuda.get_device_name(0)}")
            return cuda_dev
        else:
            logger.warning("CUDA device is available but cannot execute kernels; falling back to MPS or CPU.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS device")
        return device
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
        return device


def get_default_dtype(device: torch.device, requested_dtype: Optional[str] = None) -> torch.dtype:
    """Return the optimal torch dtype for the selected device."""
    if requested_dtype:
        dt = requested_dtype.lower().strip()
        if dt in ("bfloat16", "bf16"):
            return torch.bfloat16
        elif dt in ("float16", "fp16"):
            return torch.float16
        elif dt in ("float32", "fp32"):
            return torch.float32

    if device.type == "cuda":
        # Check if CUDA device supports bfloat16 (Ampere / 8.0+)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        # Apple Silicon MPS has mature FP16 support
        return torch.float16
    else:
        return torch.float32


def clear_device_memory(device: Optional[torch.device] = None) -> None:
    """Flush garbage collector and empty accelerator device cache."""
    gc.collect()
    if device is None or device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if device is None or device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception as e:
                logger.debug(f"mps.empty_cache() encountered: {e}")


def get_memory_info(device: torch.device) -> Dict[str, Any]:
    """Retrieve memory statistics for the current device."""
    info: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        info.update({
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_mb": round(total, 2),
            "device_name": torch.cuda.get_device_name(device),
        })
    elif device.type == "mps":
        info.update({
            "device_name": "Apple Silicon (MPS)",
            "unified_memory": True,
        })
    else:
        info.update({
            "device_name": "CPU Host",
        })
    return info
