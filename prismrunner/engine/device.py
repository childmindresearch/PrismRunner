"""Device management and memory handling for Apple Silicon MPS, CUDA, and CPU."""

from __future__ import annotations
import gc
import logging
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


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
            return torch.device(pref)
        elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif pref == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(
                f"Requested device '{preferred}' is not available. Falling back to auto-detection."
            )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
