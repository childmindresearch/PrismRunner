"""Model loading and tokenizer configuration with left-padding and ChatML support."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from prismrunner.engine.device import (
    get_optimal_device,
    get_default_dtype,
    clear_device_memory,
    validate_cuda_device,
)

logger = logging.getLogger(__name__)


def get_context_window(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> int:
    """Determine the maximum context window supported by the model and tokenizer."""
    # Check model config for standard context window attributes
    for attr in (
        "max_position_embeddings",
        "seq_length",
        "n_positions",
        "max_sequence_length",
    ):
        val = getattr(model.config, attr, None)
        if val and isinstance(val, int) and 512 <= val <= 1_000_000:
            return val

    # Check tokenizer max length (guarding against huge default sentinel values like 1e30)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and isinstance(tok_max, int) and 512 <= tok_max <= 1_000_000:
        return tok_max

    # Safe fallback
    return 4096


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: Optional[torch.device] = None,
    torch_dtype: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
    attn_implementation: Optional[str] = "sdpa",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device, int]:
    """Load Hugging Face model and tokenizer configured for batch ChatML generation.
    
    Enforces:
    - tokenizer.padding_side = 'left' (critical for batched causal LM generation)
    - tokenizer.pad_token configured to eos_token if not set
    - Apple Silicon MPS, CUDA, or CPU device placement
    - Optional 4-bit / 8-bit quantization (where supported by bitsandbytes)
    - Explicit local cache directory verification and offline execution support
    """
    try:
        from transformers.utils import is_torch_available
        if not is_torch_available():
            installed_torch_ver = getattr(torch, "__version__", "unknown")
            raise RuntimeError(
                f"PyTorch is disabled by the installed transformers library (found PyTorch {installed_torch_ver}). "
                "This occurs when transformers enforces PyTorch >= 2.6. "
                "To resolve, run: pip install 'transformers<4.48.0' or upgrade PyTorch: pip install --upgrade 'torch>=2.6.0'"
            )
    except ImportError:
        pass

    if device is None:
        device = get_optimal_device()
    else:
        validate_cuda_device(device, raise_on_error=True)

    selected_dtype = get_default_dtype(device, torch_dtype)

    if device.type != "cuda" and "awq" in model_name_or_path.lower():
        raise ValueError(
            f"The requested model '{model_name_or_path}' is an AWQ-quantized checkpoint which requires an NVIDIA CUDA GPU.\n"
            f"The current hardware device is '{device.type}'.\n"
            f"To run on Apple Silicon (MPS) or CPU, please pass an unquantized model, e.g.:\n"
            f"  python SP/run_stage_1.py --model Qwen/Qwen2.5-3B-Instruct --device auto\n"
            f"Or run this pipeline directly on your remote NVIDIA GPU server."
        )

    # Verify and log cache status
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        default_hub_cache = Path(HUGGINGFACE_HUB_CACHE)
    except Exception:
        default_hub_cache = Path.home() / ".cache" / "huggingface" / "hub"

    effective_cache = Path(cache_dir).resolve() if cache_dir else default_hub_cache
    model_path = Path(model_name_or_path)

    if model_path.exists() and model_path.is_dir():
        print(f"-> Loading model weights from local path: {model_path.resolve()}", flush=True)
        logger.info(f"Loading model weights from local path: {model_path.resolve()}")
    else:
        org_repo_slug = "models--" + model_name_or_path.replace("/", "--")
        cached_repo_dir = effective_cache / org_repo_slug
        if cached_repo_dir.exists():
            print(f"-> Model verified in local cache: {cached_repo_dir.name} (loading weights onto {device}...)", flush=True)
            logger.info(
                f"Model verified in local cache: {cached_repo_dir} "
                f"(loading from local storage; no download needed)."
            )
        else:
            print(f"-> Model not found in cache ({effective_cache}). Downloading '{model_name_or_path}' from Hugging Face Hub...", flush=True)
            logger.info(
                f"Model not found in local cache ({effective_cache}). "
                f"Downloading weights from Hugging Face Hub (saved for all future runs)..."
            )

    logger.info(f"Loading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )

    # CRITICAL: Left-padding is mandatory for batched decoder-only generation
    tokenizer.padding_side = "left"

    # Configure pad token if not present
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Assigned pad_token to eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    quantization_config = None
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=selected_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("Using 4-bit NF4 quantization")
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization")
        except ImportError:
            logger.warning(
                "bitsandbytes is not installed. Quantization disabled; falling back to full precision."
            )

    logger.info(f"Loading model: {model_name_or_path} with dtype={selected_dtype} onto device={device}")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = selected_dtype
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

    # Item 2: Standardized native PyTorch SDPA attention (zero external dependencies)
    chosen_attn = attn_implementation if attn_implementation is not None else "sdpa"
    if chosen_attn:
        model_kwargs["attn_implementation"] = chosen_attn
        logger.info(f"Using standardized attention implementation: {chosen_attn}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
    except Exception as e:
        logger.warning(f"Initial model load attempt failed: {e}")
        # If attention implementation caused the failure, retry with default attention
        if "attn_implementation" in model_kwargs:
            logger.info("Retrying model load without explicit attn_implementation...")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs,
            )
        else:
            raise e

    # Ensure model is on the target device if device_map was not used
    if not hasattr(model, "hf_device_map") and quantization_config is None:
        try:
            curr_dev = next(model.parameters()).device
            if curr_dev != device:
                logger.info(f"Transferring model to {device}...")
                model = model.to(device)
        except Exception:
            pass

    model.eval()
    context_window = get_context_window(model, tokenizer)
    logger.info(f"Model successfully loaded. Context window: {context_window} tokens.")

    # Determine primary model device (crucial for device_map="auto" multi-GPU or quantized placements)
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except Exception:
            model_device = device

    return model, tokenizer, model_device, context_window


def init_model_and_tokenizer(model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Universal direct initialization adhering to Item 2 specification.

    Routes device dynamically to 'cuda', 'mps', or 'cpu', selects optimal precision,
    and initializes with standardized native PyTorch SDPA attention.
    """
    from prismrunner.engine.device import resolve_device_and_dtype
    device_name, dtype = resolve_device_and_dtype()
    device = torch.device(device_name)
    model, tokenizer, _, _ = load_model_and_tokenizer(
        model_name_or_path=model_path,
        device=device,
        torch_dtype=str(dtype).replace("torch.", ""),
        attn_implementation="sdpa",
    )
    return model, tokenizer
