"""Model loading and tokenizer configuration with left-padding and ChatML support."""

from __future__ import annotations
import logging
from typing import Optional, Tuple, Any, Dict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from prismrunner.engine.device import get_optimal_device, get_default_dtype, clear_device_memory

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
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device, int]:
    """Load Hugging Face model and tokenizer configured for batch ChatML generation.
    
    Enforces:
    - tokenizer.padding_side = 'left' (critical for batched causal LM generation)
    - tokenizer.pad_token configured to eos_token if not set
    - Apple Silicon MPS, CUDA, or CPU device placement
    - Optional 4-bit / 8-bit quantization (where supported by bitsandbytes)
    """
    if device is None:
        device = get_optimal_device()

    selected_dtype = get_default_dtype(device, torch_dtype)
    logger.info(f"Loading tokenizer: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=token,
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
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = selected_dtype
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
    except Exception as e:
        logger.warning(f"Initial model load attempt failed ({e}). Retrying with direct CPU placement...")
        clear_device_memory()
        model_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

    # For MPS or CPU without device_map, explicitly place on target device
    if quantization_config is None and getattr(model, "device", None) != device:
        if device.type in ("mps", "cpu"):
            logger.info(f"Transferring model to {device}...")
            model = model.to(device)

    model.eval()
    context_window = get_context_window(model, tokenizer)
    logger.info(f"Model successfully loaded. Context window: {context_window} tokens.")

    return model, tokenizer, device, context_window
