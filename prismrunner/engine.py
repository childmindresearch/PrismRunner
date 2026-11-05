"""Hardware device detection, model loading, context budgeting, and ChatML batch inference engine."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Deterministic Greedy Decoding preset for structured extraction (KV-cache enabled)
EXTRACTION_GEN_CONFIG: Dict[str, Any] = {
    "do_sample": False,
    "use_cache": True,
}


# ==============================================================================
# 1. Device, Precision & Hardware Acceleration
# ==============================================================================

def resolve_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Automated hardware and precision detector.

    Returns:
        (device_name, dtype):
        - ("cuda", torch.bfloat16 on Ampere+ else torch.float16)
        - ("mps", torch.float16)
        - ("cpu", torch.float32)
    """
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def configure_hardware_backends(device: torch.device) -> None:
    """Enable SOTA PyTorch hardware backend optimizations (TF32 matmul, cuDNN benchmark, high-precision matmul)."""
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if device.type == "cuda" and torch.cuda.is_available():
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
    except Exception as e:
        logger.debug(f"Hardware backend tuning notice: {e}")


def validate_cuda_device(device: torch.device, raise_on_error: bool = False) -> bool:
    """Validate that CUDA kernels can execute on the target device before loading weights."""
    if device.type != "cuda":
        return True

    try:
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
                "To resolve this, reinstall PyTorch matching your CUDA version or run on CPU with device='cpu'."
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
    """Detect and return the optimal available PyTorch device (CUDA -> MPS -> CPU)."""
    if preferred and preferred.lower().strip() != "auto":
        pref = preferred.lower().strip()
        if pref.startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(pref)
            validate_cuda_device(dev, raise_on_error=True)
            configure_hardware_backends(dev)
            return dev
        elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            configure_hardware_backends(dev)
            return dev
        elif pref == "cpu":
            return torch.device("cpu")
        else:
            logger.warning(f"Requested device '{preferred}' is not available. Falling back to auto-detection.")

    if torch.cuda.is_available():
        cuda_dev = torch.device("cuda")
        if validate_cuda_device(cuda_dev, raise_on_error=False):
            configure_hardware_backends(cuda_dev)
            return cuda_dev
        logger.warning("CUDA device is available but cannot execute kernels; falling back to MPS or CPU.")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_dev = torch.device("mps")
        configure_hardware_backends(mps_dev)
        return mps_dev
    return torch.device("cpu")


def get_default_dtype(device: torch.device, requested_dtype: Optional[str] = None) -> torch.dtype:
    """Return the optimal torch dtype for the selected device."""
    if requested_dtype and requested_dtype.lower().strip() != "auto":
        dt = requested_dtype.lower().strip()
        if dt in ("bfloat16", "bf16"):
            return torch.bfloat16
        elif dt in ("float16", "fp16"):
            return torch.float16
        elif dt in ("float32", "fp32"):
            return torch.float32

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_optimal_attention(
    device: torch.device,
    dtype: torch.dtype,
    requested_attn: Optional[str] = "auto",
) -> str:
    """Dynamically select the fastest supported attention backend (FlashAttention-2 -> SDPA -> eager)."""
    req = (requested_attn or "auto").lower().strip()
    if req in ("eager", "sdpa"):
        return req

    if req in ("auto", "flash_attention_2"):
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except ImportError:
                if req == "flash_attention_2":
                    logger.info("flash-attn not installed; falling back to native PyTorch SDPA.")
        return "sdpa"

    return "sdpa"


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
            except Exception:
                pass


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
        info.update({"device_name": "Apple Silicon (MPS)", "unified_memory": True})
    else:
        info.update({"device_name": "CPU Host"})
    return info


# ==============================================================================
# 2. Dynamic Context Budgeting & Hardware Advisories
# ==============================================================================

@dataclass
class ContextBudget:
    """Calculates and enforces token reservations across the model's context window."""

    model_context_limit: int
    max_new_tokens: int = 512
    system_prompt_tokens: int = 64
    template_tokens: int = 128
    safety_buffer: int = 128

    @property
    def reserved_tokens(self) -> int:
        return self.max_new_tokens + self.system_prompt_tokens + self.template_tokens + self.safety_buffer

    @property
    def max_input_tokens(self) -> int:
        available = self.model_context_limit - self.reserved_tokens
        if available <= 0:
            raise ValueError(
                f"Context budget exceeded: model context limit ({self.model_context_limit}) "
                f"is smaller than reserved tokens ({self.reserved_tokens})."
            )
        return available

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Optional[PreTrainedTokenizerBase],
        context_limit: int,
        system_prompt: Optional[str] = None,
        template_text: Optional[str] = None,
        max_new_tokens: int = 512,
        safety_buffer: Optional[int] = None,
    ) -> "ContextBudget":
        if tokenizer is not None:
            sys_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False)) if system_prompt else 32
            tmpl_tokens = len(tokenizer.encode(template_text, add_special_tokens=False)) if template_text else 96
        else:
            sys_tokens = max(32, len(system_prompt) // 4) if system_prompt else 32
            tmpl_tokens = max(96, len(template_text) // 4) if template_text else 96

        resolved_buffer = safety_buffer if safety_buffer is not None else min(128, max(32, context_limit // 32))
        return cls(
            model_context_limit=context_limit,
            max_new_tokens=max_new_tokens,
            system_prompt_tokens=sys_tokens,
            template_tokens=tmpl_tokens,
            safety_buffer=resolved_buffer,
        )


def generate_hardware_advisory(
    model_name: str,
    context_limit: int,
    device: torch.device,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """Provide hardware-specific recommendations for batch sizes, precision, and chunking."""
    budget = ContextBudget(model_context_limit=context_limit, max_new_tokens=max_new_tokens)
    safe_input = budget.max_input_tokens

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)
        if vram_gb >= 24:
            rec_batch = 16
            rec_chunk = min(safe_input, max(2048, context_limit // 2))
            rec_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif vram_gb >= 12:
            rec_batch = 8
            rec_chunk = min(safe_input, max(1536, context_limit // 2))
            rec_dtype = "float16"
        else:
            rec_batch = 4
            rec_chunk = min(safe_input, max(1024, context_limit // 4))
            rec_dtype = "float16"
    elif device.type == "mps":
        rec_batch = 4
        rec_chunk = min(safe_input, max(1024, context_limit // 4))
        rec_dtype = "float16"
    else:
        rec_batch = 1
        rec_chunk = min(safe_input, max(512, context_limit // 8))
        rec_dtype = "float32"

    return {
        "model_name": model_name,
        "device": str(device),
        "context_limit": context_limit,
        "recommended_dtype": rec_dtype,
        "recommended_batch_size": rec_batch,
        "recommended_chunk_size": rec_chunk,
        "max_safe_input_tokens": safe_input,
        "reserved_budget_tokens": budget.reserved_tokens,
    }


def resolve_dynamic_batch_size(
    device: Optional[torch.device] = None,
    requested_batch_size: Optional[Union[int, str]] = "auto",
    items: Optional[List[Dict[str, Any]]] = None,
    text_key: str = "prompt",
    max_new_tokens: int = 512,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_batch_size: int = 32,
) -> int:
    """Dynamically compute the optimal concurrent batch size from live hardware memory and payload token length."""
    if isinstance(requested_batch_size, int) and requested_batch_size > 0:
        return requested_batch_size
    if isinstance(requested_batch_size, str) and requested_batch_size.strip().isdigit():
        val = int(requested_batch_size.strip())
        if val > 0:
            return val

    # Estimate 90th-percentile prompt token length across queued items
    est_prompt_tokens = 512
    if items:
        sample = items if len(items) <= 128 else items[:: max(1, len(items) // 128)]
        lengths: List[int] = []
        for item in sample:
            txt = str(item.get(text_key) or item.get("text") or "")
            if tokenizer is not None:
                try:
                    ids = tokenizer.encode(txt, add_special_tokens=False)
                    if isinstance(ids, list) and ids:
                        lengths.append(len(ids))
                        continue
                except Exception:
                    pass
            lengths.append(max(1, int(len(txt.split()) * 1.35)))
        if lengths:
            lengths.sort()
            p90_idx = min(len(lengths) - 1, int(len(lengths) * 0.9))
            est_prompt_tokens = lengths[p90_idx]

    est_seq_tokens = est_prompt_tokens + max(64, max_new_tokens) + 128
    dev_type = device.type if device is not None else "cpu"

    if dev_type == "cuda" and torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
            usable_gb = (free_bytes / (1024 ** 3)) * 0.75
        except Exception:
            try:
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                allocated_bytes = torch.cuda.memory_allocated(device)
                usable_gb = max(1.0, ((total_bytes - allocated_bytes) / (1024 ** 3)) * 0.75)
            except Exception:
                usable_gb = 6.0

        per_seq_gb = max(0.12, est_seq_tokens * 0.00018)
        return max(1, min(max_batch_size, int(usable_gb // per_seq_gb)))

    if dev_type == "mps":
        try:
            total_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
        except Exception:
            total_ram_gb = 16.0
        if est_seq_tokens >= 2400:
            return 4 if total_ram_gb >= 32 else 2
        elif est_seq_tokens >= 1200:
            return 6 if total_ram_gb >= 32 else 4
        return 8 if total_ram_gb >= 32 else 4

    if est_seq_tokens >= 1600:
        return 1
    return max(1, min(4, (os.cpu_count() or 2) // 2))


# ==============================================================================
# 3. Model & Tokenizer Loading (Zero-Copy / Low-CPU-Mem / Safetensors / SDPA)
# ==============================================================================

def get_context_window(
    model_or_config: Any,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> int:
    """Dynamically inspect model/config and tokenizer to determine context window length."""
    cfg = getattr(model_or_config, "config", model_or_config)
    for attr in ("max_position_embeddings", "seq_length", "n_positions", "max_sequence_length", "sliding_window"):
        val = getattr(cfg, attr, None)
        if val and isinstance(val, int) and 512 <= val <= 1_000_000:
            return val

    if tokenizer is not None:
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max and isinstance(tok_max, int) and 512 <= tok_max <= 1_000_000:
            return tok_max
    return 4096


def load_model_and_tokenizer(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    device: Optional[torch.device] = None,
    torch_dtype: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
    attn_implementation: Optional[str] = "auto",
    compile_model: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device, int]:
    """Load Hugging Face model and tokenizer with SOTA weight & memory optimizations."""
    if device is None:
        device = get_optimal_device()
    else:
        validate_cuda_device(device, raise_on_error=True)
        configure_hardware_backends(device)

    selected_dtype = get_default_dtype(device, torch_dtype)

    if device.type != "cuda" and "awq" in model_name_or_path.lower():
        raise ValueError(
            f"Model '{model_name_or_path}' is an AWQ-quantized checkpoint requiring CUDA, "
            f"but current device is '{device.type}'."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
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
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.warning("bitsandbytes is not installed. Falling back to unquantized precision.")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = str(cache_dir)

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = selected_dtype
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"

    chosen_attn = resolve_optimal_attention(device, selected_dtype, attn_implementation)
    if chosen_attn:
        model_kwargs["attn_implementation"] = chosen_attn

    def _from_pretrained_compat(kwargs: Dict[str, Any]) -> PreTrainedModel:
        try:
            return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except TypeError as exc:
            if "dtype" in str(exc) and "dtype" in kwargs:
                compat_kwargs = dict(kwargs)
                compat_kwargs["torch_dtype"] = compat_kwargs.pop("dtype")
                return AutoModelForCausalLM.from_pretrained(model_name_or_path, **compat_kwargs)
            raise

    try:
        model = _from_pretrained_compat(model_kwargs)
    except Exception as e:
        if "attn_implementation" in model_kwargs:
            logger.info(f"Retrying model load with fallback attention ({e})...")
            model_kwargs.pop("attn_implementation", None)
            model = _from_pretrained_compat(model_kwargs)
        else:
            raise

    if not hasattr(model, "hf_device_map") and quantization_config is None:
        try:
            if next(model.parameters()).device != device:
                model = model.to(device)
        except Exception:
            pass

    model.eval()
    if compile_model and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.debug(f"torch.compile skipped: {e}")

    context_window = get_context_window(model, tokenizer)
    model_device = getattr(model, "device", None) or next(model.parameters(), torch.empty(0, device=device)).device
    return model, tokenizer, model_device, context_window


def init_model_and_tokenizer(model_path: str = DEFAULT_MODEL_NAME) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Direct initialization helper with automatic device, dtype, and attention selection."""
    device_name, dtype = resolve_device_and_dtype()
    model, tokenizer, _, _ = load_model_and_tokenizer(
        model_name_or_path=model_path,
        device=torch.device(device_name),
        torch_dtype=str(dtype).replace("torch.", ""),
        attn_implementation="auto",
    )
    return model, tokenizer


# ==============================================================================
# 4. Length-Bucketed Batch Collation & Inference Engine
# ==============================================================================

def create_length_buckets(
    records: List[Dict[str, Any]],
    batch_size: int,
    text_key: str = "text",
    index_key: str = "_orig_idx",
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    window_size: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Group items by sequence length to minimize padding waste in batched inference.
    When window_size > 0, length sorting is applied within local sliding windows of
    window_size items so multi-chunk documents stay contiguous for streaming saves.
    """
    b_size = max(1, batch_size)
    indexed = [dict(item, **{index_key: i}) for i, item in enumerate(records)]
    if tokenizer is not None:
        def _seq_len(x: Dict[str, Any]) -> int:
            txt = str(x.get(text_key, ""))
            try:
                ids = tokenizer.encode(txt, add_special_tokens=False)
                return len(ids) if isinstance(ids, list) else len(txt)
            except Exception:
                return len(txt)
    else:
        def _seq_len(x: Dict[str, Any]) -> int:
            return len(str(x.get(text_key, "")))

    if window_size is not None and window_size > b_size and len(indexed) > window_size:
        buckets: List[List[Dict[str, Any]]] = []
        for w_start in range(0, len(indexed), window_size):
            window = indexed[w_start : w_start + window_size]
            sorted_window = sorted(window, key=_seq_len)
            for i in range(0, len(sorted_window), b_size):
                buckets.append(sorted_window[i : i + b_size])
        return buckets

    sorted_recs = sorted(indexed, key=_seq_len)
    return [sorted_recs[i : i + b_size] for i in range(0, len(sorted_recs), b_size)]


def restore_original_order(
    processed_records: List[Dict[str, Any]],
    index_key: str = "_orig_idx",
) -> List[Dict[str, Any]]:
    """Restore original sequence order using the tracked index and clean metadata."""
    sorted_recs = sorted(processed_records, key=lambda x: x.get(index_key, 0))
    for r in sorted_recs:
        r.pop(index_key, None)
    return sorted_recs


def is_json_bracket_balanced(text: str, open_char: str = "[", close_char: str = "]") -> bool:
    """Check if text contains a complete, balanced top-level JSON array or object (ignoring brackets inside strings)."""
    depth = 0
    in_string = False
    escape = False
    started = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == open_char:
            depth += 1
            started = True
        elif ch == close_char and started:
            depth -= 1
            if depth == 0:
                return True
    return False


class JsonBracketStoppingCriteria(StoppingCriteria):
    """Halts batch generation as soon as all sequences close their outermost JSON bracket."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_length: int,
        response_prefix: str = "[",
        eos_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.response_prefix = response_prefix.strip() if response_prefix else "["
        self.open_char = "{" if self.response_prefix.startswith("{") else "["
        self.close_char = "}" if self.open_char == "{" else "]"
        self.eos_token_id = eos_token_id
        self._token_has_close: Dict[int, bool] = {}
        self._finished: Optional[List[bool]] = None

    def _has_closing_char(self, token_id: int) -> bool:
        cached = self._token_has_close.get(token_id)
        if cached is not None:
            return cached
        piece = getattr(self.tokenizer, "decode", lambda *a, **kw: None)(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        has_close = (self.close_char in piece) if isinstance(piece, str) else True
        self._token_has_close[token_id] = has_close
        return has_close

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        try:
            batch_size, seq_len = input_ids.shape
            if seq_len <= self.prompt_length:
                return False

            if self._finished is None or len(self._finished) != batch_size:
                self._finished = [False] * batch_size

            last_tokens = input_ids[:, -1].tolist()
            needs_decode = False
            for idx in range(batch_size):
                if self._finished[idx]:
                    continue
                last_tid = int(last_tokens[idx])
                if self.eos_token_id is not None and last_tid == self.eos_token_id:
                    self._finished[idx] = True
                    continue
                if self._has_closing_char(last_tid):
                    needs_decode = True

            if all(self._finished):
                return True
            if not needs_decode:
                return False

            gen_slice = input_ids[:, self.prompt_length :]
            decoded_batch = self.tokenizer.batch_decode(
                gen_slice,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded_batch, list) or len(decoded_batch) != batch_size:
                return False

            for idx, text in enumerate(decoded_batch):
                if self._finished[idx]:
                    continue
                if self.eos_token_id is not None and (gen_slice[idx] == self.eos_token_id).any():
                    self._finished[idx] = True
                    continue
                candidate = text.lstrip()
                if self.response_prefix and not candidate.startswith((self.open_char, "```")):
                    candidate = self.response_prefix + candidate
                if is_json_bracket_balanced(candidate, self.open_char, self.close_char):
                    self._finished[idx] = True

            return all(self._finished)
        except Exception:
            return False


class BatchInferenceEngine:
    """High-throughput local batch inference engine using ChatML templates and left-padding."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        context_window: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.context_window = context_window or get_context_window(model, tokenizer)
        self.generation_kwargs = dict(generation_kwargs or {})

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def target_device(self) -> torch.device:
        """Determine the device where model input embeddings reside."""
        if hasattr(self.model, "get_input_embeddings"):
            try:
                emb = self.model.get_input_embeddings()
                if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                    dev = getattr(emb.weight, "device", None)
                    if isinstance(dev, (torch.device, str)):
                        return torch.device(dev)
            except Exception:
                pass
        dev = getattr(self.model, "device", None)
        if isinstance(dev, (torch.device, str)):
            return torch.device(dev)
        try:
            p_dev = next(self.model.parameters()).device
            if isinstance(p_dev, (torch.device, str)):
                return torch.device(p_dev)
        except Exception:
            pass
        return self.device

    def format_chatml_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})
        return messages

    def render_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        response_prefix: Optional[str] = None,
    ) -> str:
        """Render messages via tokenizer.apply_chat_template with optional JSON prefix pre-filling."""
        rendered = None
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                candidate = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(candidate, str):
                    rendered = candidate
            except Exception as e:
                logger.debug(f"apply_chat_template raised {e}, falling back to ChatML formatting")

        if rendered is None:
            rendered = ""
            for msg in messages:
                rendered += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            rendered += "<|im_start|>assistant\n"

        if response_prefix:
            rendered += response_prefix
        return rendered

    @torch.inference_mode()
    def generate_batch(
        self,
        formatted_prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        response_prefix: Optional[str] = None,
        stop_on_json_close: bool = True,
        **extra_gen_kwargs,
    ) -> List[str]:
        """Generate completions for a batch of pre-formatted prompts with context-window protection."""
        if not formatted_prompts:
            return []

        tokens_to_gen = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        curr_top_p = top_p if top_p is not None else self.top_p
        curr_top_k = top_k if top_k is not None else self.top_k
        curr_rep_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

        max_prompt_len = max(64, self.context_window - tokens_to_gen)
        encoded = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        target_dev = self.target_device
        input_ids = encoded["input_ids"].to(target_dev)
        attention_mask = encoded["attention_mask"].to(target_dev)
        prompt_length = input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": tokens_to_gen,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **self.generation_kwargs,
        }

        if stop_on_json_close and response_prefix:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                JsonBracketStoppingCriteria(
                    tokenizer=self.tokenizer,
                    prompt_length=prompt_length,
                    response_prefix=response_prefix,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            ])

        if temp is not None and temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            if curr_top_p is not None:
                gen_kwargs["top_p"] = curr_top_p
            if curr_top_k is not None:
                gen_kwargs["top_k"] = curr_top_k
        else:
            gen_kwargs.update(EXTRACTION_GEN_CONFIG)
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            if curr_top_k is not None:
                gen_kwargs["top_k"] = None

        if curr_rep_penalty is not None:
            gen_kwargs["repetition_penalty"] = curr_rep_penalty

        gen_kwargs.update(extra_gen_kwargs)
        output_ids = self.model.generate(**gen_kwargs)
        generated_tokens = output_ids[:, prompt_length:]

        completions = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        del input_ids, attention_mask, output_ids, generated_tokens

        results: List[str] = []
        prefix_clean = response_prefix.strip() if response_prefix else ""
        for c in completions:
            comp = c.strip()
            if prefix_clean and comp and not comp.startswith((prefix_clean, "```")):
                comp = prefix_clean + comp
            results.append(comp)
        return results

    def run_all(
        self,
        items: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        prompt_key: str = "prompt",
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        batch_size: Optional[Union[int, str]] = None,
        bucket_by_length: bool = True,
        bucket_window_size: Optional[int] = None,
        response_prefix: Optional[str] = None,
        stop_on_json_close: bool = True,
        on_batch_start: Optional[Callable[[int, int, List[Dict[str, Any]]], None]] = None,
        on_batch_complete: Optional[Callable[[List[str], List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        max_workers: Optional[int] = None,
        **extra_gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """Run batch inference over all items, returning results in their original order."""
        if not items:
            return []

        tokens_to_gen = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if batch_size == "auto":
            eff_batch_size = resolve_dynamic_batch_size(
                device=self.target_device,
                requested_batch_size="auto",
                items=items,
                text_key=prompt_key,
                max_new_tokens=tokens_to_gen,
                tokenizer=self.tokenizer,
            )
        elif isinstance(batch_size, int) and batch_size > 0:
            eff_batch_size = batch_size
        else:
            eff_batch_size = self.batch_size

        eff_window_size = (
            bucket_window_size
            if bucket_window_size is not None
            else eff_batch_size * 8
        )

        if bucket_by_length and len(items) > 1 and eff_batch_size > 1:
            buckets = create_length_buckets(
                items,
                eff_batch_size,
                text_key=prompt_key,
                tokenizer=self.tokenizer,
                window_size=eff_window_size,
            )
        else:
            buckets = [
                [dict(item, _orig_idx=idx) for idx, item in enumerate(items[i : i + eff_batch_size], start=i)]
                for i in range(0, len(items), eff_batch_size)
            ]

        total_batches = len(buckets)

        def _default_batch_mapper(completions: List[str], batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [dict(item, raw_output=comp) for item, comp in zip(batch_items, completions)]

        postprocess_fn = on_batch_complete or _default_batch_mapper
        results: List[Dict[str, Any]] = []
        workers = max_workers if max_workers is not None else min(4, max(1, (os.cpu_count() or 2) // 2))

        if total_batches <= 1 or workers <= 1:
            for batch_idx, batch_items in enumerate(buckets, start=1):
                if on_batch_start is not None:
                    try:
                        on_batch_start(batch_idx, total_batches, batch_items)
                    except Exception as e:
                        logger.debug(f"on_batch_start callback notice: {e}")
                formatted_batch = [
                    self.render_chat_prompt(
                        self.format_chatml_messages(item.get(prompt_key, ""), system_prompt),
                        response_prefix=response_prefix,
                    )
                    for item in batch_items
                ]
                completions = self.generate_batch(
                    formatted_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    response_prefix=response_prefix,
                    stop_on_json_close=stop_on_json_close,
                    **extra_gen_kwargs,
                )
                results.extend(postprocess_fn(completions, batch_items))
            return restore_original_order(results)

        with ThreadPoolExecutor(max_workers=workers) as io_pool:
            futures = []
            for batch_idx, batch_items in enumerate(buckets, start=1):
                if on_batch_start is not None:
                    try:
                        on_batch_start(batch_idx, total_batches, batch_items)
                    except Exception as e:
                        logger.debug(f"on_batch_start callback notice: {e}")
                formatted_batch = [
                    self.render_chat_prompt(
                        self.format_chatml_messages(item.get(prompt_key, ""), system_prompt),
                        response_prefix=response_prefix,
                    )
                    for item in batch_items
                ]
                completions = self.generate_batch(
                    formatted_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    response_prefix=response_prefix,
                    stop_on_json_close=stop_on_json_close,
                    **extra_gen_kwargs,
                )
                futures.append(io_pool.submit(postprocess_fn, completions, batch_items))

            for future in futures:
                results.extend(future.result())

        clear_device_memory(self.target_device)
        return restore_original_order(results)
