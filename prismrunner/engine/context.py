"""Context window validation, token budgeting, and hardware advisories."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import PreTrainedTokenizerBase
import torch

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Calculates and enforces token reservations across the context window."""

    model_context_limit: int
    max_new_tokens: int = 512
    system_prompt_tokens: int = 64
    template_tokens: int = 128
    safety_buffer: int = 128

    @property
    def reserved_tokens(self) -> int:
        """Total tokens reserved for generation, templates, and safety."""
        return self.max_new_tokens + self.system_prompt_tokens + self.template_tokens + self.safety_buffer

    @property
    def max_input_tokens(self) -> int:
        """Maximum safe token count available for input text / chunks."""
        available = self.model_context_limit - self.reserved_tokens
        if available <= 0:
            raise ValueError(
                f"Context budget exceeded: model context limit ({self.model_context_limit}) "
                f"is smaller than reserved tokens ({self.reserved_tokens}). "
                f"Reduce max_new_tokens ({self.max_new_tokens}) or safety_buffer ({self.safety_buffer})."
            )
        return available

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        context_limit: int,
        system_prompt: Optional[str] = None,
        template_text: Optional[str] = None,
        max_new_tokens: int = 512,
        safety_buffer: int = 128,
    ) -> "ContextBudget":
        """Compute exact token reservations based on actual tokenizer counts."""
        sys_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False)) if system_prompt else 32
        tmpl_tokens = len(tokenizer.encode(template_text, add_special_tokens=False)) if template_text else 96

        return cls(
            model_context_limit=context_limit,
            max_new_tokens=max_new_tokens,
            system_prompt_tokens=sys_tokens,
            template_tokens=tmpl_tokens,
            safety_buffer=safety_buffer,
        )


def generate_hardware_advisory(
    model_name: str,
    context_limit: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Provide hardware-specific recommendations for batch sizes, precision, and chunking."""
    is_mps = device.type == "mps"
    is_cuda = device.type == "cuda"

    # Determine recommended batch size and chunk size based on hardware capabilities
    if is_cuda:
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)
        if vram_gb >= 24:
            rec_batch = 16
            rec_chunk = min(2048, context_limit // 2)
            rec_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif vram_gb >= 12:
            rec_batch = 8
            rec_chunk = min(1536, context_limit // 2)
            rec_dtype = "float16"
        else:
            rec_batch = 4
            rec_chunk = min(1024, context_limit // 2)
            rec_dtype = "float16"
    elif is_mps:
        # Apple Silicon unified memory
        rec_batch = 4
        rec_chunk = min(1024, context_limit // 2)
        rec_dtype = "float16"
    else:
        # CPU
        rec_batch = 1
        rec_chunk = min(512, context_limit // 4)
        rec_dtype = "float32"

    budget = ContextBudget(
        model_context_limit=context_limit,
        max_new_tokens=512,
        system_prompt_tokens=64,
        template_tokens=128,
        safety_buffer=128,
    )

    return {
        "model_name": model_name,
        "device": str(device),
        "context_limit": context_limit,
        "recommended_dtype": rec_dtype,
        "recommended_batch_size": rec_batch,
        "recommended_chunk_size": rec_chunk,
        "max_safe_input_tokens": budget.max_input_tokens,
        "reserved_budget_tokens": budget.reserved_tokens,
    }
