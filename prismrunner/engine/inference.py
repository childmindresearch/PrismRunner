"""Native ChatML batch inference engine with left-padding and generation prompt alignment."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from prismrunner.engine.device import clear_device_memory

logger = logging.getLogger(__name__)

# Item 5: Deterministic Greedy Decoding preset for objective structured extraction
EXTRACTION_GEN_CONFIG: Dict[str, Any] = {
    "do_sample": False,
    "use_cache": True,
}


def create_length_buckets(
    records: List[Dict[str, Any]],
    batch_size: int,
    text_key: str = "text",
    index_key: str = "_orig_idx",
) -> List[List[Dict[str, Any]]]:
    """Group items by sequence length to eliminate padding waste in batched inference.

    Args:
        records: List of input dictionaries.
        batch_size: Number of items per batch bucket.
        text_key: Field in record to measure sequence length by.
        index_key: Metadata key used to track original record position.

    Returns:
        List of batched sub-lists sorted by text sequence length.
    """
    indexed_records = [dict(item, **{index_key: i}) for i, item in enumerate(records)]
    sorted_records = sorted(indexed_records, key=lambda x: len(str(x.get(text_key, ""))))
    return [sorted_records[i : i + batch_size] for i in range(0, len(sorted_records), batch_size)]


def restore_original_order(
    processed_records: List[Dict[str, Any]],
    index_key: str = "_orig_idx",
) -> List[Dict[str, Any]]:
    """Restore the original sequence order using the tracked index and clean metadata."""
    sorted_records = sorted(processed_records, key=lambda x: x.get(index_key, 0))
    for r in sorted_records:
        r.pop(index_key, None)
    return sorted_records


class BatchInferenceEngine:
    """Executes high-throughput batch inference adhering to ChatML templates and left-padding."""

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
        self.generation_kwargs = dict(generation_kwargs or {})

        # Enforce left-padding on the tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def target_device(self) -> torch.device:
        """Determines the device where model weights reside to avoid tensor mismatches."""
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
        """Construct standard chat message structure."""
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_prompt.strip()})
        return messages

    def render_chat_prompt(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Render messages via tokenizer.apply_chat_template with generation prompt.
        
        CRITICAL REQUIREMENT:
        Always set tokenize=False and add_generation_prompt=True so instruct models
        enter assistant generation mode directly without echoing prompt tokens.
        """
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.debug(f"apply_chat_template raised {e}, falling back to manual ChatML formatting")

        # Fallback to standard ChatML format if tokenizer has no template defined
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    @torch.inference_mode()
    def generate_batch(
        self,
        formatted_prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **extra_gen_kwargs,
    ) -> List[str]:
        """Generate completions for a list of pre-formatted ChatML prompts using left-padding."""
        if not formatted_prompts:
            return []

        tokens_to_gen = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        curr_top_p = top_p if top_p is not None else self.top_p
        curr_top_k = top_k if top_k is not None else self.top_k
        curr_rep_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

        # Tokenize with left padding and aligned attention masks
        encoded = self.tokenizer(
            formatted_prompts,
            padding=True,
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
        }

        # Base generation kwargs configured on the engine instance
        gen_kwargs.update(self.generation_kwargs)

        # Item 5: Enforced Deterministic Greedy Decoding when temperature <= 0 or do_sample is False
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

        # Apply call-specific overrides
        gen_kwargs.update(extra_gen_kwargs)

        # Run forward generation
        output_ids = self.model.generate(**gen_kwargs)

        # Slice strictly newly generated tokens to eliminate prompt echoing
        generated_tokens = output_ids[:, prompt_length:]

        completions = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Clean device memory
        del input_ids, attention_mask, output_ids, generated_tokens
        clear_device_memory(target_dev)

        return [c.strip() for c in completions]

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
        bucket_by_length: bool = True,
        on_batch_complete: Optional[Callable[[List[str], List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        max_workers: int = 2,
        **extra_gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """Run batch inference over all items, returning results mapped to original records."""
        if not items:
            return []

        # Item 1: Length-Bucketed Batch Collation to eliminate 50-70% zero-padding overhead
        if bucket_by_length and len(items) > 1 and self.batch_size > 1:
            buckets = create_length_buckets(items, self.batch_size, text_key=prompt_key)
        else:
            buckets = [
                [dict(item, _orig_idx=idx) for idx, item in enumerate(items[i : i + self.batch_size], start=i)]
                for i in range(0, len(items), self.batch_size)
            ]

        def _default_batch_mapper(completions: List[str], batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            mapped = []
            for item, comp in zip(batch_items, completions):
                res = dict(item)
                res["raw_output"] = comp
                mapped.append(res)
            return mapped

        postprocess_fn = on_batch_complete or _default_batch_mapper
        results: List[Dict[str, Any]] = []

        # If only 1 bucket or single worker requested, execute sequentially without pool overhead
        if len(buckets) <= 1 or max_workers <= 1:
            for batch_items in buckets:
                formatted_batch = [
                    self.render_chat_prompt(self.format_chatml_messages(item.get(prompt_key, ""), system_prompt))
                    for item in batch_items
                ]
                completions = self.generate_batch(
                    formatted_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **extra_gen_kwargs,
                )
                results.extend(postprocess_fn(completions, batch_items))
            return restore_original_order(results)

        # Item 3: Asynchronous Pipeline Overlapping (Producer-Consumer I/O)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as io_pool:
            futures = []
            for batch_items in buckets:
                formatted_batch = [
                    self.render_chat_prompt(self.format_chatml_messages(item.get(prompt_key, ""), system_prompt))
                    for item in batch_items
                ]
                completions = self.generate_batch(
                    formatted_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    **extra_gen_kwargs,
                )
                # Offload CPU-bound decoding, Pydantic validation, and disk I/O to background thread
                futures.append(io_pool.submit(postprocess_fn, completions, batch_items))

            for future in futures:
                results.extend(future.result())

        return restore_original_order(results)
