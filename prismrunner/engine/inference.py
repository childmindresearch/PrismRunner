"""Native ChatML batch inference engine with left-padding and generation prompt alignment."""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from prismrunner.engine.device import clear_device_memory

logger = logging.getLogger(__name__)


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Enforce left-padding on the tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
    ) -> List[str]:
        """Generate completions for a list of pre-formatted ChatML prompts using left-padding."""
        if not formatted_prompts:
            return []

        tokens_to_gen = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        # Tokenize with left padding and aligned attention masks
        encoded = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": tokens_to_gen,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["do_sample"] = False

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
        clear_device_memory(self.device)

        return [c.strip() for c in completions]

    def run_all(
        self,
        items: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        prompt_key: str = "prompt",
    ) -> List[Dict[str, Any]]:
        """Run batch inference over all items, returning results mapped to original records."""
        results: List[Dict[str, Any]] = []
        total_items = len(items)

        for i in range(0, total_items, self.batch_size):
            batch_items = items[i : i + self.batch_size]
            formatted_batch: List[str] = []

            for item in batch_items:
                user_content = item.get(prompt_key, "")
                messages = self.format_chatml_messages(user_content, system_prompt)
                formatted_prompt = self.render_chat_prompt(messages)
                formatted_batch.append(formatted_prompt)

            completions = self.generate_batch(
                formatted_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            for item, comp in zip(batch_items, completions):
                res = dict(item)
                res["raw_output"] = comp
                results.append(res)

        return results
