"""Built-in prompt templates and variable renderer for PrismRunner."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


DEFAULT_PROMPTS: Dict[str, Dict[str, Any]] = {
    "classify": {
        "version": "1.0",
        "task": "classify",
        "system_prompt": "You are PrismRunner, an accurate local classification engine. Follow structured format instructions strictly.",
        "instruction": "Classify the following text into the most appropriate category.",
        "variables": {
            "categories": "Technology, Business, Science, Health, Entertainment, Politics, Other",
            "guidelines": "Select the most accurate category that reflects the primary subject matter of the text. Assign a confidence score from 0.0 to 1.0.",
        },
        "prompt": (
            "{instruction}\n\n"
            "Allowed Categories:\n{categories}\n\n"
            "Guidelines:\n{guidelines}\n\n"
            'Source Content:\n"""\n{{ text }}\n"""\n\n'
            "{schema_block}"
        ),
    },
    "extract": {
        "version": "1.0",
        "task": "extract",
        "system_prompt": "You are PrismRunner, an exact information extraction engine. Output valid JSON array only.",
        "instruction": "Extract all distinct key items, entities, or structured records from the provided text.",
        "variables": {
            "target": "Named entities, organizations, key technical terms, action items, and deliverables",
            "guidelines": "Extract clear, distinct items conforming to the schema. Avoid duplicate items.",
        },
        "prompt": (
            "{instruction}\n\n"
            "Target Elements:\n{target}\n\n"
            "Guidelines:\n{guidelines}\n\n"
            'Source Content:\n"""\n{{ text }}\n"""\n\n'
            "{schema_block}"
        ),
    },
    "summarize": {
        "version": "1.0",
        "task": "summarize",
        "system_prompt": "You are PrismRunner, an expert document synthesis engine. Output structured JSON only.",
        "instruction": "Summarize the primary message and extract essential takeaway points from the text.",
        "variables": {
            "audience": "Executive and technical stakeholders",
            "guidelines": "Provide a concise, faithful narrative summary followed by distinct key bullet points.",
        },
        "prompt": (
            "{instruction}\n\n"
            "Target Audience:\n{audience}\n\n"
            "Guidelines:\n{guidelines}\n\n"
            'Source Content:\n"""\n{{ text }}\n"""\n\n'
            "{schema_block}"
        ),
    },
}


class PromptTemplate:
    """Manages template loading (in-memory dict, string, YAML file, or built-in defaults) and rendering."""

    @classmethod
    def load(
        cls,
        source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        default_task: Optional[str] = None,
        fallback_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load template dictionary from in-memory dict, raw string, YAML file, or built-in task defaults."""
        if source is not None:
            if isinstance(source, dict):
                return dict(source)
            if isinstance(source, str) and ("\n" in source or not source.endswith((".yaml", ".yml"))):
                return {
                    "version": "custom",
                    "task": default_task or "custom",
                    "instruction": "",
                    "variables": {},
                    "prompt": source,
                }
            path_to_try = Path(source)
            if path_to_try.exists():
                with open(path_to_try, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        if default_task and default_task in DEFAULT_PROMPTS and fallback_prompt is None:
            return dict(DEFAULT_PROMPTS[default_task])

        return {
            "version": "1.0",
            "task": default_task or "custom",
            "instruction": f"Perform {default_task or 'task'} accurately on the text below.",
            "variables": {},
            "prompt": fallback_prompt or 'Source Content:\n"""\n{{ text }}\n"""\n\n{schema_block}',
            "system_prompt": system_prompt,
        }

    @staticmethod
    def render(
        template: Dict[str, Any],
        text: str,
        schema_block: str = "",
        variables: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Render prompt text substituting variables, instruction, schema directives, and source text."""
        raw_tmpl = template.get("prompt", "{{ text }}")
        merged_vars = dict(template.get("variables", {}))
        if variables:
            merged_vars.update(variables)

        merged_vars["instruction"] = instruction if instruction is not None else template.get("instruction", "")
        merged_vars["schema_block"] = schema_block

        rendered = raw_tmpl
        for var_name, var_val in merged_vars.items():
            rendered = rendered.replace(f"{{{var_name}}}", str(var_val))

        for slot in ("{{ text }}", "{{ text}}", "{{ item }}", "{{ item}}", "{{ document }}", "{text}", "{item}", "{document}", "{content}"):
            if slot in rendered:
                rendered = rendered.replace(slot, text)

        return rendered
