"""Universal prompt template loader and renderer for PrismRunner."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class PromptTemplate:
    """Manages template loading, package fallback path resolution, and variable substitution."""

    @staticmethod
    def resolve_template_path(path: Union[str, Path]) -> Path:
        """Resolve template file path, checking package-relative fallbacks if needed."""
        p = Path(path)
        if not p.exists():
            pkg_fallback = Path(__file__).resolve().parent.parent / p
            if pkg_fallback.exists():
                return pkg_fallback
        return p

    @classmethod
    def load(
        cls,
        source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        default_task: Optional[str] = None,
        fallback_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load template dictionary from in-memory dict, in-memory string, or YAML file."""
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

        path_to_try = Path(source) if source else None
        if path_to_try:
            resolved = cls.resolve_template_path(path_to_try)
            if resolved.exists():
                with open(resolved, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}

        return {
            "version": "1.0",
            "task": default_task or "custom",
            "instruction": f"Perform {default_task or 'task'} accurately on the text below.",
            "variables": {},
            "prompt": fallback_prompt or "Source Content:\n\"\"\"\n{{ text }}\n\"\"\"\n\n{schema_block}",
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
        """Uniformly render prompt text, substituting variables, instruction, schema directives, and text."""
        raw_tmpl = template.get("prompt", "{{ text }}")
        merged_vars = dict(template.get("variables", {}))
        if variables:
            merged_vars.update(variables)

        inst = instruction if instruction is not None else template.get("instruction", "")
        merged_vars["instruction"] = inst
        merged_vars["schema_block"] = schema_block

        rendered = raw_tmpl
        for var_name, var_val in merged_vars.items():
            rendered = rendered.replace(f"{{{var_name}}}", str(var_val))

        # Uniform text slot substitution across standard aliases
        for slot in ("{{ text }}", "{{ text}}", "{{ item }}", "{{ item}}", "{{ document }}", "{text}", "{item}", "{document}", "{content}"):
            if slot in rendered:
                rendered = rendered.replace(slot, text)

        return rendered
