"""Dynamic Schema-to-Prompt Compiler.

Converts Pydantic v2 models into clean, human-readable prompt instructions,
constraint bullet points, and concrete sample JSON structures without ever
dumping raw OpenAPI schemas or $defs/$ref blocks.
"""

from __future__ import annotations
import enum
import json
import typing
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union, get_args, get_origin
from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass
class CompiledPromptSchema:
    """Compiled schema specifications formatted for prompt injection."""

    schema_name: str
    field_bullets: str
    sample_json: str
    schema_directive: str
    model_class: Type[BaseModel]

    def render_prompt_block(self, as_array: bool = True) -> str:
        """Render the complete schema specification block for prompt templates."""
        target_shape = "a valid JSON array of objects" if as_array else "a valid JSON object"
        sample_display = f"[\n  {self.sample_json}\n]" if as_array else self.sample_json

        return (
            f"Expected Output Format: Return ONLY {target_shape} conforming to the following structure:\n\n"
            f"Field Specifications:\n{self.field_bullets}\n\n"
            f"Example Structure:\n```json\n{sample_display}\n```\n"
            f"IMPORTANT: Output valid, parseable JSON only. Do not include markdown preamble, chatter, or explanations."
        )


def _unwrap_type(annotation: Any) -> Any:
    """Unwrap Optional, Union, and Annotated types down to primary type."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        return annotation
    return annotation


def _extract_numeric_bounds(field_info: FieldInfo) -> Dict[str, Any]:
    """Extract ge, le, gt, lt bounds from field metadata or attributes."""
    bounds: Dict[str, Any] = {}
    for attr in ("ge", "le", "gt", "lt"):
        val = getattr(field_info, attr, None)
        if val is not None:
            bounds[attr] = val

    for meta in getattr(field_info, "metadata", []):
        for attr in ("ge", "le", "gt", "lt"):
            if hasattr(meta, attr):
                val = getattr(meta, attr)
                if val is not None:
                    bounds[attr] = val
    return bounds


def _inspect_field(
    name: str,
    field_info: FieldInfo,
) -> typing.Tuple[str, Any]:
    """Analyze a Pydantic field and produce a human-readable specification and sample value."""
    raw_type = field_info.annotation
    unwrapped = _unwrap_type(raw_type)
    origin = get_origin(unwrapped)
    bounds = _extract_numeric_bounds(field_info)
    desc = field_info.description or ""
    desc_str = f" - {desc}" if desc else ""

    # Check Enum
    if isinstance(unwrapped, type) and issubclass(unwrapped, enum.Enum):
        allowed = [e.value for e in unwrapped]
        spec = f"- \"{name}\": one of {allowed}{desc_str}"
        sample = allowed[0] if allowed else "value"
        return spec, sample

    # Check Literal
    if origin is typing.Literal:
        allowed = list(get_args(unwrapped))
        spec = f"- \"{name}\": one of {allowed}{desc_str}"
        sample = allowed[0] if allowed else "value"
        return spec, sample

    # Check Numeric types
    if unwrapped in (int, float):
        type_name = "integer" if unwrapped is int else "number"
        bound_parts = []
        if "ge" in bounds and "le" in bounds:
            bound_parts.append(f"from {bounds['ge']} to {bounds['le']}")
        else:
            if "ge" in bounds:
                bound_parts.append(f">= {bounds['ge']}")
            if "gt" in bounds:
                bound_parts.append(f"> {bounds['gt']}")
            if "le" in bounds:
                bound_parts.append(f"<= {bounds['le']}")
            if "lt" in bounds:
                bound_parts.append(f"< {bounds['lt']}")

        bounds_str = f" ({', '.join(bound_parts)})" if bound_parts else ""
        spec = f"- \"{name}\": {type_name}{bounds_str}{desc_str}"

        # Generate sample within bounds
        if "ge" in bounds:
            sample = bounds["ge"]
        elif "gt" in bounds:
            sample = bounds["gt"] + (1 if unwrapped is int else 0.1)
        elif "le" in bounds:
            sample = bounds["le"]
        else:
            sample = 1 if unwrapped is int else 0.95
        return spec, sample

    # Check Boolean
    if unwrapped is bool:
        spec = f"- \"{name}\": boolean (true/false){desc_str}"
        return spec, True

    # Check List
    if origin in (list, List):
        inner_args = get_args(unwrapped)
        if inner_args:
            inner_type = _unwrap_type(inner_args[0])
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                sub_compiled = compile_schema_to_prompt(inner_type)
                spec = f"- \"{name}\": list of {inner_type.__name__} objects with fields:\n  " + sub_compiled.field_bullets.replace("\n", "\n  ") + desc_str
                sample = [json.loads(sub_compiled.sample_json)]
                return spec, sample
            elif inner_type in (str, int, float, bool):
                inner_name = "string" if inner_type is str else ("integer" if inner_type is int else ("number" if inner_type is float else "boolean"))
                spec = f"- \"{name}\": list of {inner_name}s{desc_str}"
                sample = ["example_item_1", "example_item_2"] if inner_type is str else ([1, 2] if inner_type is int else ([1.0, 2.0] if inner_type is float else [True, False]))
                return spec, sample
        spec = f"- \"{name}\": list of items{desc_str}"
        sample = ["item_1", "item_2"]
        return spec, sample

    # Check Nested BaseModel
    if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
        sub_compiled = compile_schema_to_prompt(unwrapped)
        spec = f"- \"{name}\": object with fields:\n  " + sub_compiled.field_bullets.replace("\n", "\n  ") + desc_str
        sample = json.loads(sub_compiled.sample_json)
        return spec, sample

    # Check Dict / Mapping
    if origin in (dict, Dict) or unwrapped is dict:
        spec = f"- \"{name}\": key-value dictionary{desc_str}"
        sample = {"key": "value"}
        return spec, sample

    # Default: String
    spec = f"- \"{name}\": string{desc_str}"
    sample = f"example_{name}"
    return spec, sample


def compile_schema_to_prompt(model_cls: Type[BaseModel]) -> CompiledPromptSchema:
    """Compile a Pydantic v2 model into prompt directives, bullets, and sample JSON.
    
    Guarantees no raw OpenAPI schema dumps, no $defs, and no $ref references.
    """
    bullets: List[str] = []
    sample_dict: Dict[str, Any] = {}

    for name, field_info in model_cls.model_fields.items():
        spec, sample_val = _inspect_field(name, field_info)
        bullets.append(spec)
        sample_dict[name] = sample_val

    field_bullets_str = "\n".join(bullets)
    sample_json_str = json.dumps(sample_dict, indent=2)
    directive_str = (
        "Return a valid JSON array of objects conforming to the schema specification."
    )

    return CompiledPromptSchema(
        schema_name=model_cls.__name__,
        field_bullets=field_bullets_str,
        sample_json=sample_json_str,
        schema_directive=directive_str,
        model_class=model_cls,
    )


def load_schema_from_string(schema_spec: str) -> Type[BaseModel]:
    """Dynamically load a Pydantic model class from 'path/to/file.py:ClassName' or 'module.path:ClassName'."""
    import importlib
    import importlib.util
    from pathlib import Path

    if ":" not in schema_spec:
        raise ValueError(
            f"Invalid schema specification '{schema_spec}'. "
            f"Expected format 'path/to/file.py:ClassName' or 'module.path:ClassName'."
        )

    module_path, class_name = schema_spec.split(":", 1)
    module_path = module_path.strip()
    class_name = class_name.strip()

    if module_path.endswith(".py") or "/" in module_path or "\\" in module_path:
        file_path = Path(module_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)

    if not hasattr(mod, class_name):
        raise AttributeError(f"Module '{module_path}' has no attribute '{class_name}'")

    cls = getattr(mod, class_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError(f"'{class_name}' must be a subclass of pydantic.BaseModel")

    return cls

