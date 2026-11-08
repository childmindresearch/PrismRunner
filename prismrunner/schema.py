"""Pydantic schemas, dynamic schema-to-prompt compiler, and robust JSON output parser."""

from __future__ import annotations
from dataclasses import dataclass
import enum
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import typing
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ==============================================================================
# 1. Default Built-In Task Schemas
# ==============================================================================

class ClassificationItem(BaseModel):
    """Classification record representing chain-of-thought reasoning, assigned category, and confidence."""

    reasoning: Optional[str] = Field(
        default=None,
        description="Brief justification or evidence analyzed before assigning the category.",
    )
    category: str = Field(description="The primary label or category assigned to the text.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class EntityItem(BaseModel):
    """Information extraction record representing a named entity or action item."""

    entity: str = Field(description="The extracted entity name, key term, or action item.")
    category: str = Field(description="Classification type of entity (e.g. Person, Org, Action, Date).")
    details: Optional[str] = Field(
        default=None,
        description="Associated metadata, owner, deadline, or descriptive context.",
    )


class SummaryItem(BaseModel):
    """Document synthesis record representing key points and cohesive summary."""

    summary: str = Field(description="A concise narrative summary capturing the main message.")
    key_points: List[str] = Field(
        default_factory=list,
        description="Key takeaways or bullet points highlighting critical findings.",
    )
    sentiment: Optional[str] = Field(
        default=None,
        description="Overall tone or sentiment if applicable (e.g. positive, neutral, negative).",
    )


# ==============================================================================
# 2. Schema-to-Prompt Compiler
# ==============================================================================

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
    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
        return annotation
    return annotation


def _extract_numeric_bounds(field_info: FieldInfo) -> Dict[str, Any]:
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


def _inspect_field(name: str, field_info: FieldInfo) -> typing.Tuple[str, Any]:
    raw_type = field_info.annotation
    unwrapped = _unwrap_type(raw_type)
    origin = get_origin(unwrapped)
    bounds = _extract_numeric_bounds(field_info)
    desc = field_info.description or ""
    desc_str = f" - {desc}" if desc else ""

    if isinstance(unwrapped, type) and issubclass(unwrapped, enum.Enum):
        allowed = [e.value for e in unwrapped]
        return f'- "{name}": one of {allowed}{desc_str}', (allowed[0] if allowed else "value")

    if origin is typing.Literal:
        allowed = list(get_args(unwrapped))
        return f'- "{name}": one of {allowed}{desc_str}', (allowed[0] if allowed else "value")

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
        spec = f'- "{name}": {type_name}{bounds_str}{desc_str}'

        if "ge" in bounds:
            sample = bounds["ge"]
        elif "gt" in bounds:
            sample = bounds["gt"] + (1 if unwrapped is int else 0.1)
        elif "le" in bounds:
            sample = bounds["le"]
        else:
            sample = 1 if unwrapped is int else 0.95
        return spec, sample

    if unwrapped is bool:
        return f'- "{name}": boolean (true/false){desc_str}', True

    if origin in (list, List):
        inner_args = get_args(unwrapped)
        if inner_args:
            inner_type = _unwrap_type(inner_args[0])
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                sub_compiled = compile_schema_to_prompt(inner_type)
                spec = (
                    f'- "{name}": list of {inner_type.__name__} objects with fields:\n  '
                    + sub_compiled.field_bullets.replace("\n", "\n  ")
                    + desc_str
                )
                return spec, [json.loads(sub_compiled.sample_json)]
            elif inner_type in (str, int, float, bool):
                inner_name = (
                    "string"
                    if inner_type is str
                    else ("integer" if inner_type is int else ("number" if inner_type is float else "boolean"))
                )
                sample = (
                    ["example_item_1", "example_item_2"]
                    if inner_type is str
                    else ([1, 2] if inner_type is int else ([1.0, 2.0] if inner_type is float else [True, False]))
                )
                return f'- "{name}": list of {inner_name}s{desc_str}', sample
        return f'- "{name}": list of items{desc_str}', ["item_1", "item_2"]

    if isinstance(unwrapped, type) and issubclass(unwrapped, BaseModel):
        sub_compiled = compile_schema_to_prompt(unwrapped)
        spec = (
            f'- "{name}": object with fields:\n  '
            + sub_compiled.field_bullets.replace("\n", "\n  ")
            + desc_str
        )
        return spec, json.loads(sub_compiled.sample_json)

    if origin in (dict, Dict) or unwrapped is dict:
        return f'- "{name}": key-value dictionary{desc_str}', {"key": "value"}

    return f'- "{name}": string{desc_str}', f"example_{name}"


def compile_schema_to_prompt(model_cls: Type[BaseModel]) -> CompiledPromptSchema:
    """Compile a Pydantic v2 model into human-readable prompt bullets and sample JSON."""
    bullets: List[str] = []
    sample_dict: Dict[str, Any] = {}

    for name, field_info in model_cls.model_fields.items():
        spec, sample_val = _inspect_field(name, field_info)
        bullets.append(spec)
        sample_dict[name] = sample_val

    return CompiledPromptSchema(
        schema_name=model_cls.__name__,
        field_bullets="\n".join(bullets),
        sample_json=json.dumps(sample_dict, indent=2),
        schema_directive="Return a valid JSON array of objects conforming to the schema specification.",
        model_class=model_cls,
    )


def load_schema_from_string(schema_spec: str) -> Type[BaseModel]:
    """Dynamically load a Pydantic model class from 'path/to/file.py:ClassName' or 'module.path:ClassName'."""
    if ":" not in schema_spec:
        raise ValueError(
            f"Invalid schema specification '{schema_spec}'. "
            "Expected format 'path/to/file.py:ClassName' or 'module.path:ClassName'."
        )

    module_path, class_name = [p.strip() for p in schema_spec.split(":", 1)]

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


# ==============================================================================
# 3. Universal Output Parser
# ==============================================================================

FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
THINK_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
TRAILING_COMMA_PATTERN = re.compile(r",\s*([\]}])")
BULLET_PATTERN = re.compile(r"^[-*•]\s+(.*)$|^\d+[\.\)]\s+(.*)$")


@dataclass
class ParseResult(Generic[T]):
    """Result of parsing and validating model generation."""

    items: List[T]
    raw_text: str
    success: bool
    parsing_strategy: str
    error: Optional[str] = None


def is_phantom_object(data: Any) -> bool:
    """Detect whether a parsed dict is a phantom object where every field is None."""
    if not isinstance(data, dict):
        return False
    if not data:
        return True
    return all(v is None for v in data.values())


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by local reasoning models."""
    cleaned = THINK_BLOCK_PATTERN.sub("", text)
    lower = cleaned.lower()
    if "</think>" in lower:
        idx = lower.rfind("</think>")
        cleaned = cleaned[idx + len("</think>") :]
    return cleaned.strip()


def _strip_markdown_fences(text: str) -> str:
    cleaned = _strip_think_blocks(text)
    match = FENCE_PATTERN.search(cleaned)
    return match.group(1).strip() if match else cleaned.strip()


def _repair_common_json_glitches(text: str) -> str:
    repaired = TRAILING_COMMA_PATTERN.sub(r"\1", text)
    # Escape unescaped Python triple quotes ("""...""") that LLMs often emit inside JSON string values
    if '"""' in repaired:
        repaired = re.sub(r'(?<!\\)"""', r'\"\"\"', repaired)
    return repaired


def _salvage_truncated_json(text: str) -> Optional[Any]:
    """Salvage truncated JSON arrays or objects cut off by max_new_tokens without external dependencies."""
    candidate = _repair_common_json_glitches(_strip_markdown_fences(text)).strip()
    if not candidate:
        return None

    # 1. If it starts as a JSON array, salvage all complete top-level {...} objects before the cutoff
    first_bracket = candidate.find("[")
    first_brace = candidate.find("{")
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        arr_text = candidate[first_bracket:]
        last_close_brace = arr_text.rfind("}")
        if last_close_brace != -1:
            repaired_array = _repair_common_json_glitches(arr_text[: last_close_brace + 1] + "]")
            try:
                return json.loads(repaired_array)
            except Exception:
                pass

    # 2. Balance unclosed quotes and brackets stack-wise
    start_idx = first_bracket if (first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace)) else first_brace
    if start_idx != -1:
        fragment = candidate[start_idx:]
        stack: List[str] = []
        in_string = False
        escape = False
        for ch in fragment:
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
            if ch in ("[", "{"):
                stack.append("]" if ch == "[" else "}")
            elif ch in ("]", "}") and stack and stack[-1] == ch:
                stack.pop()

        closed = fragment + ('"' if in_string else "")
        closed = re.sub(r"[,:\s]+$", "", closed)
        closed = _repair_common_json_glitches(closed + "".join(reversed(stack)))
        try:
            return json.loads(closed)
        except Exception:
            pass

    # 3. Optional json_repair fallback if installed in the environment
    try:
        import json_repair  # type: ignore
        return json_repair.loads(candidate)
    except Exception:
        return None


def _extract_bracket_substring(text: str) -> Optional[str]:
    first_bracket, last_bracket = text.find("["), text.rfind("]")
    first_brace, last_brace = text.find("{"), text.rfind("}")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        if last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1]
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1]
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]
    return None


def sanitize_json_output(raw_text: str) -> Any:
    """Strip <think> blocks and markdown fences, extract outermost JSON brackets, repair glitches, and parse."""
    text = _strip_think_blocks(raw_text)
    if not text:
        return []

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    first_bracket, last_bracket = text.find("["), text.rfind("]")
    first_brace, last_brace = text.find("{"), text.rfind("}")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        if last_brace != -1 and last_brace > first_brace:
            text = text[first_brace : last_brace + 1].strip()
    elif first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        text = text[first_bracket : last_bracket + 1].strip()

    text = _repair_common_json_glitches(text)
    try:
        return json.loads(text)
    except Exception:
        salvaged = _salvage_truncated_json(raw_text)
        if salvaged is not None:
            return salvaged
        raise


def parse_raw_text(text: str) -> Any:
    """Attempt multi-stage extraction of structured JSON or fallback to clean lines."""
    clean_text = text.strip()
    if not clean_text:
        return []

    for candidate_fn in (
        lambda: sanitize_json_output(clean_text),
        lambda: json.loads(clean_text),
        lambda: json.loads(_strip_markdown_fences(clean_text)),
    ):
        try:
            return candidate_fn()
        except Exception:
            pass

    stripped = _strip_markdown_fences(clean_text)
    bracket_slice = _extract_bracket_substring(stripped)
    if bracket_slice:
        for s in (bracket_slice, _repair_common_json_glitches(bracket_slice)):
            try:
                return json.loads(s)
            except Exception:
                pass

    jsonl_candidates = []
    for line in stripped.splitlines():
        line = line.strip()
        if (line.startswith("{") and line.endswith("}")) or (line.startswith("[") and line.endswith("]")):
            try:
                jsonl_candidates.append(json.loads(line))
            except Exception:
                continue
    if jsonl_candidates:
        return jsonl_candidates

    lines = []
    for line in clean_text.splitlines():
        line = line.strip()
        if not line:
            continue
        bullet_match = BULLET_PATTERN.match(line)
        if bullet_match:
            item = bullet_match.group(1) or bullet_match.group(2)
            if item:
                lines.append(item.strip())
        else:
            lines.append(line)
    return lines


class UniversalOutputParser:
    """Parses raw LLM text into validated Pydantic instances or clean strings."""

    @staticmethod
    def parse(
        text: str,
        model_cls: Optional[Type[BaseModel]] = None,
    ) -> ParseResult[Any]:
        if not text or not text.strip():
            return ParseResult(items=[], raw_text=text, success=True, parsing_strategy="empty")

        raw_parsed = parse_raw_text(text)

        if model_cls is None:
            if isinstance(raw_parsed, list):
                clean_strings = [
                    elem.strip() if isinstance(elem, str) else (json.dumps(elem) if isinstance(elem, dict) else str(elem).strip())
                    for elem in raw_parsed
                ]
                return ParseResult(items=clean_strings, raw_text=text, success=True, parsing_strategy="unstructured_strings")
            elif isinstance(raw_parsed, dict):
                return ParseResult(items=[json.dumps(raw_parsed)], raw_text=text, success=True, parsing_strategy="unstructured_dict")
            return ParseResult(items=[str(raw_parsed).strip()], raw_text=text, success=True, parsing_strategy="unstructured_raw")

        if isinstance(raw_parsed, list):
            candidates = raw_parsed
        elif isinstance(raw_parsed, dict):
            wrapper_keys = ("items", "results", "data", "extractions", "entities", "records", "output", "elements")
            found_list = next(
                (raw_parsed[wk] for wk in wrapper_keys if wk in raw_parsed and isinstance(raw_parsed[wk], list) and wk not in model_cls.model_fields),
                None,
            )
            candidates = found_list if found_list is not None else [raw_parsed]
        else:
            candidates = [raw_parsed]

        validated_items: List[BaseModel] = []
        errors: List[str] = []

        for cand in candidates:
            if isinstance(cand, str) and cand.strip().startswith("{") and cand.strip().endswith("}"):
                try:
                    cand = json.loads(cand)
                except Exception:
                    pass

            if not isinstance(cand, dict) and isinstance(cand, (str, int, float, bool)):
                if len(model_cls.model_fields) == 1:
                    single_field = next(iter(model_cls.model_fields.keys()))
                    cand = {single_field: cand}
                else:
                    for fallback_field in ("item", "name", "text"):
                        if fallback_field in model_cls.model_fields:
                            cand = {fallback_field: cand}
                            break

            if is_phantom_object(cand):
                continue

            if isinstance(cand, dict):
                try:
                    val = model_cls.model_validate(cand)
                    if all(v is None for v in val.model_dump().values()):
                        continue
                    validated_items.append(val)
                except ValidationError as ve:
                    errors.append(str(ve))
            else:
                errors.append(f"Expected dict but got {type(cand).__name__}")

        if validated_items:
            return ParseResult(items=validated_items, raw_text=text, success=True, parsing_strategy="pydantic_validated")

        error_summary = "; ".join(errors[:3]) if errors else "No valid schema records found."
        return ParseResult(items=[], raw_text=text, success=False, parsing_strategy="failed_validation", error=error_summary)

