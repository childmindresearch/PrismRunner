"""Universal robust output parser with markdown fence stripping, JSON repair, and Pydantic validation."""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")

FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
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


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown ```json ... ``` or ``` ... ``` wrappers."""
    # Look for code block contents
    match = FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _repair_common_json_glitches(text: str) -> str:
    """Fix common LLM JSON syntax errors like trailing commas."""
    # Remove trailing commas before closing braces or brackets: , \s* } -> }
    return TRAILING_COMMA_PATTERN.sub(r"\1", text)


def _extract_bracket_substring(text: str) -> Optional[str]:
    """Find the outermost JSON array [...] or object {...} slice."""
    # Check for array first
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1]

    # Check for object
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1]

    return None


def parse_raw_text(text: str) -> Any:
    """Attempt multi-stage extraction of structured JSON or fallback to lines."""
    clean_text = text.strip()
    if not clean_text:
        return []

    # 1. Direct JSON parse
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # 2. Markdown fence stripping
    stripped = _strip_markdown_fences(clean_text)
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # 3. Outer bracket extraction
    bracket_slice = _extract_bracket_substring(stripped)
    if bracket_slice:
        try:
            return json.loads(bracket_slice)
        except Exception:
            # Try repairing trailing commas on bracket slice
            try:
                return json.loads(_repair_common_json_glitches(bracket_slice))
            except Exception:
                pass

    # 4. JSON Lines (one JSON object per line)
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

    # 5. Fallback: bullet points or non-empty newline-delimited text
    lines = []
    for line in clean_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip bullet prefixes like "- ", "* ", "1. "
        bullet_match = BULLET_PATTERN.match(line)
        if bullet_match:
            item = bullet_match.group(1) or bullet_match.group(2)
            if item:
                lines.append(item.strip())
        else:
            lines.append(line)

    return lines


class UniversalOutputParser:
    """Parses raw text into validated Pydantic instances or clean strings."""

    @staticmethod
    def parse(
        text: str,
        model_cls: Optional[Type[BaseModel]] = None,
    ) -> ParseResult[Any]:
        """Universal parse method enforcing schema validation and phantom object filtering."""
        if not text or not text.strip():
            return ParseResult(items=[], raw_text=text, success=True, parsing_strategy="empty")

        raw_parsed = parse_raw_text(text)

        # Case 1: No schema passed -> cleanly return stripped strings
        if model_cls is None:
            if isinstance(raw_parsed, list):
                # Clean elements into strings
                clean_strings = []
                for elem in raw_parsed:
                    if isinstance(elem, str):
                        clean_strings.append(elem.strip())
                    elif isinstance(elem, dict):
                        clean_strings.append(json.dumps(elem))
                    else:
                        clean_strings.append(str(elem).strip())
                return ParseResult(
                    items=clean_strings,
                    raw_text=text,
                    success=True,
                    parsing_strategy="unstructured_strings",
                )
            elif isinstance(raw_parsed, dict):
                return ParseResult(
                    items=[json.dumps(raw_parsed)],
                    raw_text=text,
                    success=True,
                    parsing_strategy="unstructured_dict",
                )
            else:
                return ParseResult(
                    items=[str(raw_parsed).strip()],
                    raw_text=text,
                    success=True,
                    parsing_strategy="unstructured_raw",
                )

        # Case 2: Pydantic Schema is passed
        candidates: List[Any] = []
        if isinstance(raw_parsed, list):
            candidates = raw_parsed
        elif isinstance(raw_parsed, dict):
            # Check if LLM wrapped array in an outer container dict e.g. {"items": [...]}
            wrapper_keys = ("items", "results", "data", "extractions", "entities", "records", "output", "elements")
            found_wrapped_list = None
            for wk in wrapper_keys:
                if wk in raw_parsed and isinstance(raw_parsed[wk], list) and wk not in model_cls.model_fields:
                    found_wrapped_list = raw_parsed[wk]
                    break
            if found_wrapped_list is not None:
                candidates = found_wrapped_list
            else:
                candidates = [raw_parsed]
        else:
            candidates = [raw_parsed]

        validated_items: List[BaseModel] = []
        errors: List[str] = []

        for cand in candidates:
            # If candidate is a string starting with '{' and ending with '}', attempt json parsing
            if isinstance(cand, str) and cand.strip().startswith("{") and cand.strip().endswith("}"):
                try:
                    cand = json.loads(cand)
                except Exception:
                    pass

            # If candidate is a scalar (string, int, etc.) and model expects fields
            if not isinstance(cand, dict) and isinstance(cand, (str, int, float, bool)):
                # If model has exactly 1 field, map directly to that field
                if len(model_cls.model_fields) == 1:
                    single_field = next(iter(model_cls.model_fields.keys()))
                    cand = {single_field: cand}
                elif "item" in model_cls.model_fields:
                    cand = {"item": cand}
                elif "name" in model_cls.model_fields:
                    cand = {"name": cand}
                elif "text" in model_cls.model_fields:
                    cand = {"text": cand}

            # CRITICAL REQUIREMENT: Discard phantom objects where all fields are None
            if is_phantom_object(cand):
                logger.debug(f"Discarded phantom object with all None fields: {cand}")
                continue

            if isinstance(cand, dict):
                try:
                    val = model_cls.model_validate(cand)
                    # Double check validated model fields
                    if all(v is None for v in val.model_dump().values()):
                        logger.debug(f"Discarded validated phantom model instance: {val}")
                        continue
                    validated_items.append(val)
                except ValidationError as ve:
                    errors.append(str(ve))
            else:
                errors.append(f"Expected dict but got {type(cand).__name__}")

        if validated_items:
            return ParseResult(
                items=validated_items,
                raw_text=text,
                success=True,
                parsing_strategy="pydantic_validated",
            )
        else:
            error_summary = "; ".join(errors[:3]) if errors else "No valid schema records found."
            return ParseResult(
                items=[],
                raw_text=text,
                success=False,
                parsing_strategy="failed_validation",
                error=error_summary,
            )
