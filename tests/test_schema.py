"""Unit tests for prismrunner.schema (Schema-to-Prompt Compiler and UniversalOutputParser)."""

import enum
import json
import unittest
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from prismrunner.schema import (
    ClassificationItem,
    UniversalOutputParser,
    compile_schema_to_prompt,
    is_phantom_object,
    sanitize_json_output,
)


class StatusEnum(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"


class SubItem(BaseModel):
    item_name: str
    quantity: int


class SampleSchema(BaseModel):
    category: StatusEnum = Field(description="Operational status")
    priority: Literal["low", "high"] = Field(description="Priority ranking")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence metric")
    items: List[SubItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StrictItem(BaseModel):
    name: str
    value: int


class NullableItem(BaseModel):
    name: Optional[str] = None
    score: Optional[float] = None


class TestSchemaAndParser(unittest.TestCase):

    def test_schema_compiler(self):
        compiled = compile_schema_to_prompt(SampleSchema)
        self.assertNotIn("$defs", compiled.field_bullets)
        self.assertNotIn("$ref", compiled.field_bullets)
        self.assertIn("'pending'", compiled.field_bullets)
        self.assertIn("'low'", compiled.field_bullets)
        self.assertIn("from 0.0 to 1.0", compiled.field_bullets)
        self.assertIn("SubItem objects with fields:", compiled.field_bullets)

        sample = json.loads(compiled.sample_json)
        self.assertEqual(sample["category"], "pending")
        self.assertEqual(sample["priority"], "low")
        self.assertIsInstance(sample["items"], list)

        block = compiled.render_prompt_block(as_array=True)
        self.assertIn("JSON array of objects", block)

    def test_output_parser_fences_and_trailing_commas(self):
        raw = """Here is the output:
```json
[
  {"name": "Alpha", "value": 10,},
  {"name": "Beta", "value": 20}
]
```"""
        res = UniversalOutputParser.parse(raw, model_cls=StrictItem)
        self.assertTrue(res.success)
        self.assertEqual(len(res.items), 2)
        self.assertEqual(res.items[0].name, "Alpha")
        self.assertEqual(res.items[1].value, 20)

    def test_phantom_object_filtering(self):
        self.assertTrue(is_phantom_object({"a": None, "b": None}))
        raw = '[{"name": null, "score": null}, {"name": "Valid", "score": 0.9}]'
        res = UniversalOutputParser.parse(raw, model_cls=NullableItem)
        self.assertTrue(res.success)
        self.assertEqual(len(res.items), 1)
        self.assertEqual(res.items[0].name, "Valid")

    def test_container_unwrapping_and_unstructured_fallback(self):
        wrapped = '{"items": [{"name": "Delta", "value": 4}]}'
        res = UniversalOutputParser.parse(wrapped, model_cls=StrictItem)
        self.assertTrue(res.success)
        self.assertEqual(res.items[0].name, "Delta")

        bullets = "- First\n- Second"
        unstructured = UniversalOutputParser.parse(bullets, model_cls=None)
        self.assertEqual(unstructured.items, ["First", "Second"])

        sanitized = sanitize_json_output("```json\n{\"a\": 1,}\n```")
        self.assertEqual(sanitized, {"a": 1})

    def test_reasoning_order_think_tags_and_truncated_json_salvage(self):
        # 1. ClassificationItem puts 'reasoning' before 'category' to avoid the token-1 reasoning penalty
        field_names = list(ClassificationItem.model_fields.keys())
        self.assertEqual(field_names[0], "reasoning")
        self.assertEqual(field_names[1], "category")

        # 2. <think>...</think> blocks containing brackets do not corrupt JSON extraction
        think_raw = (
            "<think>\nLet's check array[0] and dict {\"temp\": 1} first.\n</think>\n"
            '[{"name": "Gamma", "value": 42}]'
        )
        res_think = UniversalOutputParser.parse(think_raw, model_cls=StrictItem)
        self.assertTrue(res_think.success)
        self.assertEqual(len(res_think.items), 1)
        self.assertEqual(res_think.items[0].name, "Gamma")

        # 3. Truncated JSON array cut off mid-item by max_new_tokens salvages completed items
        truncated_raw = '[{"name": "One", "value": 1}, {"name": "Two", "value": 2}, {"name": "CutOff", "val'
        res_trunc = UniversalOutputParser.parse(truncated_raw, model_cls=StrictItem)
        self.assertTrue(res_trunc.success)
        self.assertEqual([item.name for item in res_trunc.items], ["One", "Two"])


if __name__ == "__main__":
    unittest.main()

