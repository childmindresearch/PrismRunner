"""Unit tests for Universal Robust Output Parser."""

import unittest
from typing import Optional
from pydantic import BaseModel, Field

from prismrunner.schemas.parser import UniversalOutputParser, is_phantom_object


class SchemaWithDefaults(BaseModel):
    name: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None)
    score: Optional[float] = Field(default=None)


class StrictSchema(BaseModel):
    name: str
    value: int


class TestUniversalOutputParser(unittest.TestCase):

    def test_markdown_code_fence_stripping(self):
        raw = """Here is the extracted information:
```json
[
  {"name": "Apple", "value": 10},
  {"name": "Google", "value": 20}
]
```
Hope this helps!"""
        result = UniversalOutputParser.parse(raw, model_cls=StrictSchema)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].name, "Apple")
        self.assertEqual(result.items[1].value, 20)

    def test_json_lines_parsing(self):
        raw = """{"name": "Alpha", "value": 1}
{"name": "Beta", "value": 2}"""
        result = UniversalOutputParser.parse(raw, model_cls=StrictSchema)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].name, "Alpha")
        self.assertEqual(result.items[1].name, "Beta")

    def test_trailing_comma_repair(self):
        raw = """[
  {"name": "Gamma", "value": 3,},
]"""
        result = UniversalOutputParser.parse(raw, model_cls=StrictSchema)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].name, "Gamma")

    def test_discard_phantom_none_objects(self):
        # CRITICAL REQUIREMENT: Discard phantom objects where all fields are None
        raw = """[
  {"name": null, "category": null, "score": null},
  {"name": "ValidItem", "category": "General", "score": 0.95},
  {"name": null, "category": null, "score": null}
]"""
        result = UniversalOutputParser.parse(raw, model_cls=SchemaWithDefaults)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].name, "ValidItem")
        self.assertEqual(result.items[0].category, "General")

    def test_is_phantom_object_helper(self):
        self.assertTrue(is_phantom_object({"a": None, "b": None}))
        self.assertTrue(is_phantom_object({}))
        self.assertFalse(is_phantom_object({"a": None, "b": "present"}))

    def test_no_schema_clean_strings(self):
        # CRITICAL REQUIREMENT: If no schema is passed, cleanly return stripped strings
        raw = """- First item
- Second item
- Third item"""
        result = UniversalOutputParser.parse(raw, model_cls=None)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 3)
        self.assertEqual(result.items[0], "First item")
        self.assertEqual(result.items[1], "Second item")
        self.assertEqual(result.items[2], "Third item")

    def test_container_dict_unwrapping(self):
        raw = '{"items": [{"name": "Delta", "value": 40}, {"name": "Epsilon", "value": 50}]}'
        result = UniversalOutputParser.parse(raw, model_cls=StrictSchema)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].name, "Delta")
        self.assertEqual(result.items[1].name, "Epsilon")

    def test_markdown_bullet_json_strings(self):
        raw = """Here are the items:
- {"name": "ItemA", "value": 1}
- {"name": "ItemB", "value": 2}"""
        result = UniversalOutputParser.parse(raw, model_cls=StrictSchema)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].name, "ItemA")
        self.assertEqual(result.items[1].name, "ItemB")

    def test_single_field_scalar_mapping(self):
        class Diagnosis(BaseModel):
            condition: str

        raw = """- Type 2 Diabetes
- Essential Hypertension"""
        result = UniversalOutputParser.parse(raw, model_cls=Diagnosis)
        self.assertTrue(result.success)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].condition, "Type 2 Diabetes")
        self.assertEqual(result.items[1].condition, "Essential Hypertension")

    def test_sanitize_json_output_standalone(self):
        from prismrunner.schemas.parser import sanitize_json_output
        raw = """```json
        {"name": "Standalone", "value": 42}
        ```"""
        parsed = sanitize_json_output(raw)
        self.assertEqual(parsed, {"name": "Standalone", "value": 42})

        raw_list = """Preamble chatter:
```
[{"name": "Alpha", "value": 1}, {"name": "Beta", "value": 2},]
```
Postamble chatter."""
        parsed_list = sanitize_json_output(raw_list)
        self.assertEqual(parsed_list, [{"name": "Alpha", "value": 1}, {"name": "Beta", "value": 2}])


if __name__ == "__main__":
    unittest.main()
