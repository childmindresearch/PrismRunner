"""Unit tests for Dynamic Schema-to-Prompt Compiler."""

import enum
import unittest
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

from prismrunner.schemas.compiler import compile_schema_to_prompt


class StatusEnum(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TestSchema(BaseModel):
    category: StatusEnum = Field(description="Operational status")
    priority: Literal["low", "medium", "high"] = Field(description="Priority ranking")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence metric")
    count: int = Field(gt=0, le=100, description="Positive count")
    notes: Optional[str] = Field(default=None, description="Additional remarks")
    tags: List[str] = Field(default_factory=list, description="Categorical tags")


class TestSchemaCompiler(unittest.TestCase):

    def test_compile_no_raw_openapi_schemas(self):
        compiled = compile_schema_to_prompt(TestSchema)

        # CRITICAL REQUIREMENT: NEVER inject raw OpenAPI schemas with $defs, $ref, or anyOf
        self.assertNotIn("$defs", compiled.field_bullets)
        self.assertNotIn("$ref", compiled.field_bullets)
        self.assertNotIn("anyOf", compiled.field_bullets)
        self.assertNotIn("$defs", compiled.sample_json)
        self.assertNotIn("$ref", compiled.sample_json)

    def test_enum_and_literal_extraction(self):
        compiled = compile_schema_to_prompt(TestSchema)

        # Check Enum values extracted as readable options
        self.assertIn("'pending'", compiled.field_bullets)
        self.assertIn("'approved'", compiled.field_bullets)
        self.assertIn("'rejected'", compiled.field_bullets)

        # Check Literal values extracted
        self.assertIn("'low'", compiled.field_bullets)
        self.assertIn("'medium'", compiled.field_bullets)
        self.assertIn("'high'", compiled.field_bullets)

    def test_numeric_bounds(self):
        compiled = compile_schema_to_prompt(TestSchema)

        # Check bounds formatting
        self.assertIn("from 0.0 to 1.0", compiled.field_bullets)
        self.assertIn("> 0", compiled.field_bullets)
        self.assertIn("<= 100", compiled.field_bullets)

    def test_sample_json_generation(self):
        compiled = compile_schema_to_prompt(TestSchema)

        # Verify valid JSON sample
        import json
        sample = json.loads(compiled.sample_json)
        self.assertIn("category", sample)
        self.assertEqual(sample["category"], "pending")
        self.assertIn("priority", sample)
        self.assertEqual(sample["priority"], "low")
        self.assertIn("confidence", sample)
        self.assertGreaterEqual(sample["confidence"], 0.0)

    def test_render_prompt_block(self):
        compiled = compile_schema_to_prompt(TestSchema)
        rendered_array = compiled.render_prompt_block(as_array=True)
        self.assertIn("JSON array of objects", rendered_array)
        self.assertIn("```json", rendered_array)

        rendered_obj = compiled.render_prompt_block(as_array=False)
        self.assertIn("JSON object", rendered_obj)


if __name__ == "__main__":
    unittest.main()
