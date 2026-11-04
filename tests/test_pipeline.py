"""Integration tests for PrismPipeline and prompt customization."""

import unittest
from unittest.mock import MagicMock
from pydantic import BaseModel, Field

from prismrunner import PrismPipeline, Record


class Medication(BaseModel):
    drug_name: str = Field(description="Medication name")
    dosage: str = Field(description="Dosage strength")


class TestPrismPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrismPipeline()

    def test_dry_run_and_custom_schema_prompt(self):
        payloads = self.pipeline.run(
            task="extract",
            input_data=["Patient prescribed Metformin 500mg daily."],
            schema_cls=Medication,
            instruction="Extract diabetes medications.",
            dry_run=True,
        )
        self.assertEqual(len(payloads), 1)
        self.assertIn("Extract diabetes medications.", payloads[0]["prompt"])
        self.assertIn("drug_name", payloads[0]["prompt"])
        self.assertIn("Metformin 500mg", payloads[0]["prompt"])

    def test_custom_string_and_dict_prompt_templates(self):
        payloads = self.pipeline.run(
            task="extract",
            input_data=[Record(id="r1", index=0, text="Sample note")],
            schema_cls=Medication,
            prompt_template="Note: {{ text }}\n{schema_block}",
            dry_run=True,
        )
        self.assertIn("Note: Sample note", payloads[0]["prompt"])

    def test_end_to_end_mocked_inference(self):
        mock_engine = MagicMock()
        mock_engine.run_all.return_value = [
            {
                "record_id": "rec_1",
                "record_index": 0,
                "chunk_index": 0,
                "total_chunks": 1,
                "text": "Metformin 500mg",
                "raw_output": '[{"drug_name": "Metformin", "dosage": "500mg"}]',
                "metadata": {},
            }
        ]
        self.pipeline.engine = mock_engine

        results = self.pipeline.run(
            task="extract",
            input_data=[Record(id="rec_1", index=0, text="Metformin 500mg")],
            schema_cls=Medication,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["extracted_items"], [{"drug_name": "Metformin", "dosage": "500mg"}])


if __name__ == "__main__":
    unittest.main()

