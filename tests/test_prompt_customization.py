"""Tests for in-memory prompt customization in PrismRunner."""

import unittest
from unittest.mock import MagicMock
from pydantic import BaseModel, Field
from prismrunner.pipeline import PrismPipeline
from prismrunner.mapping_pipeline import SchemaMapPipeline
from prismrunner.loaders.dataset import Record
from prismrunner.catalog.registry import SchemaCatalog


class CustomMedication(BaseModel):
    drug_name: str = Field(..., description="Trade or generic drug name.")
    dosage: str = Field(..., description="Dosage and frequency.")


class TestPromptCustomization(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrismPipeline()

    def test_instruction_and_variables_override(self):
        records = [Record(id="rec_1", index=0, text="Patient was prescribed Metformin 500mg daily.")]
        payloads = self.pipeline.run(
            task="extract",
            input_data=records,
            schema_cls=CustomMedication,
            instruction="Extract only prescribed diabetes medications.",
            variables={"guidelines": "Exclude over-the-counter vitamins."},
            dry_run=True,
        )

        self.assertEqual(len(payloads), 1)
        prompt_text = payloads[0]["prompt"]
        self.assertIn("Extract only prescribed diabetes medications.", prompt_text)
        self.assertIn("Exclude over-the-counter vitamins.", prompt_text)
        self.assertIn("Patient was prescribed Metformin 500mg daily.", prompt_text)
        self.assertIn("drug_name", prompt_text)

    def test_string_prompt_template_override(self):
        records = [Record(id="rec_2", index=0, text="Clinical note content goes here.")]
        custom_template = (
            "### Clinical Context:\n{{ text }}\n\n"
            "### Target Schema:\n{schema_block}\n\n"
            "### Custom Note:\nPlease adhere strictly to the schema above."
        )

        payloads = self.pipeline.run(
            task="extract",
            input_data=records,
            schema_cls=CustomMedication,
            prompt_template=custom_template,
            dry_run=True,
        )

        self.assertEqual(len(payloads), 1)
        prompt_text = payloads[0]["prompt"]
        self.assertIn("### Clinical Context:\nClinical note content goes here.", prompt_text)
        self.assertIn("### Custom Note:\nPlease adhere strictly to the schema above.", prompt_text)
        self.assertIn("drug_name", prompt_text)

    def test_dict_prompt_template_override(self):
        records = [Record(id="rec_3", index=0, text="CBC revealed hemoglobin of 14.2 g/dL.")]
        custom_dict = {
            "version": "notebook-1.0",
            "system_prompt": "You are a specialized laboratory analyzer.",
            "instruction": "Identify laboratory measurements.",
            "variables": {
                "rule": "Flag any abnormal findings.",
            },
            "prompt": "Instruction: {instruction}\nRule: {rule}\nNote: {{ text }}\n\n{schema_block}",
        }

        payloads = self.pipeline.run(
            task="extract",
            input_data=records,
            schema_cls=CustomMedication,
            prompt_template=custom_dict,
            dry_run=True,
        )

        self.assertEqual(len(payloads), 1)
        prompt_text = payloads[0]["prompt"]
        self.assertIn("Instruction: Identify laboratory measurements.", prompt_text)
        self.assertIn("Rule: Flag any abnormal findings.", prompt_text)
        self.assertIn("Note: CBC revealed hemoglobin of 14.2 g/dL.", prompt_text)

    def test_system_prompt_override_in_execution(self):
        records = [Record(id="rec_4", index=0, text="Sample text")]
        mock_engine = MagicMock()
        mock_engine.run_all.return_value = [
            {"record_id": "rec_4", "record_index": 0, "chunk_index": 0, "raw_output": "[]"}
        ]
        self.pipeline.engine = mock_engine

        self.pipeline.run(
            task="extract",
            input_data=records,
            schema_cls=CustomMedication,
            system_prompt="Custom System Prompt: Return strict JSON only.",
        )

        mock_engine.run_all.assert_called_once()
        _, kwargs = mock_engine.run_all.call_args
        self.assertEqual(kwargs.get("system_prompt"), "Custom System Prompt: Return strict JSON only.")

    def test_prompt_template_direct(self):
        from prismrunner.prompt import PromptTemplate

        # Test in-memory string load
        tmpl_str = PromptTemplate.load("Source: {{ text }}\n{schema_block}", default_task="test")
        self.assertEqual(tmpl_str["task"], "test")

        # Test render with variable substitution and slot replacement
        rendered = PromptTemplate.render(
            template=tmpl_str,
            text="Document sample text.",
            schema_block="[JSON_SCHEMA]",
        )
        self.assertIn("Source: Document sample text.", rendered)
        self.assertIn("[JSON_SCHEMA]", rendered)

        # Test fallback path resolution
        resolved = PromptTemplate.resolve_template_path("prompts/classify.yaml")
        self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
