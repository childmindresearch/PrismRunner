"""Tests for CLI options, catalog extensions, and package fallbacks."""

import unittest
from click.testing import CliRunner
from prismrunner.cli import main
from prismrunner.catalog.registry import SchemaCatalog
from prismrunner.pipeline import PrismPipeline


class TestCliAndCatalogExtensions(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_extract_with_custom_schema_dry_run(self):
        result = self.runner.invoke(
            main,
            [
                "extract",
                "--input-text",
                "Patient prescribed 10mg Lisinopril daily.",
                "--schema",
                "schemas/clinical_records.py:Medication",
                "--dry-run",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("DRY RUN MODE", result.output)
        self.assertIn("drug_name", result.output)
        self.assertIn("dosage", result.output)

    def test_schema_catalog_init_with_directory(self):
        catalog = SchemaCatalog(schemas_dir="schemas")
        self.assertGreater(len(catalog), 0)
        self.assertIn("Medication", catalog)
        self.assertIn("Diagnosis", catalog)

    def test_package_fallback_prompt_resolution(self):
        pipeline = PrismPipeline()
        # Test loading prompt template when current path is resolved
        extract_tmpl = pipeline.load_prompt_template("extract")
        self.assertIn("prompt", extract_tmpl)
        self.assertIn("instruction", extract_tmpl)

    def test_cli_extract_with_dynamic_options_dry_run(self):
        result = self.runner.invoke(
            main,
            [
                "extract",
                "--input-text",
                "Patient allergic to penicillin.",
                "--schema",
                "schemas/clinical_records.py:Medication",
                "--system-prompt",
                "Custom extraction assistant.",
                "--single-object",
                "--no-dedup",
                "--top-p",
                "0.9",
                "--top-k",
                "50",
                "--repetition-penalty",
                "1.1",
                "--temperature",
                "0.2",
                "--max-tokens",
                "256",
                "--attn-implementation",
                "sdpa",
                "--dry-run",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("DRY RUN MODE", result.output)
        self.assertIn("drug_name", result.output)

    def test_cli_map_with_dynamic_options_dry_run(self):
        result = self.runner.invoke(
            main,
            [
                "map",
                "--input-path",
                "docs/README.md",
                "--schemas-dir",
                "schemas",
                "--chunk-strategy",
                "token",
                "--id-column-name",
                "record_identifier",
                "--wide-table-name",
                "summary_wide",
                "--sqlite-db-name",
                "custom.db",
                "--wide-separator",
                " | ",
                "--max-tokens",
                "300",
                "--temperature",
                "0.1",
                "--dry-run",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("DRY RUN MODE", result.output)
        self.assertIn("Candidate Schemas", result.output)

    def test_cli_classify_dynamic_fields_dry_run(self):
        result = self.runner.invoke(
            main,
            [
                "classify",
                "--input-text",
                "Patient exhibits acute appendicitis symptoms.",
                "--category-field",
                "label",
                "--confidence-field",
                "score",
                "--reasoning-field",
                "explanation",
                "--dry-run",
            ],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("DRY RUN MODE", result.output)


if __name__ == "__main__":
    unittest.main()
