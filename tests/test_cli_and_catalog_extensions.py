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


if __name__ == "__main__":
    unittest.main()
