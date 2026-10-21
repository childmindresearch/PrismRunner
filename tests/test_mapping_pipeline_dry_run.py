"""Unit tests for SchemaMappingPipeline dry-run execution."""

import unittest
from pydantic import BaseModel, Field

from prismrunner.catalog.registry import SchemaCatalog
from prismrunner.mapping_pipeline import SchemaMappingPipeline
from prismrunner.loaders.dataset import Record


class LabResult(BaseModel):
    """Laboratory test names, values, and reference flags."""
    test_name: str
    value: float
    flag: str


class Diagnosis(BaseModel):
    """Clinical diagnoses and conditions."""
    condition: str


class TestMappingPipelineDryRun(unittest.TestCase):

    def test_mapping_pipeline_dry_run(self):
        catalog = SchemaCatalog()
        catalog.register(LabResult)
        catalog.register(Diagnosis)

        pipeline = SchemaMappingPipeline(catalog=catalog)
        records = [
            Record(id="doc_001", index=0, text="Patient lab shows elevated glucose 140 mg/dL."),
            Record(id="doc_002", index=1, text="Diagnosed with Type 2 Diabetes Mellitus."),
        ]

        summary = pipeline.run(
            input_data=records,
            dry_run=True,
        )

        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["schemas"], 2)


if __name__ == "__main__":
    unittest.main()
