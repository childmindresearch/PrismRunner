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

    def test_mapping_cross_chunk_deduplication(self):
        from unittest.mock import MagicMock
        import tempfile
        from pathlib import Path

        catalog = SchemaCatalog()
        catalog.register(Diagnosis)

        pipeline = SchemaMappingPipeline(catalog=catalog)
        # Mock engine to return duplicate diagnosis from two chunks of the same doc
        mock_engine = MagicMock()
        mock_engine.run_all.return_value = [
            {"record_id": "doc_001", "chunk_index": 0, "raw_output": '[{"condition": "Asthma"}]'},
            {"record_id": "doc_001", "chunk_index": 1, "raw_output": '[{"condition": "Asthma"}]'},
        ]
        pipeline.base_pipeline.engine = mock_engine

        with tempfile.TemporaryDirectory() as tmp_dir:
            records = [
                Record(id="doc_001", index=0, text="Patient has asthma. More notes about asthma."),
            ]
            summary = pipeline.run(
                input_data=records,
                output_dir=tmp_dir,
                use_triage=False,
                resume=False,
                chunk_size=10,
                export_formats=["csv"],
            )
            # Check the relational CSV has exactly 1 row for Asthma, not 2
            diag_csv = Path(tmp_dir) / "relational_csv" / "Diagnosis.csv"
            self.assertTrue(diag_csv.exists())
            import pandas as pd
            df = pd.read_csv(diag_csv)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["condition"], "Asthma")


if __name__ == "__main__":
    unittest.main()
