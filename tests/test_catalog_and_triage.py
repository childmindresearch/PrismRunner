"""Unit tests for Schema Catalog and Triage Pre-Screening."""

import unittest
from pydantic import BaseModel, Field

from prismrunner.catalog.registry import SchemaCatalog
from prismrunner.triage.filter import SchemaTriageFilter, TriageResult
from prismrunner.schemas.parser import UniversalOutputParser


class DiagnosisSchema(BaseModel):
    """Clinical diagnoses and diagnostic codes."""
    condition: str
    icd10: str


class MedicationSchema(BaseModel):
    """Prescribed pharmaceutical drugs and dosages."""
    drug: str
    dosage: str


class TestCatalogAndTriage(unittest.TestCase):

    def test_catalog_registration(self):
        catalog = SchemaCatalog()
        catalog.register(DiagnosisSchema)
        catalog.register(MedicationSchema)

        self.assertEqual(len(catalog), 2)
        self.assertIn("DiagnosisSchema", catalog)
        self.assertIn("MedicationSchema", catalog)

        summary = catalog.build_triage_summary()
        self.assertIn('"DiagnosisSchema": Clinical diagnoses and diagnostic codes.', summary)
        self.assertIn('"MedicationSchema": Prescribed pharmaceutical drugs and dosages.', summary)

    def test_triage_prompt_construction(self):
        catalog = SchemaCatalog()
        catalog.register(DiagnosisSchema)
        catalog.register(MedicationSchema)

        triage = SchemaTriageFilter(catalog)
        prompt = triage.build_triage_prompt("Patient presents with acute bronchitis and asthma.")

        self.assertIn("Candidate Schemas Catalog:", prompt)
        self.assertIn("DiagnosisSchema", prompt)
        self.assertIn("MedicationSchema", prompt)
        self.assertIn("acute bronchitis and asthma", prompt)

    def test_triage_result_parsing(self):
        raw_output = """```json
{
  "relevant_schemas": ["DiagnosisSchema"]
}
```"""
        parsed = UniversalOutputParser.parse(raw_output, model_cls=TriageResult)
        self.assertTrue(parsed.success)
        self.assertEqual(len(parsed.items), 1)
        res: TriageResult = parsed.items[0]
        self.assertEqual(res.relevant_schemas, ["DiagnosisSchema"])


if __name__ == "__main__":
    unittest.main()
