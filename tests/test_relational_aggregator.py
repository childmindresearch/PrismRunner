"""Unit tests for Relational Normalizer, SQLite Exporter, and Consolidated Wide Table."""

import os
import sqlite3
import tempfile
import unittest
from typing import List, Optional
from pydantic import BaseModel, Field

from prismrunner.aggregator.relational import RelationalNormalizer, has_extracted_content
from prismrunner.aggregator.sqlite_export import SQLiteExporter
from prismrunner.aggregator.wide_table import WideTableBuilder


class MedicationItem(BaseModel):
    name: str = Field(description="Drug name")
    dosage: str = Field(description="Dosage amount")


class PatientEncounter(BaseModel):
    patient_id: str
    chief_complaint: str
    visit_score: int = Field(default=3)
    medications: List[MedicationItem] = Field(default_factory=list)


class TestRelationalAggregator(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_has_extracted_content(self):
        self.assertFalse(has_extracted_content(None))
        self.assertFalse(has_extracted_content(""))
        self.assertFalse(has_extracted_content({}))
        self.assertFalse(has_extracted_content({"source_id": "123", "item_index": 0}))
        self.assertTrue(has_extracted_content({"source_id": "123", "diagnosis": "Asthma"}))
        self.assertTrue(has_extracted_content("Sample Fact"))
        self.assertTrue(has_extracted_content(10))

    def test_relational_normalizer(self):
        enc = PatientEncounter(
            patient_id="P_001",
            chief_complaint="Chest pain and shortness of breath",
            visit_score=4,
            medications=[
                MedicationItem(name="Aspirin", dosage="81mg"),
                MedicationItem(name="Metoprolol", dosage="25mg"),
            ],
        )

        tables = RelationalNormalizer.normalize_record(
            data=enc,
            root_schema_name="PatientEncounter",
            source_id="doc_100",
        )

        # Expect root table and medications child table
        self.assertIn("PatientEncounter", tables)
        self.assertIn("PatientEncounter_medications", tables)

        root_rows = tables["PatientEncounter"]
        self.assertEqual(len(root_rows), 1)
        self.assertEqual(root_rows[0]["source_id"], "doc_100")
        self.assertEqual(root_rows[0]["patient_id"], "P_001")
        self.assertEqual(root_rows[0]["visit_score"], 4)

        child_rows = tables["PatientEncounter_medications"]
        self.assertEqual(len(child_rows), 2)
        self.assertEqual(child_rows[0]["source_id"], "doc_100")
        self.assertEqual(child_rows[0]["item_index"], 0)
        self.assertEqual(child_rows[0]["name"], "Aspirin")
        self.assertEqual(child_rows[1]["name"], "Metoprolol")

    def test_sqlite_exporter(self):
        enc = PatientEncounter(
            patient_id="P_002",
            chief_complaint="Headache",
            medications=[MedicationItem(name="Ibuprofen", dosage="400mg")],
        )
        tables = RelationalNormalizer.normalize_record(
            data=enc,
            root_schema_name="PatientEncounter",
            source_id="doc_200",
        )

        db_path = os.path.join(self.temp_dir.name, "test.db")
        SQLiteExporter.export(tables, db_path)

        self.assertTrue(os.path.exists(db_path))

        # Query database and verify foreign key joins
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT patient_id, chief_complaint FROM PatientEncounter WHERE source_id = 'doc_200'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "P_002")

        cursor.execute("SELECT name, dosage FROM PatientEncounter_medications WHERE source_id = 'doc_200'")
        med_row = cursor.fetchone()
        self.assertIsNotNone(med_row)
        self.assertEqual(med_row[0], "Ibuprofen")
        conn.close()

    def test_sqlite_exporter_custom_id_column(self):
        tables = {
            "Patient": [{"mrn": "12345", "diagnosis": "Hypertension"}]
        }
        db_path = os.path.join(self.temp_dir.name, "custom_id.db")
        SQLiteExporter.export(tables, db_path, id_column_name="mrn")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='Patient'")
        indexes = [row[0] for row in cursor.fetchall()]
        self.assertIn("idx_Patient_mrn", indexes)
        conn.close()

    def test_wide_table_builder(self):
        enc1 = PatientEncounter(
            patient_id="P_001",
            chief_complaint="Flu",
            medications=[MedicationItem(name="Tamiflu", dosage="75mg")],
        )
        enc2 = PatientEncounter(
            patient_id="P_002",
            chief_complaint="Cough",
            medications=[
                MedicationItem(name="Guaifenesin", dosage="200mg"),
                MedicationItem(name="Honey", dosage="1tbsp"),
            ],
        )

        tables1 = RelationalNormalizer.normalize_record(enc1, "Encounter", "doc_1")
        tables2 = RelationalNormalizer.normalize_record(enc2, "Encounter", "doc_2")

        # Merge tables
        all_tables = {
            "Encounter": tables1["Encounter"] + tables2["Encounter"],
            "Encounter_medications": tables1["Encounter_medications"] + tables2["Encounter_medications"],
        }

        df = WideTableBuilder.build(all_tables)

        # Must have exactly 2 rows (1 row per document ID)
        self.assertEqual(len(df), 2)
        self.assertIn("source_id", df.columns)
        self.assertIn("Encounter_patient_id", df.columns)
        self.assertIn("num_Encounter_medications", df.columns)
        self.assertIn("Encounter_medications_name", df.columns)

        # Check counts
        doc1_row = df[df["source_id"] == "doc_1"].iloc[0]
        self.assertEqual(doc1_row["num_Encounter_medications"], 1)

        doc2_row = df[df["source_id"] == "doc_2"].iloc[0]
        self.assertEqual(doc2_row["num_Encounter_medications"], 2)
        self.assertIn("Guaifenesin", doc2_row["Encounter_medications_name"])
        self.assertIn("Honey", doc2_row["Encounter_medications_name"])

    def test_relational_normalizer_custom_id_column_and_separator(self):
        enc = PatientEncounter(
            patient_id="P_005",
            chief_complaint="Migraine",
            medications=[MedicationItem(name="Sumatriptan", dosage="50mg")],
        )

        dataset = [
            {
                "mrn": "patient_99",
                "schema_name": "Encounter",
                "data": enc,
            }
        ]

        tables = RelationalNormalizer.normalize_dataset(
            dataset,
            id_column_name="mrn",
            child_separator="__",
        )

        self.assertIn("Encounter", tables)
        self.assertIn("Encounter__medications", tables)
        self.assertEqual(tables["Encounter"][0]["mrn"], "patient_99")
        self.assertEqual(tables["Encounter__medications"][0]["mrn"], "patient_99")
        self.assertEqual(tables["Encounter__medications"][0]["name"], "Sumatriptan")

    def test_wide_table_builder_custom_id_column_and_separator(self):
        tables = {
            "Encounter": [{"mrn": "rec_1", "chief_complaint": "Allergy"}],
            "Encounter_meds": [
                {"mrn": "rec_1", "item_index": 0, "name": "Claritin"},
                {"mrn": "rec_1", "item_index": 1, "name": "Flonase"},
            ],
        }

        df = WideTableBuilder.build(
            tables,
            id_column_name="mrn",
            list_join_char=" /// ",
        )

        self.assertEqual(len(df), 1)
        self.assertIn("mrn", df.columns)
        self.assertEqual(df.iloc[0]["mrn"], "rec_1")
        self.assertEqual(df.iloc[0]["Encounter_meds_name"], "Claritin /// Flonase")


if __name__ == "__main__":
    unittest.main()
