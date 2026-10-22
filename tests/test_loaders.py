"""Unit tests for Universal Dataset Loaders."""

import json
import os
import tempfile
import unittest
import pandas as pd

from prismrunner.loaders.dataset import DatasetLoader, export_results


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_csv_with_id_and_text(self):
        csv_path = os.path.join(self.temp_dir.name, "sample.csv")
        df = pd.DataFrame([
            {"id": "doc_101", "text": "First test document.", "label": "A"},
            {"id": "doc_102", "text": "Second test document.", "label": "B"},
        ])
        df.to_csv(csv_path, index=False)

        records = DatasetLoader.load(csv_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].id, "doc_101")
        self.assertEqual(records[0].text, "First test document.")
        self.assertEqual(records[0].metadata["label"], "A")
        self.assertEqual(records[1].id, "doc_102")

    def test_load_json_array(self):
        json_path = os.path.join(self.temp_dir.name, "sample.json")
        data = [
            {"id": "j_1", "content": "JSON document 1"},
            {"id": "j_2", "content": "JSON document 2"},
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        records = DatasetLoader.load(json_path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].id, "j_1")
        self.assertEqual(records[0].text, "JSON document 1")

    def test_direct_raw_string(self):
        raw_text = "This is a direct string passed into the inference engine."
        records = DatasetLoader.load(raw_text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "rec_000001")
        self.assertEqual(records[0].text, raw_text)

    def test_load_markdown_file(self):
        md_path = os.path.join(self.temp_dir.name, "notes.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Clinical Notes\nPatient reports persistent cough.")
        records = DatasetLoader.load(md_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "notes_0")
        self.assertIn("persistent cough", records[0].text)

    def test_load_pandas_dataframe_direct(self):
        df = pd.DataFrame({
            "patient_id": ["PT-01", "PT-02"],
            "clinical_note": ["Patient has fever.", "Patient has rash."],
            "age": [45, 62],
        })
        records = DatasetLoader.load(df, text_column="clinical_note", id_column="patient_id")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].id, "PT-01")
        self.assertEqual(records[0].text, "Patient has fever.")
        self.assertEqual(records[0].metadata["age"], 45)

    def test_load_list_of_dicts_direct(self):
        data = [
            {"id": "d1", "text": "Item 1", "category": "A"},
            {"id": "d2", "text": "Item 2", "category": "B"},
        ]
        records = DatasetLoader.load(data)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].id, "d1")
        self.assertEqual(records[0].text, "Item 1")

    def test_load_list_of_strings_direct(self):
        notes = ["First simple note.", "Second simple note."]
        records = DatasetLoader.load(notes)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].id, "rec_000001")
        self.assertEqual(records[0].text, "First simple note.")
        self.assertEqual(records[1].id, "rec_000002")

    def test_load_descriptive_error_missing_text_column(self):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        with self.assertRaises(ValueError) as ctx:
            DatasetLoader.load(df, text_column="non_existent_column")
        self.assertIn("non_existent_column", str(ctx.exception))
        self.assertIn("Available columns are", str(ctx.exception))

    def test_load_descriptive_error_unresolved_text_column(self):
        # Multiple columns, none match standard names
        df = pd.DataFrame({"alpha_field": [1, 2], "beta_field": [3, 4]})
        with self.assertRaises(ValueError) as ctx:
            DatasetLoader.load(df)
        self.assertIn("Could not auto-detect text column", str(ctx.exception))
        self.assertIn("Please explicitly specify text_column", str(ctx.exception))

    def test_load_metadata_and_exclude_columns(self):
        df = pd.DataFrame({
            "id": ["1"],
            "text": ["content"],
            "keep_meta": ["keep_me"],
            "drop_meta": ["drop_me"],
        })
        records = DatasetLoader.load(df, metadata_columns=["keep_meta"])
        self.assertIn("keep_meta", records[0].metadata)
        self.assertNotIn("drop_meta", records[0].metadata)

        records2 = DatasetLoader.load(df, exclude_columns=["drop_meta"])
        self.assertIn("keep_meta", records2[0].metadata)
        self.assertNotIn("drop_meta", records2[0].metadata)

    def test_export_results_json_and_csv(self):
        data = [
            {"id": "out_1", "index": 0, "category": "Tech"},
            {"id": "out_2", "index": 1, "category": "Science"},
        ]
        json_out = os.path.join(self.temp_dir.name, "out.json")
        export_results(data, json_out, output_format="json")
        self.assertTrue(os.path.exists(json_out))

        csv_out = os.path.join(self.temp_dir.name, "out.csv")
        export_results(data, csv_out, output_format="csv")
        self.assertTrue(os.path.exists(csv_out))

    def test_load_single_dict(self):
        single_dict = {"id": "rec_single", "text": "Single document content", "author": "Dr. Smith"}
        records = DatasetLoader.load(single_dict)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, "rec_single")
        self.assertEqual(records[0].text, "Single document content")
        self.assertEqual(records[0].metadata.get("author"), "Dr. Smith")


if __name__ == "__main__":
    unittest.main()
