"""Unit tests for prismrunner.data (DatasetLoader, TokenAwareChunker, and DeterministicStitcher)."""

import os
import tempfile
import unittest
import pandas as pd

from prismrunner.data import (
    DatasetLoader,
    DeterministicStitcher,
    Record,
    TokenAwareChunker,
    export_results,
)
from prismrunner.schema import ClassificationItem, EntityItem, SummaryItem


class TestDataComponents(unittest.TestCase):

    def test_dataset_loader_inputs_and_export(self):
        # DataFrame input
        df = pd.DataFrame({
            "patient_id": ["PT-1", "PT-2"],
            "clinical_note": ["Note A", "Note B"],
            "age": [30, 40],
        })
        recs = DatasetLoader.load(df, text_column="clinical_note", id_column="patient_id")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].id, "PT-1")
        self.assertEqual(recs[0].metadata["age"], 30)

        # List of strings and single string
        self.assertEqual(len(DatasetLoader.load(["A", "B"])), 2)
        self.assertEqual(DatasetLoader.load("Direct string")[0].text, "Direct string")

        # File round-trip
        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "out.json")
            export_results([{"id": "1", "text": "hello"}], out_json)
            loaded = DatasetLoader.load(out_json)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].text, "hello")

    def test_token_aware_chunker(self):
        long_text = (
            "Sentence one introduces the primary concept clearly. "
            "Sentence two elaborates on technical constraints and implementation details. "
            "Sentence three examines edge cases and performance tradeoffs."
        )
        chunker = TokenAwareChunker(tokenizer=None, chunk_size=12, chunk_overlap=4, strategy="sentence")
        chunks = chunker.chunk_record(Record(id="r1", index=0, text=long_text))
        self.assertGreater(len(chunks), 1)
        for idx, ch in enumerate(chunks):
            self.assertEqual(ch.record_id, "r1")
            self.assertEqual(ch.chunk_index, idx)
            self.assertEqual(ch.total_chunks, len(chunks))

    def test_deterministic_stitcher(self):
        # 1. Classification majority voting
        cls_chunks = [
            {"record_id": "d1", "record_index": 0, "chunk_index": 0, "parsed_items": [ClassificationItem(category="Tech", confidence=0.9)]},
            {"record_id": "d1", "record_index": 0, "chunk_index": 1, "parsed_items": [ClassificationItem(category="Biz", confidence=0.8)]},
            {"record_id": "d1", "record_index": 0, "chunk_index": 2, "parsed_items": [ClassificationItem(category="Tech", confidence=0.95)]},
        ]
        cls_res = DeterministicStitcher.stitch_classification(cls_chunks)
        self.assertEqual(cls_res[0]["category"], "Tech")

        # 2. Extraction deduplication
        ext_chunks = [
            {"record_id": "d1", "record_index": 0, "chunk_index": 0, "parsed_items": [EntityItem(entity="PyTorch", category="Lib")]},
            {"record_id": "d1", "record_index": 0, "chunk_index": 1, "parsed_items": [EntityItem(entity="pytorch", category="Lib"), EntityItem(entity="CUDA", category="GPU")]},
        ]
        ext_res = DeterministicStitcher.stitch_extraction(ext_chunks)
        self.assertEqual(ext_res[0]["total_items_extracted"], 2)

        # 3. Summarization stitching
        sum_chunks = [
            {"record_id": "d1", "record_index": 0, "chunk_index": 0, "parsed_items": [SummaryItem(summary="Part A", key_points=["P1"])]},
            {"record_id": "d1", "record_index": 0, "chunk_index": 1, "parsed_items": [SummaryItem(summary="Part B", key_points=["P1", "P2"])]},
        ]
        sum_res = DeterministicStitcher.stitch_summarization(sum_chunks)
        self.assertIn("--- [Part 1/2] ---", sum_res[0]["summary"])
        self.assertEqual(sum_res[0]["key_points"], ["P1", "P2"])


if __name__ == "__main__":
    unittest.main()

