"""Unit tests for Deterministic Stitcher."""

import unittest
from prismrunner.stitcher.stitcher import DeterministicStitcher
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem


class TestDeterministicStitcher(unittest.TestCase):

    def test_classification_majority_voting(self):
        chunk_results = [
            {
                "record_id": "doc_1",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [ClassificationItem(category="Technology", confidence=0.9)],
            },
            {
                "record_id": "doc_1",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [ClassificationItem(category="Business", confidence=0.8)],
            },
            {
                "record_id": "doc_1",
                "record_index": 0,
                "chunk_index": 2,
                "parsed_items": [ClassificationItem(category="Technology", confidence=0.95)],
            },
        ]
        stitched = DeterministicStitcher.stitch_classification(chunk_results)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["id"], "doc_1")
        self.assertEqual(stitched[0]["category"], "Technology")
        self.assertEqual(stitched[0]["num_chunks_evaluated"], 3)
        self.assertEqual(stitched[0]["vote_distribution"]["Technology"], 2)

    def test_classification_deterministic_tie_breaking(self):
        # Tie between 'Alpha' (chunk 0) and 'Beta' (chunk 1)
        chunk_results = [
            {
                "record_id": "doc_tie",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [ClassificationItem(category="Alpha", confidence=0.9)],
            },
            {
                "record_id": "doc_tie",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [ClassificationItem(category="Beta", confidence=0.9)],
            },
        ]
        stitched = DeterministicStitcher.stitch_classification(chunk_results)
        self.assertEqual(len(stitched), 1)
        # Deterministic rule: earlier chunk_index wins tie
        self.assertEqual(stitched[0]["category"], "Alpha")

    def test_extraction_deduplication(self):
        chunk_results = [
            {
                "record_id": "doc_ext",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [
                    EntityItem(entity="PyTorch", category="Framework", details="Deep learning engine"),
                    EntityItem(entity="CUDA", category="Platform", details="NVIDIA runtime"),
                ],
            },
            {
                "record_id": "doc_ext",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [
                    # Duplicate of PyTorch in different case
                    EntityItem(entity="pytorch", category="Framework", details="Deep learning framework"),
                    EntityItem(entity="MPS", category="Platform", details="Apple Silicon runtime"),
                ],
            },
        ]
        stitched = DeterministicStitcher.stitch_extraction(chunk_results)
        self.assertEqual(len(stitched), 1)
        items = stitched[0]["extracted_items"]
        self.assertEqual(len(items), 3)  # PyTorch, CUDA, MPS
        entities = [i["entity"].lower() for i in items]
        self.assertIn("pytorch", entities)
        self.assertIn("cuda", entities)
        self.assertIn("mps", entities)

    def test_summarization_single_chunk_clean(self):
        chunk_results = [
            {
                "record_id": "doc_s1",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [SummaryItem(summary="Single section concise summary.", key_points=["Pt 1"])],
            }
        ]
        stitched = DeterministicStitcher.stitch_summarization(chunk_results)
        self.assertEqual(len(stitched), 1)
        # Single chunk should NOT have "--- [Part" synthetic header
        self.assertEqual(stitched[0]["summary"], "Single section concise summary.")
        self.assertNotIn("--- [Part", stitched[0]["summary"])

    def test_summarization_multi_chunk_hierarchical(self):
        chunk_results = [
            {
                "record_id": "doc_s2",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [SummaryItem(summary="First part synthesis.", key_points=["Pt 1"])],
            },
            {
                "record_id": "doc_s2",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [SummaryItem(summary="Second part synthesis.", key_points=["Pt 2"])],
            },
        ]
        stitched = DeterministicStitcher.stitch_summarization(chunk_results)
        self.assertEqual(len(stitched), 1)
        summary = stitched[0]["summary"]
        self.assertIn("--- [Part 1/2] ---", summary)
        self.assertIn("First part synthesis.", summary)
        self.assertIn("--- [Part 2/2] ---", summary)
        self.assertIn("Second part synthesis.", summary)
        self.assertEqual(len(stitched[0]["key_points"]), 2)


if __name__ == "__main__":
    unittest.main()
