"""End-to-end integration tests for PrismPipeline."""

import unittest
from unittest.mock import MagicMock
from prismrunner.pipeline import PrismPipeline
from prismrunner.loaders.dataset import Record
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem


class TestPipelineE2E(unittest.TestCase):

    def setUp(self):
        self.pipeline = PrismPipeline()

    def test_dry_run_classify_pipeline(self):
        input_records = [
            Record(id="test_001", index=0, text="Deep learning model training on Apple Silicon MPS."),
            Record(id="test_002", index=1, text="Quarterly earnings and financial growth report."),
        ]

        payloads = self.pipeline.run(
            task="classify",
            input_data=input_records,
            dry_run=True,
            variables={"categories": "AI/ML, Finance, Health"},
        )

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["record_id"], "test_001")
        self.assertIn("AI/ML, Finance, Health", payloads[0]["prompt"])
        self.assertIn("Deep learning model training on Apple Silicon MPS.", payloads[0]["prompt"])
        self.assertIn("Return ONLY a valid JSON array of objects", payloads[0]["prompt"])

    def test_dry_run_with_chunking(self):
        long_text = "Paragraph analysis. " * 30
        input_records = [
            Record(id="chunked_001", index=0, text=long_text),
        ]

        payloads = self.pipeline.run(
            task="summarize",
            input_data=input_records,
            chunk=True,
            chunk_size=20,
            chunk_overlap=5,
            dry_run=True,
        )

        self.assertGreater(len(payloads), 1)
        for p in payloads:
            self.assertEqual(p["record_id"], "chunked_001")
            self.assertEqual(p["total_chunks"], len(payloads))

    def test_end_to_end_mocked_inference_and_stitching(self):
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.run_all.return_value = [
            {
                "record_id": "rec_1",
                "record_index": 0,
                "chunk_index": 0,
                "total_chunks": 2,
                "text": "Part 1 text",
                "raw_output": '```json\n[{"category": "Technology", "confidence": 0.95, "reasoning": "Mentions GPUs"}]\n```',
                "metadata": {},
            },
            {
                "record_id": "rec_1",
                "record_index": 0,
                "chunk_index": 1,
                "total_chunks": 2,
                "text": "Part 2 text",
                "raw_output": '```json\n[{"category": "Technology", "confidence": 0.90, "reasoning": "Mentions MPS"}]\n```',
                "metadata": {},
            },
        ]
        self.pipeline.engine = mock_engine

        # We can simulate the pipeline inference stage directly
        raw_results = mock_engine.run_all.return_value
        for res in raw_results:
            from prismrunner.schemas.parser import UniversalOutputParser
            parse_res = UniversalOutputParser.parse(res["raw_output"], model_cls=ClassificationItem)
            res["parsed_items"] = parse_res.items

        from prismrunner.stitcher.stitcher import DeterministicStitcher
        stitched = DeterministicStitcher.stitch_classification(raw_results)

        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["id"], "rec_1")
        self.assertEqual(stitched[0]["category"], "Technology")
        self.assertAlmostEqual(stitched[0]["confidence"], 0.925, places=2)
        self.assertEqual(stitched[0]["num_chunks_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
