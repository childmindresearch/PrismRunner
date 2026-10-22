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

    def test_extraction_custom_pydantic_schema(self):
        from pydantic import BaseModel, Field

        class AdverseEvent(BaseModel):
            event_name: str
            severity: str
            action_taken: str
            resolved: bool

        chunk_results = [
            {
                "record_id": "PT-101",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [
                    AdverseEvent(event_name="moderate rash", severity="moderate", action_taken="discontinued", resolved=False),
                ],
            },
            {
                "record_id": "PT-101",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [
                    # Duplicate across chunk boundary
                    AdverseEvent(event_name="Moderate Rash", severity="moderate", action_taken="discontinued", resolved=False),
                    AdverseEvent(event_name="fever", severity="mild", action_taken="antipyretic", resolved=True),
                ],
            },
        ]
        stitched = DeterministicStitcher.stitch_extraction(chunk_results)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["record_id"], "PT-101")
        self.assertEqual(stitched[0]["total_items_extracted"], 2)
        event_names = [i["event_name"].lower() for i in stitched[0]["extracted_items"]]
        self.assertIn("moderate rash", event_names)
        self.assertIn("fever", event_names)


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

    def test_classification_custom_pydantic_schema_and_fields(self):
        from pydantic import BaseModel

        class SentimentItem(BaseModel):
            sentiment: str
            score: float
            rationale: str

        chunk_results = [
            {
                "record_id": "rev_1",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [SentimentItem(sentiment="Positive", score=0.7, rationale="Good performance")],
            },
            {
                "record_id": "rev_1",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [SentimentItem(sentiment="Negative", score=0.95, rationale="Crash occurred")],
            },
        ]

        # Test highest_confidence tie-breaker
        stitched = DeterministicStitcher.stitch_classification(
            chunk_results,
            category_field="sentiment",
            confidence_field="score",
            reasoning_field="rationale",
            tie_breaker="highest_confidence",
        )
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["category"], "Negative")
        self.assertEqual(stitched[0]["confidence"], 0.95)

    def test_summarization_custom_pydantic_schema_and_headers(self):
        from pydantic import BaseModel

        class ExecutiveBrief(BaseModel):
            overview: str
            takeaways: list

        chunk_results = [
            {
                "record_id": "rep_1",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [ExecutiveBrief(overview="Section 1 overview", takeaways=["T1"])],
            },
            {
                "record_id": "rep_1",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [ExecutiveBrief(overview="Section 2 overview", takeaways=["T2"])],
            },
        ]

        stitched = DeterministicStitcher.stitch_summarization(
            chunk_results,
            summary_field="overview",
            points_field="takeaways",
            part_header_template="### Chapter {index} of {total}",
        )
        self.assertEqual(len(stitched), 1)
        self.assertIn("### Chapter 1 of 2", stitched[0]["summary"])
        self.assertIn("Section 1 overview", stitched[0]["summary"])
        self.assertIn("### Chapter 2 of 2", stitched[0]["summary"])
        self.assertEqual(stitched[0]["key_points"], ["T1", "T2"])

    def test_extraction_custom_dedup_key_and_no_dedup(self):
        from pydantic import BaseModel

        class Finding(BaseModel):
            finding_id: str
            description: str

        chunk_results = [
            {
                "record_id": "doc_f",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [
                    Finding(finding_id="F-01", description="Mild pain"),
                    Finding(finding_id="F-02", description="High fever"),
                ],
            },
            {
                "record_id": "doc_f",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [
                    Finding(finding_id="F-01", description="Mild pain repeated in chunk 2"),
                ],
            },
        ]

        # Test custom dedup key
        stitched = DeterministicStitcher.stitch_extraction(
            chunk_results,
            dedup=True,
            dedup_key="finding_id",
        )
        self.assertEqual(stitched[0]["total_items_extracted"], 2)

        # Test dedup=False
        stitched_all = DeterministicStitcher.stitch_extraction(
            chunk_results,
            dedup=False,
        )
        self.assertEqual(stitched_all[0]["total_items_extracted"], 3)

    def test_stitch_extraction_empty_warning_on_parse_error(self):
        chunk_results = [
            {
                "record_id": "err_doc",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [],
                "parse_error": "JSONDecodeError: Unterminated string",
            }
        ]
        stitched = DeterministicStitcher.stitch_extraction(chunk_results)
        self.assertEqual(len(stitched), 1)
        self.assertEqual(stitched[0]["total_items_extracted"], 0)
        self.assertIn("warning", stitched[0])
        self.assertIn("JSONDecodeError", stitched[0]["warning"])

    def test_stitch_summarization_dedup_key_points(self):
        chunk_results = [
            {
                "record_id": "rep_dedup",
                "record_index": 0,
                "chunk_index": 0,
                "parsed_items": [{"summary": "Part 1", "key_points": ["Point A", "Point B"]}],
            },
            {
                "record_id": "rep_dedup",
                "record_index": 0,
                "chunk_index": 1,
                "parsed_items": [{"summary": "Part 2", "key_points": ["Point B", "Point C"]}],
            },
        ]
        stitched = DeterministicStitcher.stitch_summarization(chunk_results)
        self.assertEqual(stitched[0]["key_points"], ["Point A", "Point B", "Point C"])


if __name__ == "__main__":
    unittest.main()

