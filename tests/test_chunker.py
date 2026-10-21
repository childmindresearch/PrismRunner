"""Unit tests for Token-Aware Chunker."""

import unittest
from prismrunner.chunking.chunker import TokenAwareChunker
from prismrunner.loaders.dataset import Record


class TestTokenAwareChunker(unittest.TestCase):

    def setUp(self):
        self.chunker = TokenAwareChunker(
            tokenizer=None,
            chunk_size=128,  # characters/heuristic tokens
            chunk_overlap=32,
            strategy="sentence",
        )

    def test_single_chunk_short_text(self):
        rec = Record(id="rec_001", index=0, text="This is a short single sentence.", metadata={"source": "test"})
        chunks = self.chunker.chunk_record(rec)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].record_id, "rec_001")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].total_chunks, 1)
        self.assertEqual(chunks[0].metadata["source"], "test")

    def test_multi_chunk_sentence_strategy(self):
        long_text = (
            "Sentence one introduces the primary concept clearly. "
            "Sentence two elaborates on technical constraints and implementation details. "
            "Sentence three examines edge cases and performance tradeoffs. "
            "Sentence four evaluates benchmark results and hardware utilization metrics. "
            "Sentence five concludes with architectural recommendations and next steps."
        )
        small_chunker = TokenAwareChunker(
            tokenizer=None,
            chunk_size=15,  # small threshold to force multi-chunk
            chunk_overlap=5,
            strategy="sentence",
        )
        rec = Record(id="rec_002", index=1, text=long_text)
        chunks = small_chunker.chunk_record(rec)

        self.assertGreater(len(chunks), 1)
        # Verify deterministic lineage tracking
        for idx, ch in enumerate(chunks):
            self.assertEqual(ch.record_id, "rec_002")
            self.assertEqual(ch.record_index, 1)
            self.assertEqual(ch.chunk_index, idx)
            self.assertEqual(ch.total_chunks, len(chunks))

    def test_token_strategy(self):
        long_text = "word " * 100
        token_chunker = TokenAwareChunker(
            tokenizer=None,
            chunk_size=20,
            chunk_overlap=5,
            strategy="token",
        )
        rec = Record(id="rec_003", index=2, text=long_text)
        chunks = token_chunker.chunk_record(rec)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[-1].chunk_index, len(chunks) - 1)


if __name__ == "__main__":
    unittest.main()
