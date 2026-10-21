"""Token-aware sliding window chunker supporting sentence boundaries and strict token splitting."""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from prismrunner.loaders.dataset import Record

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """Represents a chunk slice of a parent record with deterministic lineage tracking."""

    record_id: str
    record_index: int
    chunk_index: int
    total_chunks: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenAwareChunker:
    """Splits texts into token-budgeted sliding window chunks."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        strategy: str = "sentence",
    ):
        self.tokenizer = tokenizer
        self.chunk_size = max(4, chunk_size)
        self.chunk_overlap = min(max(0, chunk_overlap), max(0, self.chunk_size // 2))
        self.strategy = strategy.lower().strip()
        if self.strategy not in ("sentence", "token"):
            self.strategy = "sentence"

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer or fast word/char heuristic fallback."""
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback estimation: ~4 chars per token
        return max(1, len(text) // 4)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving boundary punctuation."""
        # Regex splits on sentence-ending punctuation followed by whitespace and uppercase/number
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])"
        raw_sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Strict token-level sliding window splitting."""
        if self.tokenizer is None:
            # Word-based sliding window fallback
            words = text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            return [" ".join(words[i : i + self.chunk_size]) for i in range(0, len(words), step)]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: List[str] = []

        for i in range(0, len(token_ids), step):
            window_ids = token_ids[i : i + self.chunk_size]
            chunk_str = self.tokenizer.decode(window_ids, skip_special_tokens=True).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Semantic sentence boundary chunking with token overlap."""
        sentences = self._split_sentences(text)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self.count_tokens(sentence)

            # If a single sentence exceeds chunk_size, split that sentence via token strategy
            if s_tokens > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                sub_chunks = self._chunk_by_tokens(sentence)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + s_tokens <= self.chunk_size:
                current_sentences.append(sentence)
                current_tokens += s_tokens
            else:
                # Flush current chunk
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

                # Implement overlap by retaining tail sentences within chunk_overlap tokens
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for prev in reversed(current_sentences):
                    prev_tok = self.count_tokens(prev)
                    if overlap_tokens + prev_tok <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev)
                        overlap_tokens += prev_tok
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_sentences)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def chunk_record(self, record: Record) -> List[ChunkRecord]:
        """Split a Record into ChunkRecords, retaining record IDs and chronological indexes."""
        text = record.text.strip()
        total_toks = self.count_tokens(text)

        # Optimization: if text already fits within chunk_size, avoid unnecessary fragmentation
        if total_toks <= self.chunk_size:
            return [
                ChunkRecord(
                    record_id=record.id,
                    record_index=record.index,
                    chunk_index=0,
                    total_chunks=1,
                    text=text,
                    token_count=total_toks,
                    metadata=record.metadata,
                )
            ]

        if self.strategy == "sentence":
            raw_chunks = self._chunk_by_sentences(text)
        else:
            raw_chunks = self._chunk_by_tokens(text)

        if not raw_chunks:
            raw_chunks = [text]

        total_chunks = len(raw_chunks)
        chunk_records: List[ChunkRecord] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_records.append(
                ChunkRecord(
                    record_id=record.id,
                    record_index=record.index,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    text=chunk_text,
                    token_count=self.count_tokens(chunk_text),
                    metadata=record.metadata,
                )
            )

        return chunk_records

    def chunk_dataset(self, records: List[Record]) -> List[ChunkRecord]:
        """Process all records into ChunkRecords."""
        all_chunks: List[ChunkRecord] = []
        for rec in records:
            all_chunks.extend(self.chunk_record(rec))
        logger.info(f"Chunked {len(records)} records into {len(all_chunks)} total chunks.")
        return all_chunks
