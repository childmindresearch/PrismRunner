"""Deterministic stitcher for aggregating chunk outputs back into record-level results."""

from __future__ import annotations
import collections
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from prismrunner.chunking.chunker import ChunkRecord
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem

logger = logging.getLogger(__name__)


@dataclass
class StitchedRecord:
    """Consolidated record result after deterministic aggregation."""

    id: str
    index: int
    task: str
    result: Any
    num_chunks: int
    metadata: Dict[str, Any]


class DeterministicStitcher:
    """Stitches multi-chunk inference outputs into unified record results."""

    @staticmethod
    def stitch_classification(
        chunk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate classification predictions via majority voting with deterministic tie-breaking.
        
        Tie-breaking rule:
        If multiple categories share the maximum vote count, select the category
        predicted by the earliest chronological chunk (smallest chunk_index).
        """
        # Group by record_id
        grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in chunk_results:
            grouped[item["record_id"]].append(item)

        stitched: List[Dict[str, Any]] = []

        for rec_id, chunks in grouped.items():
            # Sort chronologically by chunk_index
            chunks.sort(key=lambda c: c["chunk_index"])
            total_chunks = len(chunks)
            base_meta = chunks[0].get("metadata", {})
            rec_idx = chunks[0]["record_index"]

            # Collect categories
            votes: collections.Counter = collections.Counter()
            earliest_chunk_for_cat: Dict[str, int] = {}
            confidences: Dict[str, List[float]] = collections.defaultdict(list)
            all_reasons: List[str] = []

            for ch in chunks:
                parsed_items = ch.get("parsed_items", [])
                for pi in parsed_items:
                    cat = None
                    conf = 1.0
                    reason = None

                    if isinstance(pi, ClassificationItem):
                        cat = pi.category.strip()
                        conf = pi.confidence
                        reason = pi.reasoning
                    elif isinstance(pi, dict):
                        cat = str(pi.get("category", "")).strip()
                        conf = float(pi.get("confidence", 1.0))
                        reason = pi.get("reasoning")
                    elif isinstance(pi, str):
                        cat = pi.strip()

                    if cat:
                        votes[cat] += 1
                        confidences[cat].append(conf)
                        if cat not in earliest_chunk_for_cat:
                            earliest_chunk_for_cat[cat] = ch["chunk_index"]
                        if reason:
                            all_reasons.append(reason)

            if not votes:
                winning_cat = "Unknown"
                avg_conf = 0.0
            else:
                max_count = max(votes.values())
                candidates = [c for c, count in votes.items() if count == max_count]

                # Deterministic tie-breaking: pick the category seen in the earliest chunk
                winning_cat = min(candidates, key=lambda c: earliest_chunk_for_cat.get(c, 0))
                cat_confs = confidences.get(winning_cat, [1.0])
                avg_conf = round(sum(cat_confs) / len(cat_confs), 3)

            rec_output = {
                "id": rec_id,
                "index": rec_idx,
                "category": winning_cat,
                "confidence": avg_conf,
                "num_chunks_evaluated": total_chunks,
                "vote_distribution": dict(votes),
            }
            if all_reasons:
                rec_output["reasoning"] = "; ".join(all_reasons[:3])
            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched

    @staticmethod
    def stitch_extraction(
        chunk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate extracted entities across chunks with case-insensitive deduplication."""
        grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in chunk_results:
            grouped[item["record_id"]].append(item)

        stitched: List[Dict[str, Any]] = []

        for rec_id, chunks in grouped.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            base_meta = chunks[0].get("metadata", {})
            rec_idx = chunks[0]["record_index"]

            seen_entities: set = set()
            unified_items: List[Dict[str, Any]] = []

            for ch in chunks:
                parsed_items = ch.get("parsed_items", [])
                for pi in parsed_items:
                    entity_dict = None
                    if isinstance(pi, BaseModel):
                        entity_dict = pi.model_dump()
                    elif isinstance(pi, dict):
                        entity_dict = pi
                    elif isinstance(pi, str) and pi.strip():
                        entity_dict = {"entity": pi.strip(), "category": "General"}

                    if entity_dict:
                        key_val = str(entity_dict.get("entity", entity_dict.get("name", ""))).strip().lower()
                        if key_val and key_val not in seen_entities:
                            seen_entities.add(key_val)
                            unified_items.append(entity_dict)

            rec_output = {
                "id": rec_id,
                "index": rec_idx,
                "extracted_items": unified_items,
                "total_items_extracted": len(unified_items),
                "num_chunks_evaluated": len(chunks),
            }
            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched

    @staticmethod
    def stitch_summarization(
        chunk_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assemble document summaries hierarchically across chronological chunks.
        
        Rule:
        - Single chunk (N=1): Return clean summary without synthetic part headers.
        - Multi-chunk (N>1): Formats segments under '--- [Part i/N] ---' headers.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in chunk_results:
            grouped[item["record_id"]].append(item)

        stitched: List[Dict[str, Any]] = []

        for rec_id, chunks in grouped.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            total_chunks = len(chunks)
            base_meta = chunks[0].get("metadata", {})
            rec_idx = chunks[0]["record_index"]

            summary_parts: List[str] = []
            aggregated_points: List[str] = []

            for ch in chunks:
                parsed_items = ch.get("parsed_items", [])
                chunk_summary_text = ""

                for pi in parsed_items:
                    if isinstance(pi, SummaryItem):
                        chunk_summary_text = pi.summary.strip()
                        aggregated_points.extend(pi.key_points)
                    elif isinstance(pi, dict):
                        chunk_summary_text = str(pi.get("summary", "")).strip()
                        if "key_points" in pi and isinstance(pi["key_points"], list):
                            aggregated_points.extend([str(k) for k in pi["key_points"]])
                    elif isinstance(pi, str) and pi.strip():
                        chunk_summary_text = pi.strip()

                if not chunk_summary_text:
                    # Fallback to raw output if parsed item is empty
                    chunk_summary_text = ch.get("raw_output", "").strip()

                if chunk_summary_text:
                    summary_parts.append(chunk_summary_text)

            # Format final summary
            if total_chunks == 1 or len(summary_parts) <= 1:
                final_summary = summary_parts[0] if summary_parts else ""
            else:
                formatted_segments = []
                for idx, part in enumerate(summary_parts, 1):
                    formatted_segments.append(f"--- [Part {idx}/{len(summary_parts)}] ---\n{part}")
                final_summary = "\n\n".join(formatted_segments)

            rec_output = {
                "id": rec_id,
                "index": rec_idx,
                "summary": final_summary,
                "key_points": aggregated_points,
                "num_chunks_evaluated": total_chunks,
            }
            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched
