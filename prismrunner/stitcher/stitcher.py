"""Deterministic stitcher for aggregating chunk outputs back into record-level results."""

from __future__ import annotations
import collections
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DeterministicStitcher:
    """Stitches multi-chunk inference outputs into unified record results."""

    @staticmethod
    def stitch_classification(
        chunk_results: List[Dict[str, Any]],
        category_field: Optional[str] = None,
        confidence_field: Optional[str] = None,
        reasoning_field: Optional[str] = None,
        tie_breaker: str = "earliest",
    ) -> List[Dict[str, Any]]:
        """Aggregate classification predictions via majority voting with customizable fields and tie-breaking."""
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
            unrecognized_items: List[Dict[str, Any]] = []

            for ch in chunks:
                parsed_items = ch.get("parsed_items", [])
                for pi in parsed_items:
                    pi_dict: Optional[Dict[str, Any]] = None
                    if isinstance(pi, BaseModel):
                        pi_dict = pi.model_dump()
                    elif isinstance(pi, dict):
                        pi_dict = pi
                    elif isinstance(pi, str) and pi.strip():
                        pi_dict = {"category": pi.strip(), "confidence": 1.0}

                    cat = None
                    conf = 1.0
                    reason = None

                    if pi_dict is not None:
                        # 1. Resolve Category Field
                        if category_field and category_field in pi_dict:
                            val = pi_dict[category_field]
                            if val is not None:
                                cat = str(val).strip()
                        else:
                            for cand in ["category", "label", "class", "sentiment", "prediction", "diagnosis", "status"]:
                                if cand in pi_dict and pi_dict[cand] is not None:
                                    cat = str(pi_dict[cand]).strip()
                                    break
                            if not cat:
                                for v in pi_dict.values():
                                    if isinstance(v, str) and v.strip():
                                        cat = v.strip()
                                        break

                        # 2. Resolve Confidence Field
                        if confidence_field and confidence_field in pi_dict:
                            try:
                                conf = float(pi_dict[confidence_field])
                            except (ValueError, TypeError):
                                conf = 1.0
                        else:
                            for cand in ["confidence", "score", "prob", "probability"]:
                                if cand in pi_dict and pi_dict[cand] is not None:
                                    try:
                                        conf = float(pi_dict[cand])
                                        break
                                    except (ValueError, TypeError):
                                        pass

                        # 3. Resolve Reasoning Field
                        if reasoning_field and reasoning_field in pi_dict:
                            reason = str(pi_dict[reasoning_field])
                        else:
                            for cand in ["reasoning", "reason", "explanation", "rationale"]:
                                if cand in pi_dict and pi_dict[cand]:
                                    reason = str(pi_dict[cand])
                                    break

                        if cat:
                            votes[cat] += 1
                            confidences[cat].append(conf)
                            if cat not in earliest_chunk_for_cat:
                                earliest_chunk_for_cat[cat] = ch["chunk_index"]
                            if reason:
                                all_reasons.append(reason)
                        else:
                            unrecognized_items.append(pi_dict)

            warning_msg = None
            if not votes:
                winning_cat = "Unknown"
                avg_conf = 0.0
                if unrecognized_items:
                    avail_keys = list(unrecognized_items[0].keys())
                    warning_msg = (
                        f"Could not identify category field in parsed items. "
                        f"Available fields: {avail_keys}. "
                        f"Please specify category_field='<field_name>'."
                    )
                    logger.warning(f"Record '{rec_id}': {warning_msg}")
            else:
                max_count = max(votes.values())
                candidates = [c for c, count in votes.items() if count == max_count]

                # Tie-breaking
                if tie_breaker == "highest_confidence":
                    winning_cat = max(candidates, key=lambda c: max(confidences.get(c, [0.0])))
                else:
                    # Default: earliest chunk
                    winning_cat = min(candidates, key=lambda c: earliest_chunk_for_cat.get(c, 0))

                cat_confs = confidences.get(winning_cat, [1.0])
                avg_conf = round(sum(cat_confs) / len(cat_confs), 3)

            rec_output = {
                "id": rec_id,
                "record_id": rec_id,
                "index": rec_idx,
                "record_index": rec_idx,
                "category": winning_cat,
                "confidence": avg_conf,
                "num_chunks_evaluated": total_chunks,
                "vote_distribution": dict(votes),
            }
            if all_reasons:
                rec_output["reasoning"] = "; ".join(all_reasons[:3])
            if warning_msg:
                rec_output["warning"] = warning_msg

            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched

    @staticmethod
    def stitch_extraction(
        chunk_results: List[Dict[str, Any]],
        dedup: bool = True,
        dedup_key: Optional[Union[str, Callable[[Dict[str, Any]], Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate extracted entities across chunks with customizable deduplication."""
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
                        if not dedup:
                            unified_items.append(entity_dict)
                            continue

                        # Determine deduplication key:
                        key_val = None
                        if dedup_key is not None:
                            if callable(dedup_key):
                                try:
                                    key_val = str(dedup_key(entity_dict)).strip().lower()
                                except Exception as e:
                                    logger.warning(f"Custom dedup_key function failed for record {rec_id}: {e}")
                            elif isinstance(dedup_key, str):
                                if dedup_key in entity_dict and str(entity_dict[dedup_key]).strip():
                                    key_val = str(entity_dict[dedup_key]).strip().lower()
                                else:
                                    logger.warning(
                                        f"Specified dedup_key '{dedup_key}' not found in entity fields "
                                        f"{list(entity_dict.keys())}. Falling back to automatic resolution."
                                    )

                        # Automatic key resolution fallback
                        if not key_val:
                            # 1. Direct standard keys
                            for k in ["entity", "name", "event_name", "item", "title", "label", "term"]:
                                if k in entity_dict and str(entity_dict[k]).strip():
                                    key_val = str(entity_dict[k]).strip().lower()
                                    break

                        # 2. Keys ending with _name, _title, or _term
                        if not key_val:
                            for k, v in entity_dict.items():
                                if (k.endswith("_name") or k.endswith("_title") or k.endswith("_term")) and str(v).strip():
                                    key_val = str(v).strip().lower()
                                    break

                        # 3. First non-empty string value
                        if not key_val:
                            for v in entity_dict.values():
                                if isinstance(v, str) and v.strip():
                                    key_val = v.strip().lower()
                                    break

                        # 4. Fallback to serialized representation
                        if not key_val:
                            try:
                                key_val = json.dumps(entity_dict, sort_keys=True, default=str).lower()
                            except Exception:
                                key_val = str(sorted(entity_dict.items())).lower()

                        if key_val not in seen_entities:
                            seen_entities.add(key_val)
                            unified_items.append(entity_dict)

            rec_output = {
                "id": rec_id,
                "record_id": rec_id,
                "index": rec_idx,
                "record_index": rec_idx,
                "extracted_items": unified_items,
                "parsed_items": unified_items,
                "total_items_extracted": len(unified_items),
                "num_chunks_evaluated": len(chunks),
            }
            if len(unified_items) == 0:
                parse_errs = [ch.get("parse_error") for ch in chunks if ch.get("parse_error")]
                if parse_errs:
                    rec_output["warning"] = f"Extraction yielded 0 items. Underlying parse issues: {'; '.join(parse_errs[:2])}"
            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched

    @staticmethod
    def stitch_summarization(
        chunk_results: List[Dict[str, Any]],
        summary_field: Optional[str] = None,
        points_field: Optional[str] = None,
        part_header_template: Optional[str] = "--- [Part {index}/{total}] ---",
    ) -> List[Dict[str, Any]]:
        """Assemble document summaries hierarchically across chronological chunks with configurable fields."""
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
                    pi_dict: Optional[Dict[str, Any]] = None
                    if isinstance(pi, BaseModel):
                        pi_dict = pi.model_dump()
                    elif isinstance(pi, dict):
                        pi_dict = pi
                    elif isinstance(pi, str) and pi.strip():
                        chunk_summary_text = pi.strip()

                    if pi_dict is not None:
                        # Extract summary text
                        if summary_field and summary_field in pi_dict:
                            chunk_summary_text = str(pi_dict[summary_field]).strip()
                        else:
                            for cand in ["summary", "overview", "abstract", "synthesis", "narrative", "text", "description"]:
                                if cand in pi_dict and pi_dict[cand]:
                                    chunk_summary_text = str(pi_dict[cand]).strip()
                                    break
                            if not chunk_summary_text:
                                for v in pi_dict.values():
                                    if isinstance(v, str) and len(v.strip().split()) > 3:
                                        chunk_summary_text = v.strip()
                                        break

                        # Extract key points
                        if points_field and points_field in pi_dict:
                            val = pi_dict[points_field]
                            if isinstance(val, list):
                                aggregated_points.extend([str(k) for k in val])
                            elif val:
                                aggregated_points.append(str(val))
                        else:
                            for cand in ["key_points", "bullet_points", "takeaways", "points", "findings", "highlights", "action_items"]:
                                if cand in pi_dict and isinstance(pi_dict[cand], list):
                                    aggregated_points.extend([str(k) for k in pi_dict[cand]])
                                    break

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
                    if part_header_template:
                        header = part_header_template.format(index=idx, total=len(summary_parts))
                        formatted_segments.append(f"{header}\n{part}")
                    else:
                        formatted_segments.append(part)
                final_summary = "\n\n".join(formatted_segments)

            # Deduplicate aggregated points while preserving chronological insertion order
            seen_points = set()
            deduped_points: List[str] = []
            for pt in aggregated_points:
                pt_norm = pt.strip()
                if pt_norm and pt_norm not in seen_points:
                    seen_points.add(pt_norm)
                    deduped_points.append(pt_norm)

            rec_output = {
                "id": rec_id,
                "record_id": rec_id,
                "index": rec_idx,
                "record_index": rec_idx,
                "summary": final_summary,
                "key_points": deduped_points,
                "num_chunks_evaluated": total_chunks,
            }
            if not final_summary and not deduped_points:
                parse_errs = [ch.get("parse_error") for ch in chunks if ch.get("parse_error")]
                if parse_errs:
                    rec_output["warning"] = f"Summarization yielded 0 content. Underlying parse issues: {'; '.join(parse_errs[:2])}"
            rec_output.update(base_meta)
            stitched.append(rec_output)

        stitched.sort(key=lambda r: r["index"])
        return stitched
