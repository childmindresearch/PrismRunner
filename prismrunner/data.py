"""Dataset loading, token-aware sliding-window chunking, and deterministic multi-chunk stitching."""

from __future__ import annotations
import collections
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Dataset Loading & Result Export
# ==============================================================================

@dataclass
class Record:
    """Represents a single input record with deterministic ID and metadata."""

    id: str
    index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _detect_text_column(columns: List[str]) -> Optional[str]:
    candidates = ["text", "content", "body", "document", "message", "description", "input", "item", "note", "clinical_note"]
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return columns[0] if len(columns) == 1 else None


def _detect_id_column(columns: List[str]) -> Optional[str]:
    candidates = ["id", "uuid", "key", "record_id", "item_id", "row_id", "patient_id", "subject_id", "doc_id", "index"]
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


class DatasetLoader:
    """Loads and normalizes tabular and unstructured datasets into Record instances."""

    @staticmethod
    def load(
        file_path_or_data: Union[str, Path, pd.DataFrame, List[Dict[str, Any]], List[str], List[Record], Dict[str, Any]],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Record]:
        """Load Parquet, CSV, JSON, TXT, DataFrame, or native Python lists into typed Record instances."""
        if isinstance(file_path_or_data, list) and all(isinstance(r, Record) for r in file_path_or_data):
            return file_path_or_data[:limit] if limit else file_path_or_data

        if isinstance(file_path_or_data, list) and (not file_path_or_data or all(isinstance(s, str) for s in file_path_or_data)):
            items = file_path_or_data[:limit] if limit else file_path_or_data
            return [Record(id=f"rec_{i + 1:06d}", index=i, text=s.strip(), metadata={}) for i, s in enumerate(items)]

        df: Optional[pd.DataFrame] = None
        source_name = "in_memory"

        if isinstance(file_path_or_data, pd.DataFrame):
            df = file_path_or_data.copy()
            source_name = "DataFrame"
        elif isinstance(file_path_or_data, dict):
            df = pd.DataFrame([file_path_or_data])
            source_name = "single_dict"
        elif isinstance(file_path_or_data, list) and all(isinstance(d, dict) for d in file_path_or_data):
            df = pd.DataFrame(file_path_or_data)
            source_name = "list_of_dicts"
        elif isinstance(file_path_or_data, (str, Path)):
            str_val = str(file_path_or_data)
            if not os.path.exists(str_val) and ("\n" in str_val or len(str_val) > 200 or not Path(str_val).suffix):
                return [Record(id="rec_000001", index=0, text=str_val.strip(), metadata={})]

            path = Path(file_path_or_data)
            if not path.exists():
                return [Record(id="rec_000001", index=0, text=str_val.strip(), metadata={})]

            source_name = path.name
            suffix = path.suffix.lower()

            if suffix == ".parquet":
                df = pq.read_table(path).to_pandas()
            elif suffix in (".csv", ".tsv"):
                df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
            elif suffix in (".json", ".jsonl"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        for key in ("records", "data", "rows", "items"):
                            if key in data and isinstance(data[key], list):
                                df = pd.DataFrame(data[key])
                                break
                        else:
                            df = pd.DataFrame([data])
                    else:
                        df = pd.DataFrame([{"text": str(data)}])
                except json.JSONDecodeError:
                    df = pd.read_json(path, lines=True)
            elif suffix in (".txt", ".md", ".markdown", ".text"):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return [Record(id=f"{path.stem}_0", index=0, text=content.strip(), metadata={"filename": path.name})]
            else:
                raise ValueError(
                    f"Unsupported file format '{suffix}' for {path}. "
                    "Supported formats: .parquet, .csv, .tsv, .json, .jsonl, .txt, .md"
                )

        if df is None:
            raise TypeError(
                f"Unsupported input type '{type(file_path_or_data).__name__}'. "
                "Expected str, Path, pd.DataFrame, List[dict], List[str], or List[Record]."
            )

        if limit is not None and limit > 0:
            df = df.iloc[:limit]

        cols = list(df.columns)

        if text_column:
            if text_column not in df.columns:
                raise ValueError(f"Specified text column '{text_column}' was not found in {source_name}. Available columns are: {cols}.")
            chosen_text_col = text_column
        else:
            chosen_text_col = _detect_text_column(cols)
            if not chosen_text_col:
                raise ValueError(
                    f"Could not auto-detect text column in {source_name}. Available columns: {cols}. "
                    f"Please explicitly specify text_column='<column_name>'."
                )

        if id_column:
            if id_column not in df.columns:
                raise ValueError(f"Specified id column '{id_column}' was not found in {source_name}. Available columns are: {cols}.")
            chosen_id_col = id_column
        else:
            chosen_id_col = _detect_id_column(cols)

        records: List[Record] = []
        for idx, row_dict in enumerate(df.to_dict(orient="records")):
            rec_id = str(row_dict[chosen_id_col]) if chosen_id_col and pd.notna(row_dict.get(chosen_id_col)) else f"rec_{idx + 1:06d}"
            text_val = str(row_dict.get(chosen_text_col, ""))

            meta = {}
            for k, v in row_dict.items():
                if k == chosen_text_col or pd.isna(v):
                    continue
                if metadata_columns is not None and k not in metadata_columns:
                    continue
                if exclude_columns is not None and k in exclude_columns:
                    continue
                meta[k] = v

            records.append(Record(id=rec_id, index=idx, text=text_val, metadata=meta))

        return records


def export_results(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    output_format: Optional[str] = None,
) -> Path:
    """Export pipeline results into JSON, CSV, or Parquet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or out.suffix.lstrip(".")).lower() or "json"

    if fmt == "json":
        if not str(out).endswith(".json"):
            out = out.with_suffix(".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    elif fmt == "csv":
        if not str(out).endswith(".csv"):
            out = out.with_suffix(".csv")
        pd.DataFrame(results).to_csv(out, index=False)
    elif fmt == "parquet":
        if not str(out).endswith(".parquet"):
            out = out.with_suffix(".parquet")
        pd.DataFrame(results).to_parquet(out, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}. Supported: json, csv, parquet")

    return out


# ==============================================================================
# 2. Token-Aware Chunker
# ==============================================================================

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")


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
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text) // 4)

    def _split_sentences(self, text: str) -> List[str]:
        sentences = [s.strip() for s in SENTENCE_PATTERN.split(text) if s.strip()]
        return sentences if sentences else [text]

    def _chunk_by_tokens(self, text: str) -> List[str]:
        if self.tokenizer is None:
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
        sentences = self._split_sentences(text)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            s_tokens = self.count_tokens(sentence)
            if s_tokens > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences, current_tokens = [], 0
                chunks.extend(self._chunk_by_tokens(sentence))
                continue

            if current_tokens + s_tokens <= self.chunk_size:
                current_sentences.append(sentence)
                current_tokens += s_tokens
            else:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

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
        text = record.text.strip()
        total_toks = self.count_tokens(text)

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

        raw_chunks = self._chunk_by_sentences(text) if self.strategy == "sentence" else self._chunk_by_tokens(text)
        if not raw_chunks:
            raw_chunks = [text]

        total_chunks = len(raw_chunks)
        return [
            ChunkRecord(
                record_id=record.id,
                record_index=record.index,
                chunk_index=idx,
                total_chunks=total_chunks,
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
                metadata=record.metadata,
            )
            for idx, chunk_text in enumerate(raw_chunks)
        ]

    def chunk_dataset(self, records: List[Record]) -> List[ChunkRecord]:
        all_chunks: List[ChunkRecord] = []
        for rec in records:
            all_chunks.extend(self.chunk_record(rec))
        return all_chunks


# ==============================================================================
# 3. Deterministic Multi-Chunk Stitcher
# ==============================================================================

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
        grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in chunk_results:
            grouped[item["record_id"]].append(item)

        stitched: List[Dict[str, Any]] = []
        for rec_id, chunks in grouped.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            total_chunks = len(chunks)
            base_meta = chunks[0].get("metadata", {})
            rec_idx = chunks[0]["record_index"]

            votes: collections.Counter = collections.Counter()
            earliest_chunk_for_cat: Dict[str, int] = {}
            confidences: Dict[str, List[float]] = collections.defaultdict(list)
            all_reasons: List[str] = []
            unrecognized_items: List[Dict[str, Any]] = []

            for ch in chunks:
                for pi in ch.get("parsed_items", []):
                    pi_dict: Optional[Dict[str, Any]] = None
                    if isinstance(pi, BaseModel):
                        pi_dict = pi.model_dump()
                    elif isinstance(pi, dict):
                        pi_dict = pi
                    elif isinstance(pi, str) and pi.strip():
                        pi_dict = {"category": pi.strip(), "confidence": 1.0}

                    if pi_dict is not None:
                        cat = None
                        if category_field and category_field in pi_dict and pi_dict[category_field] is not None:
                            cat = str(pi_dict[category_field]).strip()
                        else:
                            for cand in ("category", "label", "class", "sentiment", "prediction", "diagnosis", "status"):
                                if cand in pi_dict and pi_dict[cand] is not None:
                                    cat = str(pi_dict[cand]).strip()
                                    break
                            if not cat:
                                cat = next((v.strip() for v in pi_dict.values() if isinstance(v, str) and v.strip()), None)

                        conf = 1.0
                        if confidence_field and confidence_field in pi_dict:
                            try:
                                conf = float(pi_dict[confidence_field])
                            except (ValueError, TypeError):
                                pass
                        else:
                            for cand in ("confidence", "score", "prob", "probability"):
                                if cand in pi_dict and pi_dict[cand] is not None:
                                    try:
                                        conf = float(pi_dict[cand])
                                        break
                                    except (ValueError, TypeError):
                                        pass

                        reason = None
                        if reasoning_field and reasoning_field in pi_dict and pi_dict[reasoning_field]:
                            reason = str(pi_dict[reasoning_field])
                        else:
                            for cand in ("reasoning", "reason", "explanation", "rationale"):
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
                winning_cat, avg_conf = "Unknown", 0.0
                if unrecognized_items:
                    avail_keys = list(unrecognized_items[0].keys())
                    warning_msg = f"Could not identify category field in parsed items. Available fields: {avail_keys}."
            else:
                max_count = max(votes.values())
                candidates = [c for c, count in votes.items() if count == max_count]
                if tie_breaker == "highest_confidence":
                    winning_cat = max(candidates, key=lambda c: max(confidences.get(c, [0.0])))
                else:
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
                for pi in ch.get("parsed_items", []):
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

                        key_val = None
                        if dedup_key is not None:
                            if callable(dedup_key):
                                try:
                                    key_val = str(dedup_key(entity_dict)).strip().lower()
                                except Exception:
                                    pass
                            elif isinstance(dedup_key, str) and dedup_key in entity_dict and str(entity_dict[dedup_key]).strip():
                                key_val = str(entity_dict[dedup_key]).strip().lower()

                        if not key_val:
                            for k in ("entity", "name", "event_name", "item", "title", "label", "term"):
                                if k in entity_dict and str(entity_dict[k]).strip():
                                    key_val = str(entity_dict[k]).strip().lower()
                                    break

                        if not key_val:
                            for k, v in entity_dict.items():
                                if (k.endswith("_name") or k.endswith("_title") or k.endswith("_term")) and str(v).strip():
                                    key_val = str(v).strip().lower()
                                    break

                        if not key_val:
                            key_val = next((v.strip().lower() for v in entity_dict.values() if isinstance(v, str) and v.strip()), None)

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
                chunk_summary_text = ""
                for pi in ch.get("parsed_items", []):
                    pi_dict: Optional[Dict[str, Any]] = None
                    if isinstance(pi, BaseModel):
                        pi_dict = pi.model_dump()
                    elif isinstance(pi, dict):
                        pi_dict = pi
                    elif isinstance(pi, str) and pi.strip():
                        chunk_summary_text = pi.strip()

                    if pi_dict is not None:
                        if summary_field and summary_field in pi_dict:
                            chunk_summary_text = str(pi_dict[summary_field]).strip()
                        else:
                            for cand in ("summary", "overview", "abstract", "synthesis", "narrative", "text", "description"):
                                if cand in pi_dict and pi_dict[cand]:
                                    chunk_summary_text = str(pi_dict[cand]).strip()
                                    break
                            if not chunk_summary_text:
                                chunk_summary_text = next(
                                    (v.strip() for v in pi_dict.values() if isinstance(v, str) and len(v.strip().split()) > 3),
                                    "",
                                )

                        if points_field and points_field in pi_dict:
                            val = pi_dict[points_field]
                            if isinstance(val, list):
                                aggregated_points.extend([str(k) for k in val])
                            elif val:
                                aggregated_points.append(str(val))
                        else:
                            for cand in ("key_points", "bullet_points", "takeaways", "points", "findings", "highlights", "action_items"):
                                if cand in pi_dict and isinstance(pi_dict[cand], list):
                                    aggregated_points.extend([str(k) for k in pi_dict[cand]])
                                    break

                if not chunk_summary_text:
                    chunk_summary_text = ch.get("raw_output", "").strip()
                if chunk_summary_text:
                    summary_parts.append(chunk_summary_text)

            if total_chunks == 1 or len(summary_parts) <= 1:
                final_summary = summary_parts[0] if summary_parts else ""
            else:
                formatted_segments = []
                for idx, part in enumerate(summary_parts, 1):
                    if part_header_template:
                        formatted_segments.append(f"{part_header_template.format(index=idx, total=len(summary_parts))}\n{part}")
                    else:
                        formatted_segments.append(part)
                final_summary = "\n\n".join(formatted_segments)

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

