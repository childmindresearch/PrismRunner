"""Universal dataset loader supporting Parquet, CSV, JSON, and text with deterministic ID tracking."""

from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """Represents a single input record with deterministic ID and metadata."""

    id: str
    index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _detect_text_column(columns: List[str]) -> Optional[str]:
    """Auto-detect candidate column for text content."""
    candidates = ["text", "content", "body", "document", "message", "description", "input", "item"]
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return columns[0] if columns else None


def _detect_id_column(columns: List[str]) -> Optional[str]:
    """Auto-detect candidate column for record primary key."""
    candidates = ["id", "uuid", "key", "record_id", "item_id", "row_id", "index"]
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


class DatasetLoader:
    """Loads and normalizes tabular and unstructured datasets into Record instances."""

    @staticmethod
    def load(
        file_path_or_text: Union[str, Path],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Record]:
        """Load Parquet, CSV, JSON, TXT, or raw string into typed Record instances."""
        # Check if raw text string is passed directly (not an existing path)
        if isinstance(file_path_or_text, str) and not os.path.exists(file_path_or_text) and "\n" in file_path_or_text or (isinstance(file_path_or_text, str) and len(file_path_or_text) > 200 and not os.path.exists(file_path_or_text)):
            return [Record(id="rec_000001", index=0, text=file_path_or_text.strip(), metadata={})]

        path = Path(file_path_or_text)
        if not path.exists():
            # If short string passed and not file, treat as direct text input
            if isinstance(file_path_or_text, str):
                return [Record(id="rec_000001", index=0, text=file_path_or_text.strip(), metadata={})]
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".parquet":
            table = pq.read_table(path)
            df = table.to_pandas()
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        elif suffix in (".json", ".jsonl"):
            try:
                # First attempt reading standard JSON
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Check if dictionary has a records/data key
                    for key in ("records", "data", "rows", "items"):
                        if key in data and isinstance(data[key], list):
                            df = pd.DataFrame(data[key])
                            break
                    else:
                        df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame([{"text": str(data)}])
            except json.JSONDecodeError:
                # Attempt reading JSON Lines
                df = pd.read_json(path, lines=True)
        elif suffix in (".txt", ".md", ".markdown", ".text"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Record(id=f"{path.stem}_0", index=0, text=content.strip(), metadata={"filename": path.name})]
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .parquet, .csv, .json, .txt, .md")

        if limit is not None and limit > 0:
            df = df.iloc[:limit]

        cols = list(df.columns)
        chosen_text_col = text_column or _detect_text_column(cols)
        if not chosen_text_col or chosen_text_col not in df.columns:
            raise ValueError(
                f"Could not determine text column. Available columns: {cols}. "
                f"Please specify --column / --text-column."
            )

        chosen_id_col = id_column or _detect_id_column(cols)
        records: List[Record] = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            rec_id = str(row_dict[chosen_id_col]) if chosen_id_col and pd.notna(row_dict.get(chosen_id_col)) else f"rec_{idx + 1:06d}"
            text_val = str(row_dict.get(chosen_text_col, ""))
            # Remove text from metadata to avoid redundancy
            meta = {k: v for k, v in row_dict.items() if k != chosen_text_col and pd.notna(v)}

            records.append(Record(id=rec_id, index=int(idx), text=text_val, metadata=meta))

        logger.info(f"Loaded {len(records)} records from {path.name} (text_col='{chosen_text_col}', id_col='{chosen_id_col}')")
        return records


def export_results(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path],
    output_format: Optional[str] = None,
) -> Path:
    """Export pipeline results into JSON, CSV, or Parquet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or out.suffix.lstrip(".")).lower()
    if not fmt:
        fmt = "json"

    if fmt == "json":
        if not str(out).endswith(".json"):
            out = out.with_suffix(".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    elif fmt == "csv":
        if not str(out).endswith(".csv"):
            out = out.with_suffix(".csv")
        df = pd.DataFrame(results)
        df.to_csv(out, index=False)
    elif fmt == "parquet":
        if not str(out).endswith(".parquet"):
            out = out.with_suffix(".parquet")
        df = pd.DataFrame(results)
        df.to_parquet(out, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}. Supported: json, csv, parquet")

    logger.info(f"Successfully saved {len(results)} records to {out}")
    return out
