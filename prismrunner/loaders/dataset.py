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
    candidates = ["text", "content", "body", "document", "message", "description", "input", "item", "note", "clinical_note"]
    col_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    if len(columns) == 1:
        return columns[0]
    return None


def _detect_id_column(columns: List[str]) -> Optional[str]:
    """Auto-detect candidate column for record primary key."""
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
        file_path_or_data: Union[str, Path, pd.DataFrame, List[Dict[str, Any]], List[str], List[Record]],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Record]:
        """Load Parquet, CSV, JSON, TXT, DataFrame, or native Python lists into typed Record instances."""
        # 1. Direct pass-through if already a list of Record objects
        if isinstance(file_path_or_data, list) and all(isinstance(r, Record) for r in file_path_or_data):
            return file_path_or_data[:limit] if limit else file_path_or_data

        # 2. In-memory list of strings
        if isinstance(file_path_or_data, list) and (not file_path_or_data or all(isinstance(s, str) for s in file_path_or_data)):
            items = file_path_or_data[:limit] if limit else file_path_or_data
            return [
                Record(id=f"rec_{i + 1:06d}", index=i, text=s.strip(), metadata={})
                for i, s in enumerate(items)
            ]

        # 3. Handle DataFrame or convert list of dicts to DataFrame
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
            # Direct string text (not an existing path)
            str_val = str(file_path_or_data)
            if not os.path.exists(str_val) and ("\n" in str_val or len(str_val) > 200 or not Path(str_val).suffix):
                return [Record(id="rec_000001", index=0, text=str_val.strip(), metadata={})]

            path = Path(file_path_or_data)
            if not path.exists():
                return [Record(id="rec_000001", index=0, text=str_val.strip(), metadata={})]

            source_name = path.name
            suffix = path.suffix.lower()

            if suffix == ".parquet":
                table = pq.read_table(path)
                df = table.to_pandas()
            elif suffix in (".csv", ".tsv"):
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
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
                    f"Supported formats: .parquet, .csv, .tsv, .json, .jsonl, .txt, .md"
                )

        if df is None:
            raise TypeError(
                f"Unsupported input type '{type(file_path_or_data).__name__}'. "
                f"Expected str, Path, pd.DataFrame, List[dict], List[str], or List[Record]."
            )

        if limit is not None and limit > 0:
            df = df.iloc[:limit]

        cols = list(df.columns)

        # 4. Resolve text column with descriptive errors
        if text_column:
            if text_column not in df.columns:
                raise ValueError(
                    f"Specified text column '{text_column}' was not found in {source_name}. "
                    f"Available columns are: {cols}."
                )
            chosen_text_col = text_column
        else:
            chosen_text_col = _detect_text_column(cols)
            if not chosen_text_col:
                raise ValueError(
                    f"Could not auto-detect text column in {source_name}. Available columns: {cols}. "
                    f"Please explicitly specify text_column='<column_name>' (e.g., text_column='{cols[0]}')."
                )

        # 5. Resolve id column with descriptive errors
        if id_column:
            if id_column not in df.columns:
                raise ValueError(
                    f"Specified id column '{id_column}' was not found in {source_name}. "
                    f"Available columns are: {cols}."
                )
            chosen_id_col = id_column
        else:
            chosen_id_col = _detect_id_column(cols)

        records: List[Record] = []
        dict_rows = df.to_dict(orient="records")
        for idx, row_dict in enumerate(dict_rows):
            rec_id = str(row_dict[chosen_id_col]) if chosen_id_col and pd.notna(row_dict.get(chosen_id_col)) else f"rec_{idx + 1:06d}"
            text_val = str(row_dict.get(chosen_text_col, ""))

            # Filter metadata dynamically
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

        logger.info(f"Loaded {len(records)} records from {source_name} (text_col='{chosen_text_col}', id_col='{chosen_id_col}')")
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
