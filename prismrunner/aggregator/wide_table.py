"""Consolidated Wide Table generator producing 1 row per document with counts and exploded attributes."""

from __future__ import annotations
from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)


class WideTableBuilder:
    """Consolidates multi-schema relational extractions into a single wide table (1 row / doc)."""

    @staticmethod
    def build(
        tables: Dict[str, List[Dict[str, Any]]],
        include_counts: bool = True,
        list_join_char: str = "; ",
        source_texts: Optional[Dict[str, str]] = None,
        id_column_name: str = "source_id",
    ) -> pd.DataFrame:
        """Construct a wide DataFrame containing exactly 1 row per source document ID."""
        # Collect all unique source IDs across tables
        all_source_ids: Set[str] = set()
        for rows in tables.values():
            for r in rows:
                if id_column_name in r:
                    all_source_ids.add(str(r[id_column_name]))
                elif "source_id" in r:
                    all_source_ids.add(str(r["source_id"]))

        if source_texts:
            all_source_ids.update(source_texts.keys())

        doc_rows: Dict[str, Dict[str, Any]] = {
            sid: {id_column_name: sid} for sid in sorted(all_source_ids)
        }

        # Attach source_text if provided
        if source_texts:
            for sid, text in source_texts.items():
                if sid in doc_rows:
                    doc_rows[sid]["source_text"] = text

        for table_name, rows in tables.items():
            # Group rows by id_column_name (fallback to source_id)
            by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                sid = str(r.get(id_column_name, r.get("source_id", "")))
                if sid:
                    by_source[sid].append(r)

            # Check if this table represents 1-to-1 root attributes or 1-to-many child attributes
            is_child_table = any("item_index" in r for r in rows)

            for sid, s_rows in by_source.items():
                target_doc = doc_rows.setdefault(sid, {id_column_name: sid})

                if not is_child_table and len(s_rows) == 1:
                    # 1-to-1 root table: prefix columns with table_name
                    for k, v in s_rows[0].items():
                        if k in (id_column_name, "source_id", "source_file", "parent_item_index", "item_index"):
                            continue
                        col = f"{table_name}_{k}"
                        target_doc[col] = v
                else:
                    # 1-to-many child table: add count and join distinct values
                    if include_counts:
                        target_doc[f"num_{table_name}"] = len(s_rows)

                    # Aggregate column values
                    child_cols = set()
                    for r in s_rows:
                        child_cols.update(r.keys())

                    for c in child_cols:
                        if c in (id_column_name, "source_id", "source_file", "parent_item_index", "item_index"):
                            continue
                        vals = [str(r[c]) for r in s_rows if r.get(c) is not None and str(r[c]).strip()]
                        if vals:
                            target_doc[f"{table_name}_{c}"] = list_join_char.join(vals)

        # Build DataFrame
        df = pd.DataFrame(list(doc_rows.values()))
        if id_column_name in df.columns:
            # Keep id column as first column
            cols = [id_column_name] + [c for c in df.columns if c != id_column_name]
            df = df[cols]

        logger.info(f"Built consolidated wide table: {len(df)} rows × {len(df.columns)} columns")
        return df

    @classmethod
    def export(
        cls,
        tables: Dict[str, List[Dict[str, Any]]],
        output_path: Union[str, Path],
        source_texts: Optional[Dict[str, str]] = None,
        output_format: Optional[str] = None,
        list_join_char: str = "; ",
        id_column_name: str = "source_id",
    ) -> Path:
        """Build and export wide table to .parquet or .csv."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fmt = (output_format or out.suffix.lstrip(".")).lower()
        if not fmt:
            fmt = "parquet"

        df = cls.build(tables, source_texts=source_texts, list_join_char=list_join_char, id_column_name=id_column_name)

        if fmt == "csv":
            if not str(out).endswith(".csv"):
                out = out.with_suffix(".csv")
            df.to_csv(out, index=False)
        else:
            if not str(out).endswith(".parquet"):
                out = out.with_suffix(".parquet")
            df.to_parquet(out, index=False)

        logger.info(f"Exported wide table to: {out}")
        return out
