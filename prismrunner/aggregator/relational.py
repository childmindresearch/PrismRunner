"""Relational data normalization engine for converting nested Pydantic extractions into linked tables."""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel

METADATA_KEYS: Set[str] = {"source_id", "source_file", "parent_item_index", "item_index", "schema_name"}


def has_extracted_content(val: Any) -> bool:
    """Recursively checks if a structure contains at least one non-null, non-empty fact.
    
    Ignores structural metadata keys.
    """
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (int, float, bool)):
        return True
    if isinstance(val, (list, tuple, set)):
        return any(has_extracted_content(item) for item in val)
    if isinstance(val, dict):
        content_items = {k: v for k, v in val.items() if k not in METADATA_KEYS}
        if not content_items:
            return False
        return any(has_extracted_content(v) for v in content_items.values())
    if isinstance(val, BaseModel):
        return has_extracted_content(val.model_dump())
    return True


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


class RelationalNormalizer:
    """Normalizes nested extractions into parent-child relational tables with foreign keys."""

    @staticmethod
    def normalize_record(
        data: Any,
        root_schema_name: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        id_column_name: str = "source_id",
        child_separator: str = "_",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Decompose a single extracted payload (dict, BaseModel, or list) into normalized relational tables."""
        tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        base_meta = dict(metadata or {})

        # Convert Pydantic to dict
        if isinstance(data, BaseModel):
            payload = data.model_dump()
        elif isinstance(data, list):
            payload = [d.model_dump() if isinstance(d, BaseModel) else d for d in data]
        elif isinstance(data, dict):
            payload = data
        else:
            payload = {"value": data}

        def process_object(
            obj: Any,
            table_name: str,
            foreign_keys: Optional[Dict[str, Any]] = None,
        ):
            if foreign_keys is None:
                foreign_keys = {}

            row_data = dict(foreign_keys)
            row_data.setdefault(id_column_name, source_id)
            for mk, mv in base_meta.items():
                row_data.setdefault(mk, mv)

            if not isinstance(obj, dict):
                if has_extracted_content(obj):
                    row_data["value"] = obj
                    tables[table_name].append(row_data)
                return

            scalar_fields: Dict[str, Any] = {}
            list_fields: Dict[str, List[Any]] = {}

            for k, v in obj.items():
                if isinstance(v, list):
                    list_fields[k] = v
                elif isinstance(v, dict):
                    # Flatten nested dictionary scalars into current row
                    flat_sub = flatten_dict(v, parent_key=k)
                    scalar_fields.update(flat_sub)
                else:
                    scalar_fields[k] = v

            row_data.update(scalar_fields)
            if has_extracted_content(scalar_fields):
                tables[table_name].append(row_data)

            # Process 1-to-many child list items
            parent_idx = row_data.get("item_index")
            for list_key, items in list_fields.items():
                child_table = f"{table_name}{child_separator}{list_key}"
                for idx, child_item in enumerate(items):
                    child_fk = {id_column_name: source_id}
                    if parent_idx is not None:
                        child_fk["parent_item_index"] = parent_idx
                    child_fk["item_index"] = idx

                    if isinstance(child_item, dict):
                        process_object(child_item, child_table, foreign_keys=child_fk)
                    else:
                        if has_extracted_content(child_item):
                            child_row = dict(child_fk)
                            child_row["value"] = child_item
                            tables[child_table].append(child_row)

        if isinstance(payload, list):
            for idx, item in enumerate(payload):
                fk = {id_column_name: source_id, "item_index": idx}
                process_object(item, root_schema_name, foreign_keys=fk)
        else:
            process_object(payload, root_schema_name, foreign_keys={id_column_name: source_id})

        return dict(tables)

    @classmethod
    def normalize_dataset(
        cls,
        extractions: List[Dict[str, Any]],
        id_column_name: str = "source_id",
        child_separator: str = "_",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate multiple extracted records across all schemas into consolidated relational tables."""
        all_tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for record in extractions:
            src_id = str(record.get(id_column_name, record.get("source_id", record.get("record_id", record.get("id", "unknown")))))
            schema_name = record.get("schema_name", "Extraction")
            data = record.get("data", record.get("result", record.get("extracted_items", {})))
            meta = {k: v for k, v in record.items() if k not in ("data", "result", "extracted_items", "raw_output")}

            rec_tables = cls.normalize_record(
                data=data,
                root_schema_name=schema_name,
                source_id=src_id,
                metadata=meta,
                id_column_name=id_column_name,
                child_separator=child_separator,
            )

            for t_name, rows in rec_tables.items():
                all_tables[t_name].extend(rows)

        # Drop tables that have 0 rows
        pruned = {t: rows for t, rows in all_tables.items() if rows}
        return pruned
