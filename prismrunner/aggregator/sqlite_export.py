"""SQLite relational database exporter for normalized schema extractions."""

from __future__ import annotations
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Set, Union

logger = logging.getLogger(__name__)


def _infer_sql_type(val: Any) -> str:
    """Infer SQLite column type from a Python value."""
    if isinstance(val, bool):
        return "INTEGER"
    elif isinstance(val, int):
        return "INTEGER"
    elif isinstance(val, float):
        return "REAL"
    elif isinstance(val, bytes):
        return "BLOB"
    return "TEXT"


class SQLiteExporter:
    """Exports normalized tables into an indexed, relational SQLite database."""

    @staticmethod
    def export(
        tables: Dict[str, List[Dict[str, Any]]],
        db_path: Union[str, Path],
    ) -> Path:
        """Create or update SQLite database with normalized relational tables."""
        target_path = Path(db_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(target_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        try:
            with conn:
                for table_name, rows in tables.items():
                    if not rows:
                        continue

                    # Collect all columns and infer types
                    col_types: Dict[str, str] = {}
                    for row in rows:
                        for col_name, val in row.items():
                            if col_name not in col_types:
                                col_types[col_name] = _infer_sql_type(val)

                    # Ensure primary key and foreign key conventions
                    cols_def = ["_row_id INTEGER PRIMARY KEY AUTOINCREMENT"]
                    for col_name, sql_type in col_types.items():
                        cols_def.append(f'"{col_name}" {sql_type}')

                    create_stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(cols_def)});'
                    cursor.execute(create_stmt)

                    # Create index on source_id for relational joins
                    if "source_id" in col_types:
                        idx_name = f"idx_{table_name}_source_id"
                        cursor.execute(
                            f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("source_id");'
                        )

                    # Insert rows
                    columns = list(col_types.keys())
                    placeholders = ", ".join(["?"] * len(columns))
                    col_names_quoted = ", ".join([f'"{c}"' for c in columns])
                    insert_stmt = f'INSERT INTO "{table_name}" ({col_names_quoted}) VALUES ({placeholders});'

                    records_to_insert = []
                    for row in rows:
                        row_tuple = tuple(row.get(col) for col in columns)
                        records_to_insert.append(row_tuple)

                    cursor.executemany(insert_stmt, records_to_insert)

            logger.info(f"Exported {len(tables)} relational tables to SQLite: {target_path}")
        finally:
            conn.close()

        return target_path
