"""Granular fault-tolerant resume tracker for multi-schema extraction runs."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

logger = logging.getLogger(__name__)


class CheckpointTracker:
    """Tracks completed (record_id, schema_name, chunk_index) tasks for granular resume."""

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: Set[str] = set()
        self._cache: Dict[str, Any] = {}
        self._load()

    def _make_key(self, record_id: str, schema_name: str, chunk_index: int) -> str:
        return f"{record_id}::{schema_name}::{chunk_index}"

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._cache = data
                    self._completed = set(data.keys())
                logger.info(f"Loaded {len(self._completed)} prior checkpoint entries from {self.path.name}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint file {self.path}: {e}")

    def save(self) -> None:
        """Persist current completed entries to disk."""
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)
        tmp.replace(self.path)

    def is_completed(self, record_id: str, schema_name: str, chunk_index: int = 0) -> bool:
        """Check if an extraction unit has already completed."""
        return self._make_key(record_id, schema_name, chunk_index) in self._completed

    def mark_completed(
        self,
        record_id: str,
        schema_name: str,
        chunk_index: int = 0,
        result_payload: Optional[Any] = None,
    ) -> None:
        """Record completed extraction and update checkpoint state."""
        key = self._make_key(record_id, schema_name, chunk_index)
        self._completed.add(key)
        self._cache[key] = {
            "record_id": record_id,
            "schema_name": schema_name,
            "chunk_index": chunk_index,
            "data": result_payload,
        }

    def get_cached_result(self, record_id: str, schema_name: str, chunk_index: int = 0) -> Optional[Any]:
        """Retrieve stored result from previous run."""
        key = self._make_key(record_id, schema_name, chunk_index)
        entry = self._cache.get(key)
        return entry.get("data") if entry else None

    def __len__(self) -> int:
        return len(self._completed)
