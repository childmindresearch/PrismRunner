"""Schema catalog registry for discovering and managing domain Pydantic models."""

from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class SchemaMeta:
    """Metadata describing a registered schema."""

    name: str
    schema_cls: Type[BaseModel]
    description: str
    module_path: Optional[str] = None

    @classmethod
    def from_class(cls, model_cls: Type[BaseModel], description_override: Optional[str] = None) -> "SchemaMeta":
        doc = description_override or model_cls.__doc__ or ""
        clean_doc = " ".join(doc.strip().splitlines()) if doc else f"Extracts {model_cls.__name__} structured fields."
        return cls(
            name=model_cls.__name__,
            schema_cls=model_cls,
            description=clean_doc,
            module_path=inspect.getmodule(model_cls).__name__ if inspect.getmodule(model_cls) else None,
        )


class SchemaCatalog:
    """Registry managing a collection of Pydantic models for multi-schema extraction and triage."""

    def __init__(self):
        self._schemas: Dict[str, SchemaMeta] = {}

    def register(
        self,
        schema_cls: Type[BaseModel],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a Pydantic model with the catalog."""
        if not isinstance(schema_cls, type) or not issubclass(schema_cls, BaseModel):
            raise TypeError(f"{schema_cls} is not a subclass of pydantic.BaseModel")

        schema_name = name or schema_cls.__name__
        meta = SchemaMeta.from_class(schema_cls, description_override=description)
        meta.name = schema_name
        self._schemas[schema_name] = meta
        logger.debug(f"Registered schema: {schema_name}")

    def get(self, name: str) -> Optional[SchemaMeta]:
        """Retrieve schema metadata by name."""
        return self._schemas.get(name)

    def list_schemas(self) -> List[SchemaMeta]:
        """List all registered schemas."""
        return list(self._schemas.values())

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, name: str) -> bool:
        return name in self._schemas

    def build_triage_summary(self) -> str:
        """Generate human-readable catalog summary for triage pre-screening prompts."""
        lines = []
        for meta in self._schemas.values():
            lines.append(f'- "{meta.name}": {meta.description}')
        return "\n".join(lines)

    @classmethod
    def from_directory(cls, dir_path: Union[str, Path]) -> "SchemaCatalog":
        """Auto-discover all Pydantic BaseModel subclasses in a directory of Python files."""
        catalog = cls()
        directory = Path(dir_path).resolve()
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Schemas directory not found: {directory}")

        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("__"):
                continue

            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.warning(f"Could not load schemas from {py_file}: {e}")
                continue

            for attr_name, attr_val in inspect.getmembers(module, inspect.isclass):
                if issubclass(attr_val, BaseModel) and attr_val is not BaseModel:
                    # Ignore internal models from dependencies
                    if attr_val.__module__ == module_name:
                        catalog.register(attr_val)

        logger.info(f"Loaded {len(catalog)} schemas from {directory}")
        return catalog
