#!/usr/bin/env python3
"""
SemanticPrism Stage 5: Schema-Driven Extraction & Relational Aggregation (Powered by PrismRunner)
=================================================================================================
Standalone implementation of the SchemaMap extraction and relational/wide-table aggregation
workflow (https://github.com/childmindresearch/SchemaMap), powered exclusively by PrismRunner's
local HuggingFace batch inference engine with zero dependency on SemanticPrism or SchemaMap.

Supported Schema Input Modes (SCHEMA_PATH):
  1. Single .py file with multiple Pydantic BaseModel schemas:
     e.g., "outputs/04_synthesis/schemas/master_ontology.py" or
           "outputs/04_synthesis/schemas/comprehensive_ontology.py"
  2. Directory of multiple .py schema files:
     e.g., "outputs/04_synthesis/schemas/community"

Workflow Phases & Output Artifacts:
  - Phase 1 : Dynamic Schema DAG Discovery (Top-Level vs. Child BaseModel Classification)
  - Phase 2 : Corpus Loading & Overlapping Word-Window Chunking
  - Phase 3 : Two-Pass Hardware-Batched Extraction via PrismRunner
              * Pass 3a: Fast Multi-Schema Relevance Triage (TriageResult)
              * Pass 3b: Targeted Single-Schema Deep Extraction + Chunk-Level Caching
              * Pass 3c: Exact Verbatim Chunk Consolidation (merge_pydantic_instances)
              * outputs/05_extraction/extracts/<doc_id>/<doc_id>__<module>__<Class>.json
              * outputs/05_extraction/extracts/<doc_id>/chunks/<module>.<Class>/<doc_id>_chunk_<i>.json
              * outputs/05_extraction/extracts/<doc_id>/.completed.json
              * outputs/payload_text/stage_05_extraction/<Class>.txt
  - Phase 4 : Relational & Consolidated Wide-Table Aggregation
              * outputs/05_extraction/aggregated_tables/<Schema>.csv & <Schema>_<child>.csv
              * outputs/05_extraction/aggregated_tables/consolidated_wide.csv
              * outputs/05_extraction/aggregated_tables/consolidated_wide.parquet
              * outputs/05_extraction/aggregated_tables/aggregated_data.db (SQLite)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import importlib.util
import inspect
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

# Prevent PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
from pydantic import BaseModel, Field

# Ensure local prismrunner package is importable when running from usertests/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismrunner import (
    ContextBudget,
    DatasetLoader,
    PrismPipeline,
    clear_device_memory,
)

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base)."""
        if not text:
            return 0
        return len(_TIKTOKEN_ENCODER.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Heuristic-based estimation: 1 word ~ 1.3 tokens."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)


# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================
# Edit these variables directly to configure models, hardware, paths,
# schema input mode, triage, extraction chunking, and table aggregation.

# --- 1. Generative LLM & Hardware Settings ---
MODEL_NAME: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # Local HuggingFace causal LLM for triage & extraction
DEVICE: str = "auto"                                   # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                              # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                             # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                             # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                      # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                   # "auto" (continuous device-relative) or explicit int
TEMPERATURE: float = 0.0                               # 0.0 for deterministic greedy decoding
REPETITION_PENALTY: float = 1.05                       # Mild penalty to prevent greedy repetition loops
CONSTRAIN_JSON: bool = True                            # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                               # Flush and offload GPU/MPS memory after extraction completes
DRY_RUN: bool = False                                  # Set True to preview prompts without running LLM generation

# --- 2. Corpus Input Settings ---
INPUT_PATH: str = "inputs/sample_documents.parquet"    # .parquet, .csv, .json, .jsonl, DataFrame, or dir of .txt files
ID_COLUMN: str = "id"                                  # Document ID column name
TEXT_COLUMN: str = "text"                              # Document text column name
DOCUMENT_LIMIT: Optional[int] = None                   # Optional cap on number of documents to process (None = all)

# --- 3. Schema Input Settings (Supports Single .py File OR Directory of .py Files) ---
# Option 1 (Single file): "outputs/04_synthesis/schemas/master_ontology.py" or "outputs/04_synthesis/schemas/comprehensive_ontology.py"
# Option 2 (Directory)  : "outputs/04_synthesis/schemas/community"
SCHEMA_PATH: str = "outputs/04_synthesis/schemas/master_ontology.py"
TARGET_CLASS_NAME: Optional[str] = None                # Optional single BaseModel class name filter (None = all eligible)
EXTRACT_MODE: str = "top_level"                        # "top_level" (recommended), "child", or "all"

# --- 4. Context Window, Chunking & Triage Settings ---
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"           # "auto" = full native model context window, or int cap
MAX_NEW_TOKENS: int = 4096                             # Max generated tokens per schema extraction call
TRIAGE_MAX_NEW_TOKENS: int = 1024                      # Max generated tokens for fast schema relevance triage
CHUNK_SIZE_WORDS: int = 300                            # Target word count per text chunk
CHUNK_OVERLAP_WORDS: int = 20                          # Overlapping words between consecutive chunks
ENABLE_TRIAGE: bool = True                             # Run fast triage pass to skip irrelevant schemas per chunk
TRIAGE_THRESHOLD: int = 3                              # Only run triage when candidate schema count >= threshold
FALLBACK_ON_EMPTY_TRIAGE: bool = False                 # False = skip chunk if triage selects 0 schemas; True = run all

# --- 5. Output, Caching & Logging Settings ---
OUTPUTS_DIR: str = "outputs"                           # Base outputs directory (writes to outputs/05_extraction)
SAVE_LLM_PAYLOADS: bool = True                         # Save first rendered prompt per schema to outputs/payload_text/stage_05_extraction
RESUME_MODE: str = "skip"                              # "skip" (resume from existing extracts/chunks) or "overwrite"
PRINT_CHECKPOINTS: bool = True                         # Print timestamped progress checkpoints to console

# --- 6. Relational & Consolidated Wide-Table Aggregation Settings ---
GENERATE_CSV: bool = True                              # Export normalized relational CSVs and consolidated_wide.csv
GENERATE_SQLITE: bool = True                           # Export SQLite database (aggregated_data.db)
GENERATE_PARQUET: bool = True                          # Export consolidated_wide.parquet
DB_FILENAME: str = "aggregated_data.db"                # SQLite database filename inside outputs/05_extraction/aggregated_tables
MULTI_TABLE_RELATIONAL: bool = True                    # Generate normalized parent/child relational tables
CONSOLIDATED_WIDE_TABLE: bool = True                   # Generate 1-row-per-document consolidated_wide table
WIDE_TABLE_NAME: str = "consolidated_wide"             # Base name for consolidated wide table (.csv, .parquet, SQLite table)
INCLUDE_SOURCE_TEXT: bool = True                       # Include full document source_text column in consolidated_wide
DROP_EMPTY_RECORDS: bool = True                        # Omit all-null records when building relational/wide tables
DROP_EMPTY_TABLES: bool = True                         # Omit relational tables that have 0 extracted rows


# Consolidated ontology filenames automatically skipped in directory mode when individual community .py files exist
CONSOLIDATED_ONTOLOGY_FILENAMES: Set[str] = {
    "master_ontology.py",
    "comprehensive_ontology.py",
}


# ============================================================================
# LOGGING & CHECKPOINT UTILITIES
# ============================================================================

logger = logging.getLogger("SemanticPrism5")


def configure_logging(outputs_dir: Union[str, Path] = OUTPUTS_DIR) -> Path:
    """Configures file and console logging under outputs/logs/."""
    out_base = Path(outputs_dir)
    log_dir = out_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stage_05_extraction.log"

    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return log_file


def log_checkpoint(phase: str, message: str) -> None:
    """Prints a formatted timestamped checkpoint message to stdout and logs it."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts} | Checkpoint | {phase}] {message}"
    if PRINT_CHECKPOINTS:
        print(line, flush=True)
    logger.info(f"[{phase}] {message}")


# ============================================================================
# PHASE 1: DYNAMIC SCHEMA DISCOVERY & HIERARCHY CLASSIFICATION
# ============================================================================

@dataclass
class SchemaMeta:
    """Metadata and classification for a dynamically loaded Pydantic BaseModel schema."""
    name: str
    schema_cls: Type[BaseModel]
    source_file: Path
    module_name: str
    is_top_level: bool
    is_child: bool
    referenced_models: Set[str]

    @property
    def qualified_name(self) -> str:
        """Unique identifier combining module name and class name."""
        return f"{self.module_name}.{self.name}"


def extract_referenced_models(annotation: Any) -> Set[Type[BaseModel]]:
    """
    Recursively inspects a type annotation to find all Pydantic BaseModel subclasses
    referenced directly or inside generic containers (Optional, Union, List, Set, Tuple, Dict).
    """
    found: Set[Type[BaseModel]] = set()
    if annotation is None:
        return found

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel) and annotation is not BaseModel:
        found.add(annotation)
        return found

    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            found.update(extract_referenced_models(arg))

    return found


def resolve_schema_path_With_fallbacks(schema_path: Union[str, Path]) -> Path:
    """
    Resolves SCHEMA_PATH with smart fallback checks across:
      - `outputs/04_synthesis/schemas/`
      - `outputs/04_synthesis/schemas/community/`
      - `usertests/` (the directory containing SemanticPrism5.py)
    so single-file ontologies (`master_ontology.py`, `comprehensive_ontology.py`) and the
    `community/` directory are found regardless of working directory or storage layout.
    """
    p = Path(schema_path)
    if p.exists():
        return p

    script_dir = Path(__file__).resolve().parent
    alt_candidates: List[Path] = [
        script_dir / p,
        script_dir / p.name,
        Path(OUTPUTS_DIR) / "04_synthesis" / "schemas" / p.name,
        Path(OUTPUTS_DIR) / "04_synthesis" / "schemas" / "community" / p.name,
    ]
    if p.suffix == ".py":
        alt_candidates.extend(
            [
                p.parent / "community" / p.name,
                p.parent.parent / p.name if p.parent.name == "community" else p.parent / p.name,
            ]
        )

    for cand in alt_candidates:
        if cand.exists():
            log_checkpoint(
                "Schema Discovery",
                f"Resolved '{schema_path}' -> '{cand}' via fallback lookup.",
            )
            return cand

    raise FileNotFoundError(f"Schema path does not exist: {schema_path}")


def resolve_schema_files(schema_path: Union[str, Path]) -> List[Path]:
    """
    Resolves `schema_path` into a list of `.py` files:
      - Option 1 (Single `.py` file): returns `[file_path]`.
      - Option 2 (Directory of `.py` files): returns sorted `.py` files in the directory
        (excluding `__init__.py` and private `_*.py` files). If individual community `.py`
        files exist alongside `master_ontology.py` or `comprehensive_ontology.py`, the
        consolidated files are automatically skipped to prevent duplicate extraction.
    """
    p = resolve_schema_path_With_fallbacks(schema_path)

    if p.is_file():
        if p.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {p}")
        return [p]

    if p.is_dir():
        py_files = sorted(
            f for f in p.glob("*.py")
            if f.is_file() and not f.name.startswith("_")
        )
        if not py_files:
            # Check community subdirectory if parent schemas/ directory has no .py files directly
            comm_dir = p / "community"
            if comm_dir.is_dir():
                py_files = sorted(
                    f for f in comm_dir.glob("*.py")
                    if f.is_file() and not f.name.startswith("_")
                )
        if not py_files:
            raise ValueError(f"No valid .py schema files found in directory: {p}")

        # Smart directory exclusion guard: if non-consolidated .py files exist in this directory,
        # exclude master_ontology.py and comprehensive_ontology.py to avoid redundant extraction.
        non_consolidated = [
            f for f in py_files if f.name not in CONSOLIDATED_ONTOLOGY_FILENAMES
        ]
        if non_consolidated and len(non_consolidated) < len(py_files):
            skipped = [f.name for f in py_files if f.name in CONSOLIDATED_ONTOLOGY_FILENAMES]
            log_checkpoint(
                "Schema Discovery",
                f"Directory mode detected {len(non_consolidated)} individual schema file(s); "
                f"skipping consolidated ontology file(s): {skipped}",
            )
            return non_consolidated

        return py_files

    raise ValueError(f"Invalid schema path: {p}")


def load_schemas_from_file(py_file: Path) -> List[SchemaMeta]:
    """
    Dynamically imports a Python module and inspects all Pydantic BaseModel subclasses,
    classifying each as `is_top_level` (root entity) or `is_child` (referenced inside another BaseModel).
    """
    module_name = py_file.stem
    unique_mod_key = f"_sp5_schema_mod_{module_name}_{abs(hash(str(py_file.resolve())))}"
    spec = importlib.util.spec_from_file_location(unique_mod_key, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {py_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Error executing schema module {py_file}: {e}")
        raise

    # Collect all BaseModel subclasses defined in this file
    defined_models: Dict[str, Type[BaseModel]] = {}
    module_namespace: Dict[str, Any] = dict(vars(module))
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseModel) and obj is not BaseModel:
            if getattr(obj, "__module__", None) == unique_mod_key:
                defined_models[name] = obj

    if not defined_models:
        logger.warning(f"No Pydantic BaseModel subclasses found in {py_file}")
        return []

    # Rebuild models across two passes so forward references resolve cleanly
    for _ in range(2):
        for cls in defined_models.values():
            try:
                cls.model_rebuild(_types_namespace=module_namespace)
            except Exception:
                pass

    # Analyze field references to build parent -> child dependency graph
    referenced_by_others: Set[str] = set()
    model_references: Dict[str, Set[str]] = {}

    for name, cls in defined_models.items():
        refs: Set[str] = set()
        for field_name, field_info in cls.model_fields.items():
            ref_classes = extract_referenced_models(field_info.annotation)
            for ref_cls in ref_classes:
                if ref_cls.__name__ in defined_models and ref_cls.__name__ != name:
                    refs.add(ref_cls.__name__)
                    referenced_by_others.add(ref_cls.__name__)
        model_references[name] = refs

    schemas: List[SchemaMeta] = []
    for name, cls in defined_models.items():
        is_child = name in referenced_by_others
        is_top_level = not is_child
        schemas.append(
            SchemaMeta(
                name=name,
                schema_cls=cls,
                source_file=py_file,
                module_name=module_name,
                is_top_level=is_top_level,
                is_child=is_child,
                referenced_models=model_references[name],
            )
        )

    return schemas


def load_schemas(
    schema_path: Union[str, Path],
    target_class: Optional[str] = None,
) -> List[SchemaMeta]:
    """
    Loads all schemas from a single `.py` file or directory of `.py` files.
    Optionally filters by a specific `target_class` name.
    """
    py_files = resolve_schema_files(schema_path)
    all_schemas: List[SchemaMeta] = []

    for py_file in py_files:
        try:
            file_schemas = load_schemas_from_file(py_file)
            all_schemas.extend(file_schemas)
        except Exception as e:
            logger.error(f"Failed to load schemas from {py_file}: {e}")

    if target_class:
        filtered = [s for s in all_schemas if s.name == target_class]
        if not filtered:
            available = ", ".join(s.name for s in all_schemas)
            raise ValueError(
                f"Target class '{target_class}' not found in {schema_path}. "
                f"Available classes: [{available}]"
            )
        return filtered

    return all_schemas


def get_top_level_schemas(schemas: List[SchemaMeta]) -> List[SchemaMeta]:
    """Returns only top-level (root) schemas that are not nested exclusively inside other schemas."""
    return [s for s in schemas if s.is_top_level]


def get_child_schemas(schemas: List[SchemaMeta]) -> List[SchemaMeta]:
    """Returns only child (sub-component) schemas referenced by other schemas."""
    return [s for s in schemas if s.is_child]


# ============================================================================
# PHASE 2: CORPUS LOADING & WORD-WINDOW CHUNKING
# ============================================================================

def load_input_documents(
    input_source: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]],
    id_column: str = ID_COLUMN,
    text_column: str = TEXT_COLUMN,
    limit: Optional[int] = DOCUMENT_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Loads input documents from a `.parquet`/`.pq`, `.csv`, `.json`/`.jsonl`,
    pandas DataFrame, list of dicts, or a directory of `.txt` files.
    Returns a standardized list of `{"id": str, "text": str}` dicts.
    """
    documents: List[Dict[str, Any]] = []

    if isinstance(input_source, pd.DataFrame):
        df = input_source
        if limit is not None and limit > 0:
            df = df.head(limit)
        for idx, row in df.iterrows():
            doc_id = str(row[id_column]) if id_column in df.columns and pd.notna(row[id_column]) else f"doc_{idx}"
            text_val = str(row[text_column]) if text_column in df.columns and pd.notna(row[text_column]) else ""
            if text_val.strip():
                documents.append({"id": doc_id, "text": text_val})
        return documents

    if isinstance(input_source, list):
        items = input_source[:limit] if (limit is not None and limit > 0) else input_source
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                doc_id = str(item.get(id_column, f"doc_{idx}"))
                text_val = str(item.get(text_column, ""))
            else:
                doc_id = f"doc_{idx}"
                text_val = str(item)
            if text_val.strip():
                documents.append({"id": doc_id, "text": text_val})
        return documents

    path = Path(input_source)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        txt_files = sorted(path.glob("*.txt"))
        if limit is not None and limit > 0:
            txt_files = txt_files[:limit]
        for txt_file in txt_files:
            text_val = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if text_val:
                documents.append({"id": txt_file.stem, "text": text_val})
        return documents

    # Delegate standard file formats (.parquet, .pq, .csv, .json, .jsonl, .txt) to PrismRunner's DatasetLoader
    records = DatasetLoader.load(
        file_path_or_data=path,
        text_column=text_column,
        id_column=id_column,
        limit=limit,
    )
    for idx, rec in enumerate(records):
        if isinstance(rec, dict):
            doc_id = str(rec.get("id", f"doc_{idx}"))
            text_val = str(rec.get("text", ""))
        else:
            doc_id = str(getattr(rec, "id", f"doc_{idx}"))
            text_val = str(getattr(rec, "text", ""))
        if text_val.strip():
            documents.append({"id": doc_id, "text": text_val})

    return documents


def chunk_text(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Splits a document's text into overlapping word chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_size_words - chunk_overlap_words)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size_words >= len(words):
            break

    return chunks


# ============================================================================
# PHASE 3A: SCHEMA-DRIVEN PROMPT GENERATION & TRIAGE MODELS
# ============================================================================

EXTRACTION_RULES = """
CRITICAL EXTRACTION RULES:
1. **Verbatim Fidelity & Granular Specificity**: Extract mentions, phrases, and values EXACTLY as they appear in the source text. Do NOT paraphrase, summarize, synthesize, or generalize into high-level categories.
2. **Strict Null Rule (Zero Hallucination)**: If a specific field is NOT explicitly mentioned or supported by the text, you MUST leave it as `null` (`None`) or an empty list `[]`. NEVER guess, infer, extrapolate, or fabricate default placeholder values.
3. **Context Grounding**: Every extracted attribute must be directly grounded in the provided document excerpt. If the excerpt contains no relevant information for the entire schema, return `null` (`None`) or empty lists `[]` for all fields.
""".strip()

TRIAGE_RULES = """
CRITICAL TRIAGE RULES:
1. **Evidence-Based Selection**: Select a schema ONLY if the text excerpt contains explicit factual information, entities, or measurements that belong to that schema's domain and fields.
2. **Conservative Filtering**: Do NOT select a schema based on vague tangential associations. If no schemas are relevant to the excerpt, return an empty list `[]`.
3. **Exact Class Names**: Return ONLY exact schema class names from the provided candidate list in `relevant_schemas`.
""".strip()


class TriageResult(BaseModel):
    """Structured output for fast multi-schema relevance triage on a text chunk."""
    relevant_schemas: List[str] = Field(
        default_factory=list,
        description="List of exact schema class names that have explicit factual evidence present in the text excerpt.",
    )


def _format_type_annotation(annotation: Any) -> str:
    """Formats a Python/Pydantic type annotation into a clean, readable string for prompts."""
    if annotation is None or annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return f"Optional[{_format_type_annotation(args[0])}]"
        return "Union[" + ", ".join(_format_type_annotation(a) for a in args) + "]"

    if origin in (list, List, set, Set, tuple, Tuple):
        args = get_args(annotation)
        inner = _format_type_annotation(args[0]) if args else "Any"
        return f"List[{inner}]"

    if origin in (dict, Dict):
        args = get_args(annotation)
        if len(args) == 2:
            return f"Dict[{_format_type_annotation(args[0])}, {_format_type_annotation(args[1])}]"
        return "Dict[str, Any]"

    if inspect.isclass(annotation):
        if issubclass(annotation, Enum):
            choices = [repr(e.value) for e in annotation]
            return f"Enum({annotation.__name__}: {', '.join(choices)})"
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def format_schema_field_instructions(
    schema_cls: Type[BaseModel],
    indent: int = 0,
    visited: Optional[Set[Type[BaseModel]]] = None,
) -> str:
    """
    Recursively inspects a Pydantic BaseModel and formats its fields, types, descriptions,
    Enum choices, and nested child BaseModel definitions into structured prompt instructions.
    """
    if visited is None:
        visited = set()

    visited.add(schema_cls)
    lines: List[str] = []
    prefix = "  " * indent

    for field_name, field_info in schema_cls.model_fields.items():
        type_str = _format_type_annotation(field_info.annotation)
        desc = field_info.description or "Extract exact mentions from text if present; otherwise null."
        lines.append(f"{prefix}- `{field_name}` ({type_str}): {desc}")

        # Expand nested child BaseModel definitions recursively
        referenced = extract_referenced_models(field_info.annotation)
        for child_cls in referenced:
            if child_cls not in visited:
                child_doc = (child_cls.__doc__ or "").strip().splitlines()
                child_summary = f" — {child_doc[0].strip()}" if child_doc else ""
                lines.append(f"{prefix}  * Nested `{child_cls.__name__}` structure{child_summary}:")
                lines.append(
                    format_schema_field_instructions(
                        child_cls,
                        indent=indent + 2,
                        visited=visited.copy(),
                    )
                )

    return "\n".join(lines)


def build_triage_prompt(candidate_schemas: List[SchemaMeta]) -> str:
    """
    Constructs the system prompt for Pass 1 (Fast Multi-Schema Relevance Triage),
    listing each candidate schema's class name, docstring summary, and key fields.
    """
    schema_descriptions: List[str] = []
    for meta in candidate_schemas:
        doc = (meta.schema_cls.__doc__ or "").strip()
        first_line = doc.splitlines()[0].strip() if doc else ""
        field_names = list(meta.schema_cls.model_fields.keys())
        fields_preview = ", ".join(field_names[:10])
        if len(field_names) > 10:
            fields_preview += ", ..."
        if first_line:
            desc_str = f"{first_line} (Fields: {fields_preview})"
        else:
            desc_str = f"Extracts fields: {fields_preview}"
        schema_descriptions.append(f"- **{meta.name}**: {desc_str}")

    schemas_block = "\n".join(schema_descriptions)

    return (
        "You are an expert semantic triage router.\n"
        "Review the document excerpt and identify which of the following data schemas "
        "contain explicit factual evidence in the text.\n\n"
        "### CANDIDATE SCHEMAS\n"
        f"{schemas_block}\n\n"
        f"{TRIAGE_RULES}"
    )


def build_extraction_prompt(schema_cls: Type[BaseModel]) -> str:
    """
    Constructs the schema-specific system prompt for Pass 2 (Deep Structured Extraction).
    """
    doc = (schema_cls.__doc__ or "").strip()
    doc_section = f"\nSchema Purpose: {doc}\n" if doc else "\n"
    fields_section = format_schema_field_instructions(schema_cls)

    return (
        f"You are an expert precision data extraction agent for the `{schema_cls.__name__}` schema."
        f"{doc_section}\n"
        f"### TARGET SCHEMA FIELDS (`{schema_cls.__name__}`)\n"
        f"{fields_section}\n\n"
        f"{EXTRACTION_RULES}"
    )


# ============================================================================
# PHASE 3B: RESUME TRACKER & VERBATIM CHUNK CONSOLIDATION
# ============================================================================

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that safely serializes Enums, Pydantic models, sets, and paths."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for filesystem paths and filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


class ResumeTracker:
    """
    Tracks extraction progress at the document, schema, and individual chunk level
    under `outputs/05_extraction/extracts/<sanitized_doc_id>/`.
    """

    def __init__(self, extracts_dir: Path, mode: str = RESUME_MODE):
        self.extracts_dir = extracts_dir
        self.mode = mode
        self.extracts_dir.mkdir(parents=True, exist_ok=True)

    def get_doc_dir(self, doc_id: str) -> Path:
        return self.extracts_dir / sanitize_filename(doc_id)

    def get_manifest_path(self, doc_id: str) -> Path:
        return self.get_doc_dir(doc_id) / ".completed.json"

    def get_output_file(self, doc_id: str, schema_meta: SchemaMeta) -> Path:
        safe_doc_id = sanitize_filename(doc_id)
        filename = f"{safe_doc_id}__{schema_meta.module_name}__{schema_meta.name}.json"
        return self.get_doc_dir(doc_id) / filename

    def get_chunk_cache_dir(self, doc_id: str, schema_meta: SchemaMeta) -> Path:
        return self.get_doc_dir(doc_id) / "chunks" / sanitize_filename(schema_meta.qualified_name)

    def get_chunk_file(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> Path:
        safe_doc_id = sanitize_filename(doc_id)
        return self.get_chunk_cache_dir(doc_id, schema_meta) / f"{safe_doc_id}_chunk_{chunk_idx}.json"

    def is_chunk_completed(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> bool:
        if self.mode == "overwrite":
            return False
        return self.get_chunk_file(doc_id, schema_meta, chunk_idx).exists()

    def load_chunk_result(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> Optional[BaseModel]:
        chunk_file = self.get_chunk_file(doc_id, schema_meta, chunk_idx)
        if not chunk_file.exists():
            return None
        try:
            data = json.loads(chunk_file.read_text(encoding="utf-8"))
            if not data:
                return None
            return schema_meta.schema_cls.model_validate(data)
        except Exception as e:
            logger.warning(f"Corrupted chunk cache {chunk_file}, will re-extract: {e}")
            return None

    def save_chunk_result(
        self,
        doc_id: str,
        schema_meta: SchemaMeta,
        chunk_idx: int,
        instance: Optional[BaseModel],
    ) -> Path:
        chunk_file = self.get_chunk_file(doc_id, schema_meta, chunk_idx)
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        if instance is None or not has_extracted_content(instance):
            payload_str = json.dumps(None)
        else:
            payload_str = json.dumps(instance.model_dump(mode="json"), indent=2, cls=SafeJSONEncoder)
        chunk_file.write_text(payload_str, encoding="utf-8")
        return chunk_file

    def get_completed_schemas(self, doc_id: str) -> Set[str]:
        if self.mode == "overwrite":
            return set()
        manifest = self.get_manifest_path(doc_id)
        if not manifest.exists():
            return set()
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return set(data.get("completed_schemas", []))
        except Exception:
            return set()

    def is_schema_completed(self, doc_id: str, schema_meta: SchemaMeta) -> bool:
        if self.mode == "overwrite":
            return False
        completed = self.get_completed_schemas(doc_id)
        out_file = self.get_output_file(doc_id, schema_meta)
        return schema_meta.qualified_name in completed and out_file.exists()

    def mark_schema_completed(self, doc_id: str, schema_meta: SchemaMeta) -> None:
        doc_dir = self.get_doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.get_manifest_path(doc_id)

        completed = self.get_completed_schemas(doc_id)
        completed.add(schema_meta.qualified_name)

        data = {
            "doc_id": str(doc_id),
            "completed_schemas": sorted(list(completed)),
        }
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_extraction(
        self,
        doc_id: str,
        schema_meta: SchemaMeta,
        extracted_data: Any,
    ) -> Path:
        out_file = self.get_output_file(doc_id, schema_meta)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(extracted_data, indent=2, cls=SafeJSONEncoder),
            encoding="utf-8",
        )
        self.mark_schema_completed(doc_id, schema_meta)
        return out_file


def _is_empty_value(val: Any) -> bool:
    """Checks if an extracted field value is null or empty."""
    if val is None:
        return True
    if isinstance(val, (str, list, dict, set, tuple)) and len(val) == 0:
        return True
    if isinstance(val, BaseModel):
        return not has_extracted_content(val)
    return False


def has_extracted_content(instance: BaseModel) -> bool:
    """
    Returns True if the Pydantic model instance has at least one non-null, non-empty field.
    """
    for field_name in instance.model_fields:
        val = getattr(instance, field_name, None)
        if isinstance(val, list):
            non_empty_items = [item for item in val if not _is_empty_value(item)]
            if non_empty_items:
                return True
        elif not _is_empty_value(val):
            return True
    return False


def _canonical_key(item: Any) -> str:
    """Generates a deterministic canonical JSON string key for deduplicating list items."""
    if isinstance(item, BaseModel):
        return json.dumps(item.model_dump(mode="json"), sort_keys=True, cls=SafeJSONEncoder)
    if isinstance(item, (dict, list)):
        return json.dumps(item, sort_keys=True, cls=SafeJSONEncoder)
    if isinstance(item, Enum):
        return str(item.value)
    return str(item)


def merge_pydantic_instances(
    instances: List[BaseModel],
    schema_cls: Type[BaseModel],
) -> Optional[BaseModel]:
    """
    Consolidates multiple chunk-level Pydantic model instances into a single document-level
    instance WITHOUT mathematical alteration, averaging, or string modification:
      - `list` fields: concatenates and deduplicates items while preserving exact extraction order.
      - `dict` fields: merges keys in order without mutating values.
      - Nested `BaseModel` fields: recursively consolidates child fields.
      - Scalar fields (`str`, `int`, `float`, `bool`, `Enum`): retains the first non-null
        extracted value verbatim.
    """
    valid_instances = [inst for inst in instances if inst is not None and has_extracted_content(inst)]
    if not valid_instances:
        return None

    if len(valid_instances) == 1:
        return valid_instances[0]

    merged_data: Dict[str, Any] = {}

    for field_name, field_info in schema_cls.model_fields.items():
        field_values = [
            getattr(inst, field_name, None)
            for inst in valid_instances
            if getattr(inst, field_name, None) is not None
        ]

        if not field_values:
            merged_data[field_name] = None
            continue

        first_val = field_values[0]

        # Case 1: List fields -> concatenate & deduplicate preserving order
        if isinstance(first_val, list):
            combined_list: List[Any] = []
            seen_keys: Set[str] = set()
            for val_list in field_values:
                if not isinstance(val_list, list):
                    continue
                for item in val_list:
                    if _is_empty_value(item):
                        continue
                    key = _canonical_key(item)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        combined_list.append(item)
            merged_data[field_name] = combined_list

        # Case 2: Dict fields -> merge keys in order
        elif isinstance(first_val, dict):
            combined_dict: Dict[Any, Any] = {}
            for val_dict in field_values:
                if isinstance(val_dict, dict):
                    for k, v in val_dict.items():
                        if k not in combined_dict and not _is_empty_value(v):
                            combined_dict[k] = v
            merged_data[field_name] = combined_dict

        # Case 3: Nested BaseModel -> recursively merge
        elif isinstance(first_val, BaseModel):
            child_cls = type(first_val)
            nested_models = [v for v in field_values if isinstance(v, BaseModel)]
            merged_data[field_name] = merge_pydantic_instances(nested_models, child_cls)

        # Case 4: Scalar (str, int, float, bool, Enum) -> retain first non-null value verbatim
        else:
            chosen_scalar = None
            for v in field_values:
                if not _is_empty_value(v):
                    chosen_scalar = v
                    break
            merged_data[field_name] = chosen_scalar

    try:
        return schema_cls.model_validate(merged_data)
    except Exception:
        return schema_cls(**merged_data)


# ============================================================================
# PHASE 4: RELATIONAL & CONSOLIDATED WIDE-TABLE AGGREGATION
# ============================================================================

def is_scalar(val: Any) -> bool:
    """Returns True if the value is a primitive scalar (or None)."""
    return val is None or isinstance(val, (str, int, float, bool))


def infer_sqlite_type(values: List[Any]) -> str:
    """Infers the appropriate SQLite column affinity (INTEGER, REAL, TEXT) from sample values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "TEXT"
    if all(isinstance(v, bool) for v in non_null):
        return "INTEGER"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "INTEGER"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "REAL"
    return "TEXT"


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = "__",
) -> Dict[str, Any]:
    """
    Flattens nested dictionaries into `parent__child` keys while keeping lists at the top level
    so they can be routed either to child relational tables or JSON-serialized columns.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, dict) and not v:
            items.append((new_key, None))
        else:
            items.append((new_key, v))
    return dict(items)


def extract_relational_tables(
    records: List[Dict[str, Any]],
    root_table_name: str,
    multi_table: bool = MULTI_TABLE_RELATIONAL,
    drop_empty_records: bool = DROP_EMPTY_RECORDS,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transforms extracted JSON records for a schema into normalized relational tables:
      - Parent table `<root_table_name>`
      - Child tables `<root_table_name>_<list_field>` linked via `source_id`, `source_file`,
        `parent_item_index`, and `item_index`.
    """
    tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in records:
        source_id = rec.get("_source_id", "unknown")
        source_file = rec.get("_source_file", "unknown")
        parent_idx = rec.get("_item_index", 0)

        payload = {k: v for k, v in rec.items() if not k.startswith("_")}
        flat = flatten_dict(payload)

        if drop_empty_records:
            has_data = any(
                v is not None and v != [] and v != {} and v != ""
                for v in flat.values()
            )
            if not has_data:
                continue

        parent_row: Dict[str, Any] = {
            "source_id": source_id,
            "source_file": source_file,
            "item_index": parent_idx,
        }

        for key, val in flat.items():
            if isinstance(val, list):
                if not multi_table:
                    parent_row[key] = json.dumps(val, ensure_ascii=False) if val else None
                else:
                    child_table_name = f"{root_table_name}_{key}"
                    for child_idx, item in enumerate(val):
                        if item is None or item == "" or item == {} or item == []:
                            continue
                        child_row: Dict[str, Any] = {
                            "source_id": source_id,
                            "source_file": source_file,
                            "parent_item_index": parent_idx,
                            "item_index": child_idx,
                        }
                        if isinstance(item, dict):
                            flat_child = flatten_dict(item)
                            if drop_empty_records and not any(
                                cv is not None and cv != [] and cv != {} and cv != ""
                                for cv in flat_child.values()
                            ):
                                continue
                            for ck, cv in flat_child.items():
                                if isinstance(cv, (list, dict)):
                                    child_row[ck] = json.dumps(cv, ensure_ascii=False) if cv else None
                                else:
                                    child_row[ck] = cv
                        else:
                            child_row["value"] = item
                        tables[child_table_name].append(child_row)
            else:
                parent_row[key] = val

        tables[root_table_name].append(parent_row)

    return tables


def build_consolidated_wide_table(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    source_texts: Optional[Dict[str, str]] = None,
    include_source_text: bool = INCLUDE_SOURCE_TEXT,
    drop_empty_records: bool = DROP_EMPTY_RECORDS,
) -> List[Dict[str, Any]]:
    """
    Consolidates all extracted schemas across the corpus into a single wide table where
    each row corresponds to one unique document (`source_id`):
      - Scalar fields -> `<Schema>__<Field>`
      - List of scalars -> JSON array in `<Schema>__<Field>` + count in `<Schema>__<Field>__count`
      - List of child objects -> JSON array in `<Schema>__<Field>` + count in `<Schema>__<Field>__count`
        + exploded child attributes `<Schema>__<Field>__<attr>`
    """
    doc_rows: Dict[str, Dict[str, Any]] = {}
    all_columns: List[str] = ["source_id"]
    if include_source_text and source_texts:
        all_columns.append("source_text")
    seen_columns: Set[str] = set(all_columns)

    def register_col(col_name: str) -> None:
        if col_name not in seen_columns:
            seen_columns.add(col_name)
            all_columns.append(col_name)

    if source_texts:
        for doc_id, text in source_texts.items():
            doc_rows[str(doc_id)] = {"source_id": str(doc_id)}
            if include_source_text:
                doc_rows[str(doc_id)]["source_text"] = text

    # Check if any schema class name appears in multiple modules to disambiguate prefix if needed
    schema_name_counts: Dict[str, int] = defaultdict(int)
    for mod_name, sch_name in grouped_records.keys():
        schema_name_counts[sch_name] += 1

    for (module_name, schema_name), records in sorted(grouped_records.items()):
        schema_prefix = (
            f"{module_name}__{schema_name}"
            if schema_name_counts[schema_name] > 1
            else schema_name
        )

        by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            doc_id = str(rec.get("_source_id", "unknown"))
            payload = {k: v for k, v in rec.items() if not k.startswith("_")}
            flat = flatten_dict(payload)

            if drop_empty_records:
                has_data = any(
                    v is not None and v != [] and v != {} and v != ""
                    for v in flat.values()
                )
                if not has_data:
                    continue

            by_doc[doc_id].append(flat)

        for doc_id, flat_items in by_doc.items():
            if doc_id not in doc_rows:
                doc_rows[doc_id] = {"source_id": doc_id}
                if include_source_text and source_texts is not None:
                    doc_rows[doc_id]["source_text"] = source_texts.get(doc_id)

            row = doc_rows[doc_id]

            all_keys: List[str] = []
            for item in flat_items:
                for k in item.keys():
                    if k not in all_keys:
                        all_keys.append(k)

            for key in all_keys:
                col_base = f"{schema_prefix}__{key}"
                vals = [
                    item.get(key)
                    for item in flat_items
                    if item.get(key) is not None and item.get(key) != "" and item.get(key) != []
                ]

                if not vals:
                    register_col(col_base)
                    row.setdefault(col_base, None)
                    continue

                first_val = vals[0]

                if isinstance(first_val, list):
                    combined_list: List[Any] = []
                    for v in vals:
                        if isinstance(v, list):
                            for elem in v:
                                if elem is not None and elem != "" and elem != {} and elem != []:
                                    combined_list.append(elem)

                    register_col(col_base)
                    count_col = f"{col_base}__count"
                    register_col(count_col)

                    if not combined_list:
                        row[col_base] = None
                        row[count_col] = 0
                        continue

                    row[col_base] = json.dumps(combined_list, ensure_ascii=False)
                    row[count_col] = len(combined_list)

                    # Explode child dictionary attributes into <Schema>__<Field>__<attr>
                    dict_elems = [e for e in combined_list if isinstance(e, dict)]
                    if dict_elems:
                        sub_keys: List[str] = []
                        flat_dicts = [flatten_dict(e) for e in dict_elems]
                        for fd in flat_dicts:
                            for sk in fd.keys():
                                if sk not in sub_keys:
                                    sub_keys.append(sk)

                        for sk in sub_keys:
                            sub_col = f"{col_base}__{sk}"
                            register_col(sub_col)
                            sub_vals = [
                                fd.get(sk)
                                for fd in flat_dicts
                                if fd.get(sk) is not None and fd.get(sk) != "" and fd.get(sk) != []
                            ]
                            if not sub_vals:
                                row[sub_col] = None
                            elif len(sub_vals) == 1 and is_scalar(sub_vals[0]):
                                row[sub_col] = sub_vals[0]
                            else:
                                unique_sub: List[Any] = []
                                for sv in sub_vals:
                                    if sv not in unique_sub:
                                        unique_sub.append(sv)
                                row[sub_col] = (
                                    unique_sub[0]
                                    if (len(unique_sub) == 1 and is_scalar(unique_sub[0]))
                                    else json.dumps(unique_sub, ensure_ascii=False)
                                )
                else:
                    register_col(col_base)
                    if len(vals) == 1:
                        row[col_base] = vals[0]
                    else:
                        unique_vals: List[Any] = []
                        for v in vals:
                            if v not in unique_vals:
                                unique_vals.append(v)
                        row[col_base] = (
                            unique_vals[0]
                            if len(unique_vals) == 1
                            else json.dumps(unique_vals, ensure_ascii=False)
                        )

    final_rows: List[Dict[str, Any]] = []
    for doc_id in sorted(doc_rows.keys()):
        raw_row = doc_rows[doc_id]
        if drop_empty_records:
            has_extracted_cols = any(
                k not in ("source_id", "source_text")
                and not k.endswith("__count")
                and v is not None
                and v != ""
                and v != "[]"
                for k, v in raw_row.items()
            )
            if not has_extracted_cols:
                continue

        standardized = {col: raw_row.get(col, None) for col in all_columns}
        final_rows.append(standardized)

    return final_rows


class DataAggregator:
    """
    Reads per-document JSON extraction files from `outputs/05_extraction/extracts/` and writes:
      1. Normalized relational CSV tables & SQLite tables (`MULTI_TABLE_RELATIONAL = True`)
      2. Consolidated wide table (`consolidated_wide.csv`, `consolidated_wide.parquet`, and
         the `consolidated_wide` table inside `aggregated_data.db`).
    """

    def __init__(
        self,
        extracts_dir: Union[str, Path],
        tables_dir: Union[str, Path],
        generate_csv: bool = GENERATE_CSV,
        generate_sqlite: bool = GENERATE_SQLITE,
        generate_parquet: bool = GENERATE_PARQUET,
        db_filename: str = DB_FILENAME,
        multi_table_relational: bool = MULTI_TABLE_RELATIONAL,
        consolidated_wide_table: bool = CONSOLIDATED_WIDE_TABLE,
        wide_table_name: str = WIDE_TABLE_NAME,
        include_source_text: bool = INCLUDE_SOURCE_TEXT,
        drop_empty_records: bool = DROP_EMPTY_RECORDS,
        drop_empty_tables: bool = DROP_EMPTY_TABLES,
    ):
        self.extracts_dir = Path(extracts_dir)
        self.tables_dir = Path(tables_dir)
        self.generate_csv = generate_csv
        self.generate_sqlite = generate_sqlite
        self.generate_parquet = generate_parquet
        self.db_filename = db_filename
        self.multi_table_relational = multi_table_relational
        self.consolidated_wide_table = consolidated_wide_table
        self.wide_table_name = wide_table_name
        self.include_source_text = include_source_text
        self.drop_empty_records = drop_empty_records
        self.drop_empty_tables = drop_empty_tables

    @staticmethod
    def parse_extraction_filename(file_path: Path) -> Tuple[str, str, str]:
        """Parses `<doc_id>__<module_name>__<class_name>.json`."""
        parts = file_path.stem.split("__")
        if len(parts) >= 3:
            doc_id = "__".join(parts[:-2])
            module_name = parts[-2]
            schema_name = parts[-1]
            return doc_id, module_name, schema_name
        return "unknown", "default", file_path.stem

    def collect_extractions(self) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Scans `extracts_dir` for document-level `.json` extraction files (excluding `chunks/` and `.completed.json`)."""
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        if not self.extracts_dir.exists():
            logger.warning(f"Extracts directory does not exist: {self.extracts_dir}")
            return grouped

        json_files = sorted(
            f for f in self.extracts_dir.rglob("*.json")
            if not f.name.startswith(".") and "chunks" not in f.parts
        )

        for jf in json_files:
            doc_id, module_name, schema_name = self.parse_extraction_filename(jf)
            try:
                content = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to read JSON file {jf}: {e}")
                continue

            if not content:
                continue

            items = content if isinstance(content, list) else [content]
            for idx, item in enumerate(items):
                if isinstance(item, dict) and item:
                    record: Dict[str, Any] = {
                        "_source_id": doc_id,
                        "_source_file": jf.name,
                        "_item_index": idx,
                    }
                    record.update(item)
                    grouped[(module_name, schema_name)].append(record)

        return grouped

    def run(self, source_texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Executes relational and wide-table aggregation, returning summary statistics."""
        log_checkpoint("Phase 4: Aggregation", f"Scanning extraction outputs in '{self.extracts_dir}'...")
        grouped = self.collect_extractions()

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        all_relational_tables: Dict[str, List[Dict[str, Any]]] = {}

        schema_name_counts: Dict[str, int] = defaultdict(int)
        for mod_name, sch_name in grouped.keys():
            schema_name_counts[sch_name] += 1

        for (module_name, schema_name), records in grouped.items():
            if not records:
                continue

            root_table_name = (
                f"{module_name}_{schema_name}"
                if schema_name_counts[schema_name] > 1
                else schema_name
            )
            extracted_tables = extract_relational_tables(
                records=records,
                root_table_name=root_table_name,
                multi_table=self.multi_table_relational,
                drop_empty_records=self.drop_empty_records,
            )

            for t_name, t_rows in extracted_tables.items():
                if self.drop_empty_tables and not t_rows:
                    continue
                if t_name in all_relational_tables:
                    all_relational_tables[t_name].extend(t_rows)
                else:
                    all_relational_tables[t_name] = t_rows

        wide_rows: List[Dict[str, Any]] = []
        if self.consolidated_wide_table and (grouped or source_texts):
            wide_rows = build_consolidated_wide_table(
                grouped_records=grouped,
                source_texts=source_texts,
                include_source_text=self.include_source_text,
                drop_empty_records=self.drop_empty_records,
            )
            if wide_rows or not self.drop_empty_tables:
                all_relational_tables[self.wide_table_name] = wide_rows
                log_checkpoint(
                    "Phase 4: Aggregation",
                    f"Built consolidated wide table '{self.wide_table_name}' with {len(wide_rows)} document row(s) "
                    f"and {len(wide_rows[0]) if wide_rows else 0} column(s).",
                )

        if not all_relational_tables:
            log_checkpoint("Phase 4: Aggregation", "No non-empty records found to aggregate.")
            return {"tables_created": 0, "wide_rows": 0}

        if self.generate_csv:
            self._export_csvs(all_relational_tables)

        if self.generate_parquet and wide_rows:
            self._export_wide_parquet(wide_rows)

        if self.generate_sqlite:
            self._export_sqlite(all_relational_tables)

        return {
            "tables_created": len(all_relational_tables),
            "table_names": list(all_relational_tables.keys()),
            "wide_rows": len(wide_rows),
        }

    def _export_csvs(self, all_tables: Dict[str, List[Dict[str, Any]]]) -> None:
        for table_name, rows in all_tables.items():
            if not rows:
                continue
            safe_name = sanitize_filename(table_name)
            csv_path = self.tables_dir / f"{safe_name}.csv"
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            log_checkpoint(
                "Phase 4: CSV Export",
                f"Saved '{csv_path.name}' ({len(df)} rows, {len(df.columns)} columns).",
            )

    def _export_wide_parquet(self, wide_rows: List[Dict[str, Any]]) -> None:
        safe_name = sanitize_filename(self.wide_table_name)
        pq_path = self.tables_dir / f"{safe_name}.parquet"
        try:
            df = pd.DataFrame(wide_rows)
            # Ensure mixed object columns serialize cleanly for Arrow/Parquet
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False)
                        if isinstance(x, (list, dict))
                        else (str(x) if x is not None and not isinstance(x, str) else x)
                    )
            df.to_parquet(pq_path, index=False)
            log_checkpoint(
                "Phase 4: Parquet Export",
                f"Saved '{pq_path.name}' ({len(df)} rows, {len(df.columns)} columns).",
            )
        except Exception as e:
            logger.warning(f"Could not export Parquet file {pq_path}: {e}")

    def _export_sqlite(self, all_tables: Dict[str, List[Dict[str, Any]]]) -> None:
        db_path = self.tables_dir / self.db_filename
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            for raw_table_name, rows in all_tables.items():
                if not rows:
                    continue

                table_name = sanitize_filename(raw_table_name)
                columns: List[str] = []
                for r in rows:
                    for k in r.keys():
                        if k not in columns:
                            columns.append(k)

                col_defs: List[str] = []
                for col in columns:
                    sample_vals = [r.get(col) for r in rows]
                    col_type = infer_sqlite_type(sample_vals)
                    safe_col = col.replace('"', '""')
                    col_defs.append(f'"{safe_col}" {col_type}')

                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                create_stmt = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
                cursor.execute(create_stmt)

                placeholders = ", ".join(["?"] * len(columns))
                quoted_cols = ", ".join([f'"{c.replace(chr(34), chr(34)*2)}"' for c in columns])
                insert_stmt = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

                values_to_insert: List[Tuple[Any, ...]] = []
                for r in rows:
                    row_tuple: List[Any] = []
                    for col in columns:
                        val = r.get(col)
                        if isinstance(val, bool):
                            val = int(val)
                        elif isinstance(val, (list, dict)):
                            val = json.dumps(val, ensure_ascii=False)
                        row_tuple.append(val)
                    values_to_insert.append(tuple(row_tuple))

                cursor.executemany(insert_stmt, values_to_insert)

                if "source_id" in columns:
                    idx_name = sanitize_filename(f"idx_{table_name}_source_id")
                    cursor.execute(
                        f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("source_id")'
                    )

            conn.commit()
            log_checkpoint(
                "Phase 4: SQLite Export",
                f"Saved relational & wide SQLite database to '{db_path}' ({len(all_tables)} tables).",
            )
        except Exception as e:
            logger.error(f"SQLite export failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()


# ============================================================================
# MAIN STAGE 5 PIPELINE ORCHESTRATOR (POWERED BY PRISMRUNNER)
# ============================================================================

class SemanticPrismStage5Pipeline:
    """
    End-to-end Stage 5 Schema-Driven Extraction and Relational/Wide-Table Aggregation
    pipeline powered by PrismRunner.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        torch_dtype: str = TORCH_DTYPE,
        load_in_4bit: bool = LOAD_IN_4BIT,
        load_in_8bit: bool = LOAD_IN_8BIT,
        attn_implementation: str = ATTN_IMPLEMENTATION,
        batch_size: Union[int, str] = BATCH_SIZE,
        temperature: float = TEMPERATURE,
        repetition_penalty: float = REPETITION_PENALTY,
        constrain_json: bool = CONSTRAIN_JSON,
        context_window_cap: Union[int, str] = CONTEXT_WINDOW_CAP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        triage_max_new_tokens: int = TRIAGE_MAX_NEW_TOKENS,
        chunk_size_words: int = CHUNK_SIZE_WORDS,
        chunk_overlap_words: int = CHUNK_OVERLAP_WORDS,
        enable_triage: bool = ENABLE_TRIAGE,
        triage_threshold: int = TRIAGE_THRESHOLD,
        fallback_on_empty_triage: bool = FALLBACK_ON_EMPTY_TRIAGE,
        schema_path: Union[str, Path] = SCHEMA_PATH,
        target_class_name: Optional[str] = TARGET_CLASS_NAME,
        extract_mode: str = EXTRACT_MODE,
        outputs_dir: Union[str, Path] = OUTPUTS_DIR,
        save_llm_payloads: bool = SAVE_LLM_PAYLOADS,
        resume_mode: str = RESUME_MODE,
        manage_vram: bool = MANAGE_VRAM,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.constrain_json = constrain_json
        self.context_window_cap = context_window_cap
        self.max_new_tokens = max_new_tokens
        self.triage_max_new_tokens = triage_max_new_tokens
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.enable_triage = enable_triage
        self.triage_threshold = triage_threshold
        self.fallback_on_empty_triage = fallback_on_empty_triage
        self.schema_path = Path(schema_path)
        self.target_class_name = target_class_name
        self.extract_mode = extract_mode
        self.outputs_dir = Path(outputs_dir)
        self.save_llm_payloads = save_llm_payloads
        self.resume_mode = resume_mode
        self.manage_vram = manage_vram

        # Standard Stage 5 directory layout under outputs/05_extraction
        self.stage5_dir = self.outputs_dir / "05_extraction"
        self.extracts_dir = self.stage5_dir / "extracts"
        self.tables_dir = self.stage5_dir / "aggregated_tables"
        self.payload_dir = self.outputs_dir / "payload_text" / "stage_05_extraction"

        self.extracts_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        if self.save_llm_payloads:
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        configure_logging(self.outputs_dir)
        self.tracker = ResumeTracker(self.extracts_dir, mode=self.resume_mode)

        # Initialize PrismRunner pipeline
        self.prism = PrismPipeline(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            config={
                "model": {
                    "name": self.model_name,
                    "device": self.device,
                    "torch_dtype": self.torch_dtype,
                    "load_in_4bit": self.load_in_4bit,
                    "load_in_8bit": self.load_in_8bit,
                    "attn_implementation": self.attn_implementation,
                },
                "inference": {
                    "batch_size": self.batch_size,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "repetition_penalty": self.repetition_penalty,
                    "constrain_json": self.constrain_json,
                },
            },
        )
        if isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            self.prism.config.setdefault("model", {})["context_window_cap"] = self.context_window_cap

        self._saved_payload_keys: Set[str] = set()

    def _maybe_save_payload(self, key: str, system_prompt: str, user_prompt: str) -> None:
        """Saves the first rendered prompt per schema/triage pass to `outputs/payload_text/stage_05_extraction/`."""
        if not self.save_llm_payloads or key in self._saved_payload_keys:
            return
        self._saved_payload_keys.add(key)
        safe_key = sanitize_filename(key)
        out_path = self.payload_dir / f"{safe_key}.txt"
        content = (
            f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n"
        )
        out_path.write_text(content, encoding="utf-8")

    def resolve_target_schemas(self) -> List[SchemaMeta]:
        """
        Loads schemas from `self.schema_path` (single `.py` file or directory of `.py` files)
        and filters them according to `self.target_class_name` and `self.extract_mode`.
        """
        all_schemas = load_schemas(self.schema_path, target_class=self.target_class_name)
        top_level = get_top_level_schemas(all_schemas)
        children = get_child_schemas(all_schemas)

        log_checkpoint(
            "Phase 1: Schema Discovery",
            f"Loaded {len(all_schemas)} total BaseModel(s) from '{self.schema_path}' "
            f"({len(top_level)} top-level, {len(children)} child sub-models).",
        )

        if self.target_class_name:
            target_schemas = all_schemas
        elif self.extract_mode == "top_level":
            target_schemas = top_level
        elif self.extract_mode == "child":
            target_schemas = children
        else:
            target_schemas = all_schemas

        if not target_schemas:
            raise ValueError(
                f"No target schemas selected from '{self.schema_path}' with extract_mode='{self.extract_mode}'."
            )

        schema_names_preview = ", ".join(s.qualified_name for s in target_schemas[:10])
        if len(target_schemas) > 10:
            schema_names_preview += f", ... (+{len(target_schemas) - 10} more)"
        log_checkpoint(
            "Phase 1: Schema Discovery",
            f"Selected {len(target_schemas)} target schema(s) for extraction ({self.extract_mode}): [{schema_names_preview}]",
        )
        return target_schemas

    def _safe_chunk_document(self, text: str) -> List[str]:
        """
        Splits document text into overlapping word chunks while enforcing `ContextBudget`
        safety against the model's context window.
        """
        ctx_window = (
            self.context_window_cap
            if isinstance(self.context_window_cap, int) and self.context_window_cap > 0
            else getattr(self.prism, "context_window", 32768) or 32768
        )
        try:
            budget = ContextBudget.from_tokenizer(
                tokenizer=getattr(self.prism, "tokenizer", None),
                context_limit=ctx_window,
                max_new_tokens=self.max_new_tokens,
            )
            max_chunk_tokens = budget.max_input_tokens
        except ValueError:
            max_chunk_tokens = max(256, ctx_window // 2)
        # Convert token cap to approximate word cap (1 word ~ 1.35 tokens)
        safe_word_cap = max(50, int(max_chunk_tokens / 1.35))
        effective_words = min(self.chunk_size_words, safe_word_cap)
        effective_overlap = min(self.chunk_overlap_words, max(0, effective_words // 3))
        return chunk_text(text, chunk_size_words=effective_words, chunk_overlap_words=effective_overlap)

    def _run_triage_pass(
        self,
        chunk_records: List[Dict[str, Any]],
        target_schemas: List[SchemaMeta],
        dry_run: bool = False,
    ) -> Dict[Tuple[str, int], List[SchemaMeta]]:
        """
        Executes Pass 3a (Fast Multi-Schema Relevance Triage) across all pending chunks
        in a single hardware-batched PrismRunner pass.
        """
        chunk_to_schemas: Dict[Tuple[str, int], List[SchemaMeta]] = {}
        use_triage = (
            self.enable_triage
            and len(target_schemas) >= self.triage_threshold
            and self.target_class_name is None
        )

        if not use_triage:
            for rec in chunk_records:
                doc_id = rec["doc_id"]
                chunk_idx = rec["chunk_idx"]
                pending_for_doc = rec["pending_schemas"]
                chunk_to_schemas[(doc_id, chunk_idx)] = pending_for_doc
            return chunk_to_schemas

        # Only triage chunks that have at least one uncached schema
        triage_inputs: List[Dict[str, Any]] = []
        triage_meta_by_id: Dict[str, Dict[str, Any]] = {}
        for rec in chunk_records:
            rec_id = rec["id"]
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]
            uncached = [
                s for s in pending_for_doc
                if not self.tracker.is_chunk_completed(doc_id, s, chunk_idx)
            ]
            if not uncached:
                # All pending schemas for this chunk already have cached chunk JSONs
                chunk_to_schemas[(doc_id, chunk_idx)] = pending_for_doc
            else:
                triage_inputs.append({"id": rec_id, "text": rec["text"]})
                triage_meta_by_id[rec_id] = rec

        if not triage_inputs:
            return chunk_to_schemas

        triage_sys_prompt = build_triage_prompt(target_schemas)
        triage_instruction = (
            "Identify all candidate schemas from the system prompt that have explicit factual "
            "information present in the text excerpt above. Return their exact class names in `relevant_schemas`."
        )

        if triage_inputs:
            self._maybe_save_payload(
                "_SCHEMA_RELEVANCE_TRIAGE",
                triage_sys_prompt,
                f"Document Excerpt:\n{triage_inputs[0]['text']}\n\nInstruction:\n{triage_instruction}",
            )

        log_checkpoint(
            "Phase 3a: Schema Triage",
            f"Running batched relevance triage across {len(triage_inputs)} chunk(s) "
            f"against {len(target_schemas)} candidate schemas...",
        )

        triage_results = self.prism.run(
            input_data=triage_inputs,
            id_column="id",
            text_column="text",
            task="extract",
            schema_cls=TriageResult,
            system_prompt=triage_sys_prompt,
            instruction=triage_instruction,
            max_new_tokens=self.triage_max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            batch_size=self.batch_size,
            chunk=False,
            as_array=False,
            constrain_json=self.constrain_json,
            dedup=False,
            dry_run=dry_run,
        )

        if dry_run:
            for rec_id, rec in triage_meta_by_id.items():
                chunk_to_schemas[(rec["doc_id"], rec["chunk_idx"])] = rec["pending_schemas"]
            return chunk_to_schemas

        # Map returned results by chunk record id ("record_id" or "id")
        results_by_id = {
            str(res.get("record_id") or res.get("id") or ""): res
            for res in triage_results
        }
        total_selected = 0

        for rec_id, rec in triage_meta_by_id.items():
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]

            res = results_by_id.get(rec_id, {})
            extracted_items = res.get("extracted_items") or res.get("parsed_items") or []
            selected_names_lower: Set[str] = set()

            for item in extracted_items:
                if isinstance(item, dict):
                    names = item.get("relevant_schemas", [])
                elif isinstance(item, TriageResult):
                    names = item.relevant_schemas
                else:
                    names = []
                if isinstance(names, list):
                    for n in names:
                        if isinstance(n, str) and n.strip():
                            selected_names_lower.add(n.strip().lower())

            matched = [
                s for s in pending_for_doc
                if s.name.lower() in selected_names_lower
                or s.qualified_name.lower() in selected_names_lower
            ]

            if not matched and self.fallback_on_empty_triage:
                matched = pending_for_doc

            chunk_to_schemas[(doc_id, chunk_idx)] = matched
            total_selected += len(matched)

        total_possible = len(triage_inputs) * len(target_schemas)
        skipped_calls = max(0, total_possible - total_selected)
        log_checkpoint(
            "Phase 3a: Schema Triage",
            f"Triage complete: selected {total_selected}/{total_possible} (chunk, schema) pairs "
            f"(pruned {skipped_calls} irrelevant schema extraction call(s)).",
        )
        return chunk_to_schemas

    def run_extraction(
        self,
        documents: List[Dict[str, Any]],
        target_schemas: List[SchemaMeta],
        dry_run: bool = False,
    ) -> None:
        """
        Executes Two-Pass Hardware-Batched Extraction across all corpus documents and target schemas:
          1. Chunk documents & filter already-completed `(doc_id, schema)` pairs.
          2. Run Pass 3a (Fast Triage) across pending chunks.
          3. Run Pass 3b (Targeted Single-Schema Deep Extraction) batched per schema with
             streaming chunk-level disk caching.
          4. Run Pass 3c (Exact Verbatim Consolidation) via `merge_pydantic_instances` and
             write `<doc_id>__<module>__<Class>.json`.
        """
        # Prepare per-document chunks and pending schemas
        doc_chunks_map: Dict[str, List[str]] = {}
        doc_pending_schemas: Dict[str, List[SchemaMeta]] = {}
        all_chunk_records: List[Dict[str, Any]] = []

        skipped_doc_schemas = 0
        for doc in documents:
            doc_id = str(doc["id"])
            text = str(doc["text"])
            pending = [
                s for s in target_schemas
                if not self.tracker.is_schema_completed(doc_id, s)
            ]
            skipped_doc_schemas += len(target_schemas) - len(pending)
            if not pending:
                continue

            chunks = self._safe_chunk_document(text)
            if not chunks:
                for s in pending:
                    self.tracker.save_extraction(doc_id, s, [])
                continue

            doc_chunks_map[doc_id] = chunks
            doc_pending_schemas[doc_id] = pending

            for chunk_idx, ch_text in enumerate(chunks):
                all_chunk_records.append(
                    {
                        "id": f"{doc_id}::chunk_{chunk_idx}",
                        "text": ch_text,
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "pending_schemas": pending,
                    }
                )

        if skipped_doc_schemas > 0:
            log_checkpoint(
                "Phase 3: Resume Check",
                f"Skipped {skipped_doc_schemas} already-completed (document, schema) pair(s) on disk.",
            )

        if not all_chunk_records:
            log_checkpoint("Phase 3: Extraction", "All documents and schemas are already completed.")
            return

        # Pass 3a: Fast Relevance Triage
        chunk_to_schemas = self._run_triage_pass(
            chunk_records=all_chunk_records,
            target_schemas=target_schemas,
            dry_run=dry_run,
        )

        # Group required (doc_id, chunk_idx, chunk_text) items by SchemaMeta
        schema_by_qname: Dict[str, SchemaMeta] = {s.qualified_name: s for s in target_schemas}
        chunk_text_lookup: Dict[Tuple[str, int], str] = {
            (rec["doc_id"], rec["chunk_idx"]): rec["text"]
            for rec in all_chunk_records
        }

        # Store chunk-level BaseModel results in memory: (doc_id, qualified_name) -> List[BaseModel]
        extracted_chunk_models: Dict[Tuple[str, str], List[BaseModel]] = defaultdict(list)
        schema_work_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for (doc_id, chunk_idx), selected_schemas in chunk_to_schemas.items():
            ch_text = chunk_text_lookup[(doc_id, chunk_idx)]
            for s_meta in selected_schemas:
                if self.tracker.is_chunk_completed(doc_id, s_meta, chunk_idx):
                    cached_inst = self.tracker.load_chunk_result(doc_id, s_meta, chunk_idx)
                    if cached_inst is not None and has_extracted_content(cached_inst):
                        extracted_chunk_models[(doc_id, s_meta.qualified_name)].append(cached_inst)
                else:
                    schema_work_queues[s_meta.qualified_name].append(
                        {
                            "id": f"{doc_id}::chunk_{chunk_idx}::{s_meta.qualified_name}",
                            "text": ch_text,
                            "doc_id": doc_id,
                            "chunk_idx": chunk_idx,
                            "qualified_name": s_meta.qualified_name,
                        }
                    )

        # Pass 3b: Targeted Single-Schema Deep Extraction
        total_schemas_with_work = len(schema_work_queues)
        for s_idx, (qname, work_items) in enumerate(schema_work_queues.items(), start=1):
            s_meta = schema_by_qname[qname]
            sys_prompt = build_extraction_prompt(s_meta.schema_cls)
            instruction = (
                f"Extract all information matching `{s_meta.name}` from the document excerpt above. "
                f"Use exact verbatim mentions and set any absent fields to null."
            )

            if work_items:
                self._maybe_save_payload(
                    s_meta.name,
                    sys_prompt,
                    f"Document Excerpt:\n{work_items[0]['text']}\n\nInstruction:\n{instruction}",
                )

            log_checkpoint(
                "Phase 3b: Deep Extraction",
                f"[{s_idx}/{total_schemas_with_work}] Extracting schema '{s_meta.qualified_name}' "
                f"across {len(work_items)} relevant chunk(s)...",
            )

            work_meta_by_id = {w["id"]: w for w in work_items}
            prism_inputs = [{"id": w["id"], "text": w["text"]} for w in work_items]

            def _on_chunk_done(
                annotated_chunk: Dict[str, Any],
                _meta: SchemaMeta = s_meta,
                _lookup: Dict[str, Dict[str, Any]] = work_meta_by_id,
            ) -> None:
                rec_id = str(annotated_chunk.get("record_id") or annotated_chunk.get("id") or "")
                w_info = _lookup.get(rec_id, {})
                c_doc_id = w_info.get("doc_id")
                c_idx = w_info.get("chunk_idx")
                parsed_items = annotated_chunk.get("parsed_items", [])
                if c_doc_id is None or c_idx is None:
                    return
                valid_models = [
                    m for m in parsed_items
                    if isinstance(m, BaseModel) and has_extracted_content(m)
                ]
                merged_chunk_inst = (
                    merge_pydantic_instances(valid_models, _meta.schema_cls)
                    if valid_models
                    else None
                )
                self.tracker.save_chunk_result(c_doc_id, _meta, c_idx, merged_chunk_inst)
                if merged_chunk_inst is not None:
                    extracted_chunk_models[(c_doc_id, _meta.qualified_name)].append(merged_chunk_inst)

            self.prism.run(
                input_data=prism_inputs,
                id_column="id",
                text_column="text",
                task="extract",
                schema_cls=s_meta.schema_cls,
                system_prompt=sys_prompt,
                instruction=instruction,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                batch_size=self.batch_size,
                chunk=False,
                as_array=False,
                constrain_json=self.constrain_json,
                dedup=False,
                dry_run=dry_run,
                on_chunk_processed=_on_chunk_done,
            )

        if dry_run:
            log_checkpoint("Phase 3: Dry Run", "Dry run complete; skipping consolidation and file writes.")
            return

        # Pass 3c: Exact Verbatim Chunk Consolidation per (doc_id, SchemaMeta)
        consolidated_count = 0
        non_empty_count = 0
        for doc_id, pending_list in doc_pending_schemas.items():
            for s_meta in pending_list:
                chunk_instances = extracted_chunk_models.get((doc_id, s_meta.qualified_name), [])
                merged_doc_model = merge_pydantic_instances(chunk_instances, s_meta.schema_cls)

                if merged_doc_model is not None and has_extracted_content(merged_doc_model):
                    serialized = merged_doc_model.model_dump(mode="json")
                    self.tracker.save_extraction(doc_id, s_meta, serialized)
                    non_empty_count += 1
                else:
                    self.tracker.save_extraction(doc_id, s_meta, [])
                consolidated_count += 1

        log_checkpoint(
            "Phase 3c: Consolidation",
            f"Consolidated {consolidated_count} (document, schema) output(s) "
            f"({non_empty_count} populated with verbatim data).",
        )

    def run(
        self,
        input_source: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]] = INPUT_PATH,
        id_column: str = ID_COLUMN,
        text_column: str = TEXT_COLUMN,
        limit: Optional[int] = DOCUMENT_LIMIT,
        skip_extraction: bool = False,
        skip_aggregation: bool = False,
        dry_run: bool = DRY_RUN,
    ) -> Dict[str, Any]:
        """
        Runs the complete Stage 5 pipeline:
          - Phase 1: Schema DAG Discovery
          - Phase 2: Corpus Document Loading
          - Phase 3: Batched Triage, Extraction & Verbatim Consolidation
          - Phase 4: Relational & Consolidated Wide-Table Aggregation
        """
        source_texts: Dict[str, str] = {}
        try:
            documents = load_input_documents(
                input_source=input_source,
                id_column=id_column,
                text_column=text_column,
                limit=limit,
            )
            source_texts = {str(d["id"]): str(d["text"]) for d in documents}
            log_checkpoint("Phase 2: Corpus Loader", f"Loaded {len(documents)} document(s) from '{input_source}'.")
        except Exception as e:
            if not skip_extraction:
                raise
            logger.warning(f"Could not load source documents for aggregation ({e}); proceeding without source_text.")
            documents = []

        if not skip_extraction:
            target_schemas = self.resolve_target_schemas()
            try:
                self.run_extraction(
                    documents=documents,
                    target_schemas=target_schemas,
                    dry_run=dry_run,
                )
            finally:
                if self.manage_vram and (
                    getattr(self.prism, "engine", None) is not None
                    or getattr(self.prism, "model", None) is not None
                ):
                    self.prism.engine = None
                    self.prism.model = None
                    self.prism.tokenizer = None
                    clear_device_memory()
                    log_checkpoint("VRAM Management", "Unloaded LLM and cleared device memory.")

        agg_summary: Dict[str, Any] = {}
        if not skip_aggregation and not dry_run:
            aggregator = DataAggregator(
                extracts_dir=self.extracts_dir,
                tables_dir=self.tables_dir,
                generate_csv=GENERATE_CSV,
                generate_sqlite=GENERATE_SQLITE,
                generate_parquet=GENERATE_PARQUET,
                db_filename=DB_FILENAME,
                multi_table_relational=MULTI_TABLE_RELATIONAL,
                consolidated_wide_table=CONSOLIDATED_WIDE_TABLE,
                wide_table_name=WIDE_TABLE_NAME,
                include_source_text=INCLUDE_SOURCE_TEXT,
                drop_empty_records=DROP_EMPTY_RECORDS,
                drop_empty_tables=DROP_EMPTY_TABLES,
            )
            agg_summary = aggregator.run(source_texts=source_texts)

        log_checkpoint("Stage 5 Complete", f"Outputs saved under '{self.stage5_dir}'.")
        return agg_summary


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 5: Schema-Driven Extraction & Relational/Wide-Table Aggregation (Powered by PrismRunner)"
    )
    parser.add_argument(
        "--schema-path",
        type=str,
        default=SCHEMA_PATH,
        help="Path to a single .py schema file (e.g. master_ontology.py) OR a directory of .py schema files.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=TARGET_CLASS_NAME,
        help="Optional specific Pydantic BaseModel class name to extract.",
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        choices=["top_level", "child", "all"],
        default=EXTRACT_MODE,
        help="Which schema hierarchy tier to extract ('top_level' recommended).",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=INPUT_PATH,
        help="Path to input corpus (.parquet, .csv, .json, .jsonl, or directory of .txt files).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=ID_COLUMN,
        help="Column name for document IDs.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=TEXT_COLUMN,
        help="Column name for document text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DOCUMENT_LIMIT,
        help="Maximum number of documents to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model ID or local path for PrismRunner.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Compute device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default=str(BATCH_SIZE),
        help="Inference batch size ('auto' or integer).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=OUTPUTS_DIR,
        help="Base outputs directory (writes to <outputs-dir>/05_extraction).",
    )
    parser.add_argument(
        "--no-triage",
        action="store_true",
        help="Disable fast multi-schema relevance triage and run all schemas on every chunk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extraction files instead of resuming.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction and only run relational/wide-table aggregation on existing extracts.",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Run LLM extraction only and skip table aggregation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Preview payloads and pipeline setup without loading or executing the LLM.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live LLM execution even if DRY_RUN is True.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size_val: Union[int, str] = (
        int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    )
    is_dry_run = False if args.live else bool(args.dry_run)
    resume_mode = "overwrite" if args.overwrite else RESUME_MODE

    pipeline = SemanticPrismStage5Pipeline(
        model_name=args.model,
        device=args.device,
        batch_size=batch_size_val,
        enable_triage=not args.no_triage and ENABLE_TRIAGE,
        schema_path=args.schema_path,
        target_class_name=args.target_class,
        extract_mode=args.extract_mode,
        outputs_dir=args.outputs_dir,
        resume_mode=resume_mode,
    )
    pipeline.run(
        input_source=args.input_path,
        id_column=args.id_column,
        text_column=args.text_column,
        limit=args.limit,
        skip_extraction=args.skip_extraction,
        skip_aggregation=args.skip_aggregation,
        dry_run=is_dry_run,
    )


if __name__ == "__main__":
    main()

