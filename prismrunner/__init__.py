"""PrismRunner: Lightweight, local-first LLM inference engine."""

from prismrunner.pipeline import PrismPipeline
from prismrunner.mapping_pipeline import SchemaMappingPipeline, SchemaMapPipeline
from prismrunner.loaders.dataset import DatasetLoader, Record
from prismrunner.chunking.chunker import TokenAwareChunker, ChunkRecord
from prismrunner.schemas.compiler import compile_schema_to_prompt, CompiledPromptSchema, load_schema_from_string
from prismrunner.schemas.parser import UniversalOutputParser, ParseResult, is_phantom_object
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem
from prismrunner.stitcher.stitcher import DeterministicStitcher
from prismrunner.catalog.registry import SchemaCatalog, SchemaMeta
from prismrunner.triage.filter import SchemaTriageFilter, TriageResult
from prismrunner.tracker.checkpoint import CheckpointTracker
from prismrunner.aggregator.relational import RelationalNormalizer, has_extracted_content
from prismrunner.aggregator.sqlite_export import SQLiteExporter
from prismrunner.aggregator.wide_table import WideTableBuilder

__version__ = "0.1.0"

__all__ = [
    "PrismPipeline",
    "SchemaMappingPipeline",
    "SchemaMapPipeline",
    "DatasetLoader",
    "Record",
    "TokenAwareChunker",
    "ChunkRecord",
    "compile_schema_to_prompt",
    "CompiledPromptSchema",
    "load_schema_from_string",
    "UniversalOutputParser",
    "ParseResult",
    "is_phantom_object",
    "ClassificationItem",
    "EntityItem",
    "SummaryItem",
    "DeterministicStitcher",
    "SchemaCatalog",
    "SchemaMeta",
    "SchemaTriageFilter",
    "TriageResult",
    "CheckpointTracker",
    "RelationalNormalizer",
    "has_extracted_content",
    "SQLiteExporter",
    "WideTableBuilder",
]

