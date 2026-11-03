"""PrismRunner: Lightweight, local-first LLM inference engine."""

from prismrunner.pipeline import PrismPipeline
from prismrunner.mapping_pipeline import SchemaMappingPipeline, SchemaMapPipeline
from prismrunner.loaders.dataset import DatasetLoader, Record
from prismrunner.chunking.chunker import TokenAwareChunker, ChunkRecord
from prismrunner.schemas.compiler import compile_schema_to_prompt, CompiledPromptSchema, load_schema_from_string
from prismrunner.schemas.parser import UniversalOutputParser, ParseResult, is_phantom_object, sanitize_json_output
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem
from prismrunner.stitcher.stitcher import DeterministicStitcher
from prismrunner.catalog.registry import SchemaCatalog, SchemaMeta
from prismrunner.triage.filter import SchemaTriageFilter, TriageResult
from prismrunner.tracker.checkpoint import CheckpointTracker
from prismrunner.aggregator.relational import RelationalNormalizer, has_extracted_content
from prismrunner.aggregator.sqlite_export import SQLiteExporter
from prismrunner.aggregator.wide_table import WideTableBuilder
from prismrunner.prompt import PromptTemplate
from prismrunner.engine.device import resolve_device_and_dtype
from prismrunner.engine.model import init_model_and_tokenizer
from prismrunner.engine.inference import create_length_buckets, restore_original_order, EXTRACTION_GEN_CONFIG

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
    "sanitize_json_output",
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
    "PromptTemplate",
    "resolve_device_and_dtype",
    "init_model_and_tokenizer",
    "create_length_buckets",
    "restore_original_order",
    "EXTRACTION_GEN_CONFIG",
]

