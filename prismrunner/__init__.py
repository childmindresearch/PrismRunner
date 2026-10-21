"""PrismRunner: Lightweight, local-first LLM inference engine."""

from prismrunner.pipeline import PrismPipeline
from prismrunner.loaders.dataset import DatasetLoader, Record
from prismrunner.chunking.chunker import TokenAwareChunker, ChunkRecord
from prismrunner.schemas.compiler import compile_schema_to_prompt, CompiledPromptSchema
from prismrunner.schemas.parser import UniversalOutputParser, ParseResult, is_phantom_object
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem
from prismrunner.stitcher.stitcher import DeterministicStitcher

__version__ = "0.1.0"

__all__ = [
    "PrismPipeline",
    "DatasetLoader",
    "Record",
    "TokenAwareChunker",
    "ChunkRecord",
    "compile_schema_to_prompt",
    "CompiledPromptSchema",
    "UniversalOutputParser",
    "ParseResult",
    "is_phantom_object",
    "ClassificationItem",
    "EntityItem",
    "SummaryItem",
    "DeterministicStitcher",
]
