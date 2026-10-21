"""Schema relevance triage filter for skipping irrelevant schemas per chunk."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from prismrunner.catalog.registry import SchemaCatalog, SchemaMeta
from prismrunner.chunking.chunker import ChunkRecord
from prismrunner.engine.inference import BatchInferenceEngine
from prismrunner.schemas.compiler import compile_schema_to_prompt
from prismrunner.schemas.parser import UniversalOutputParser

logger = logging.getLogger(__name__)


class TriageResult(BaseModel):
    """Output model for fast schema relevance triage classification."""

    relevant_schemas: List[str] = Field(
        default_factory=list,
        description="Exact schema class names that have factual evidence present in the text excerpt.",
    )


class SchemaTriageFilter:
    """Pre-screens dataset chunks against a schema catalog to eliminate redundant LLM calls."""

    def __init__(self, catalog: SchemaCatalog):
        self.catalog = catalog
        self.compiled_schema = compile_schema_to_prompt(TriageResult)

    def build_triage_prompt(self, text: str) -> str:
        """Construct a lightweight triage classification prompt."""
        catalog_summary = self.catalog.build_triage_summary()

        return (
            "You are an expert data triage classifier. Your task is to analyze the text excerpt and "
            "identify which of the candidate schemas have concrete, factual evidence present.\n\n"
            f"Candidate Schemas Catalog:\n{catalog_summary}\n\n"
            f'Source Content Excerpt:\n"""\n{text}\n"""\n\n'
            f"{self.compiled_schema.render_prompt_block(as_array=False)}\n"
            "Rule: If no factual evidence exists for any schema in this excerpt, return an empty list."
        )

    def triage_batch(
        self,
        chunks: List[ChunkRecord],
        engine: BatchInferenceEngine,
    ) -> Dict[str, List[SchemaMeta]]:
        """Run vectorized batch triage over chunks to determine active schemas per chunk."""
        if not chunks or len(self.catalog) == 0:
            return {}

        # Build prompts
        prompts: List[str] = []
        for ch in chunks:
            raw_prompt = self.build_triage_prompt(ch.text)
            messages = engine.format_chatml_messages(
                user_prompt=raw_prompt,
                system_prompt="You are a fast, factual schema triage filter. Output valid JSON only.",
            )
            formatted = engine.render_chat_prompt(messages)
            prompts.append(formatted)

        # Batch inference
        completions = engine.generate_batch(prompts, max_new_tokens=128, temperature=0.0)

        chunk_active_schemas: Dict[str, List[SchemaMeta]] = {}

        for ch, comp in zip(chunks, completions):
            chunk_key = f"{ch.record_id}_{ch.chunk_index}"
            parse_res = UniversalOutputParser.parse(comp, model_cls=TriageResult)

            active_metas: List[SchemaMeta] = []
            if parse_res.success and parse_res.items:
                res_obj: TriageResult = parse_res.items[0]
                for s_name in res_obj.relevant_schemas:
                    meta = self.catalog.get(s_name.strip())
                    if meta:
                        active_metas.append(meta)
                    else:
                        logger.debug(f"Triage suggested unregistered schema: {s_name}")
            else:
                # Fallback: check if raw text contains exact registered schema names
                for meta in self.catalog.list_schemas():
                    if meta.name.lower() in comp.lower():
                        active_metas.append(meta)

            chunk_active_schemas[chunk_key] = active_metas

        total_possible = len(chunks) * len(self.catalog)
        total_selected = sum(len(s) for s in chunk_active_schemas.values())
        saved_pct = round((1.0 - (total_selected / max(1, total_possible))) * 100, 1)

        logger.info(
            f"Triage complete: selected {total_selected}/{total_possible} schema-chunk pairs "
            f"({saved_pct}% LLM calls saved via triage pre-screening)."
        )
        return chunk_active_schemas
