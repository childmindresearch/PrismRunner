"""Schema relevance triage filter for skipping irrelevant schemas per chunk."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
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

    def __init__(
        self,
        catalog: SchemaCatalog,
        prompt_file: Optional[Union[str, Path]] = "prompts/triage.yaml",
        prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ):
        self.catalog = catalog
        self.prompt_template = prompt_template
        self.system_prompt_override = system_prompt

        t_path = Path(prompt_file) if prompt_file else Path("prompts/triage.yaml")
        if not t_path.exists():
            pkg_fallback = Path(__file__).resolve().parent.parent.parent / t_path
            if pkg_fallback.exists():
                t_path = pkg_fallback
        self.prompt_file = t_path
        self.compiled_schema = compile_schema_to_prompt(TriageResult)

    def build_triage_prompt(self, text: str) -> str:
        """Construct a lightweight triage classification prompt."""
        catalog_summary = self.catalog.build_triage_summary()
        schema_block = self.compiled_schema.render_prompt_block(as_array=False)

        tmpl: Dict[str, Any] = {}
        if self.prompt_template is not None:
            if isinstance(self.prompt_template, dict):
                tmpl = dict(self.prompt_template)
            elif isinstance(self.prompt_template, str):
                tmpl = {"prompt": self.prompt_template, "variables": {}}
        elif self.prompt_file.exists():
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                tmpl = yaml.safe_load(f) or {}

        if tmpl:
            raw_prompt = tmpl.get("prompt", "")
            variables = dict(tmpl.get("variables", {}))
            variables["instruction"] = tmpl.get("instruction", "")
            variables["catalog_summary"] = catalog_summary
            variables["schema_block"] = schema_block

            rendered = raw_prompt
            for k, v in variables.items():
                rendered = rendered.replace(f"{{{k}}}", str(v))
            rendered = rendered.replace("{{ text }}", text).replace("{{ text}}", text)
            return rendered

        # Built-in fallback if no template configured
        return (
            "You are an expert data triage classifier. Your task is to analyze the text excerpt and "
            "identify which of the candidate schemas have concrete, factual evidence present.\n\n"
            f"Candidate Schemas Catalog:\n{catalog_summary}\n\n"
            f'Source Content Excerpt:\n"""\n{text}\n"""\n\n'
            f"{schema_block}\n"
            "Rule: If no factual evidence exists for any schema in this excerpt, return an empty list."
        )

    def triage_batch(
        self,
        chunks: List[ChunkRecord],
        engine: BatchInferenceEngine,
        max_new_tokens: int = 256,
    ) -> Dict[str, List[SchemaMeta]]:
        """Run vectorized batch triage over chunks to determine active schemas per chunk."""
        if not chunks or len(self.catalog) == 0:
            return {}

        # Build prompts
        sys_prompt = self.system_prompt_override or "You are a fast, factual schema triage filter. Output valid JSON only."
        if not self.system_prompt_override and self.prompt_file.exists():
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                tmpl = yaml.safe_load(f) or {}
            sys_prompt = tmpl.get("system_prompt") or sys_prompt

        prompts: List[str] = []
        for ch in chunks:
            raw_prompt = self.build_triage_prompt(ch.text)
            messages = engine.format_chatml_messages(
                user_prompt=raw_prompt,
                system_prompt=sys_prompt,
            )
            formatted = engine.render_chat_prompt(messages)
            prompts.append(formatted)

        # Batch inference
        completions = engine.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=0.0)

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
