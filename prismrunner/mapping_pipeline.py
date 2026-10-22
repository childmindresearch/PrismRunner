"""SchemaMappingPipeline: End-to-end multi-schema triage, batch extraction, and relational aggregation."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import yaml
from rich.console import Console

from prismrunner.pipeline import PrismPipeline
from prismrunner.loaders.dataset import DatasetLoader
from prismrunner.chunking.chunker import TokenAwareChunker, ChunkRecord
from prismrunner.catalog.registry import SchemaCatalog, SchemaMeta
from prismrunner.triage.filter import SchemaTriageFilter
from prismrunner.tracker.checkpoint import CheckpointTracker
from prismrunner.schemas.compiler import compile_schema_to_prompt
from prismrunner.schemas.parser import UniversalOutputParser
from prismrunner.aggregator.relational import RelationalNormalizer
from prismrunner.aggregator.sqlite_export import SQLiteExporter
from prismrunner.aggregator.wide_table import WideTableBuilder

logger = logging.getLogger(__name__)
console = Console()


class SchemaMappingPipeline:
    """Executes multi-schema extraction with fast triage and relational aggregation."""

    def __init__(
        self,
        schemas_dir: Optional[Union[str, Path]] = None,
        catalog: Optional[SchemaCatalog] = None,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_pipeline = PrismPipeline(config_path=config_path, config=config)
        if catalog is not None:
            self.catalog = catalog
        elif schemas_dir is not None:
            self.catalog = SchemaCatalog.from_directory(schemas_dir)
        else:
            self.catalog = SchemaCatalog()

    def run(
        self,
        input_data: Union[str, Path, Any],
        output_dir: Union[str, Path] = "outputs",
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        use_triage: bool = True,
        resume: bool = True,
        export_formats: Optional[List[str]] = None,  # "sqlite", "csv", "wide"
        dry_run: bool = False,
        chunk_size: int = 1024,
        chunk_overlap: int = 64,
        chunk_strategy: str = "sentence",
        triage_prompt_file: Optional[Union[str, Path]] = None,
        triage_prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
        triage_system_prompt: Optional[str] = None,
        triage_max_tokens: int = 256,
        extract_prompt_file: Optional[Union[str, Path]] = None,
        extract_prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        id_column_name: str = "source_id",
        sqlite_db_name: str = "extracted_database.db",
        wide_table_name: str = "consolidated_wide_table",
        relational_csv_dir: str = "relational_csv",
        wide_list_separator: str = "; ",
    ) -> Dict[str, Any]:
        """Execute full multi-schema triage, extraction, and relational aggregation workflow."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Dataset dynamically
        records = DatasetLoader.load(
            file_path_or_data=input_data,
            text_column=text_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            exclude_columns=exclude_columns,
            limit=limit,
        )

        if not records:
            logger.warning("No records to process.")
            return {}

        # 2. Check catalog
        if len(self.catalog) == 0:
            raise ValueError("SchemaCatalog contains 0 registered schemas. Please specify valid schemas_dir or catalog.")

        console.print(f"[bold cyan]PrismRunner SchemaMap Pipeline[/bold cyan]")
        console.print(f"Loaded {len(records)} documents and {len(self.catalog)} candidate schemas.")

        # 3. Chunking
        chunker = TokenAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        )
        chunks: List[ChunkRecord] = chunker.chunk_dataset(records)
        console.print(f"Segmented into {len(chunks)} token-budgeted chunks.")

        # 4. Checkpoint Tracker
        checkpoint_file = out_dir / ".checkpoint_state.json"
        tracker = CheckpointTracker(checkpoint_file) if resume else None

        # Dry run inspection
        if dry_run:
            console.print("[bold yellow]*** DRY RUN MODE ***[/bold yellow]")
            console.print(f"Candidate Schemas ({len(self.catalog)}):")
            for meta in self.catalog.list_schemas():
                console.print(f"  • [bold]{meta.name}[/bold]: {meta.description}")
            console.print(f"\nTotal potential extraction passes without triage: {len(chunks) * len(self.catalog)}")
            return {"total_records": len(records), "total_chunks": len(chunks), "schemas": len(self.catalog)}

        # 5. Initialize Engine
        engine = self.base_pipeline.init_engine(
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            generation_kwargs=generation_kwargs,
        )

        # 6. Schema Relevance Triage
        scheduled_tasks: List[Dict[str, Any]] = []
        source_texts: Dict[str, str] = {r.id: r.text for r in records}

        if use_triage:
            console.print("\n[bold blue]Running Vectorized ChatML Batch Triage...[/bold blue]")
            triage_filter = SchemaTriageFilter(
                self.catalog,
                prompt_file=triage_prompt_file,
                prompt_template=triage_prompt_template,
                system_prompt=triage_system_prompt,
            )
            chunk_active_map = triage_filter.triage_batch(chunks, engine, max_new_tokens=triage_max_tokens)

            for ch in chunks:
                chunk_key = f"{ch.record_id}_{ch.chunk_index}"
                active_metas = chunk_active_map.get(chunk_key, [])
                for meta in active_metas:
                    if tracker and tracker.is_completed(ch.record_id, meta.name, ch.chunk_index):
                        continue
                    scheduled_tasks.append({
                        "chunk": ch,
                        "schema_meta": meta,
                    })
        else:
            # Exhaustive extraction across all schemas
            for ch in chunks:
                for meta in self.catalog.list_schemas():
                    if tracker and tracker.is_completed(ch.record_id, meta.name, ch.chunk_index):
                        continue
                    scheduled_tasks.append({
                        "chunk": ch,
                        "schema_meta": meta,
                    })

        console.print(f"Scheduled {len(scheduled_tasks)} targeted single-schema extraction passes.")

        # 7. Targeted Single-Schema Batch Extraction
        extraction_records: List[Dict[str, Any]] = []

        # If resume is active, pull cached results first
        if tracker:
            for ch in chunks:
                for meta in self.catalog.list_schemas():
                    cached_val = tracker.get_cached_result(ch.record_id, meta.name, ch.chunk_index)
                    if cached_val is not None:
                        extraction_records.append({
                            "source_id": ch.record_id,
                            "schema_name": meta.name,
                            "data": cached_val,
                            "chunk_index": ch.chunk_index,
                        })

        if scheduled_tasks:
            # Group tasks by schema for efficiency and clean compilation
            tasks_by_schema: Dict[str, List[Dict[str, Any]]] = {}
            for t in scheduled_tasks:
                tasks_by_schema.setdefault(t["schema_meta"].name, []).append(t)

            # Determine extraction template
            extract_tmpl: Dict[str, Any] = {}
            if extract_prompt_template is not None:
                if isinstance(extract_prompt_template, dict):
                    extract_tmpl = dict(extract_prompt_template)
                elif isinstance(extract_prompt_template, str):
                    extract_tmpl = {
                        "version": "custom",
                        "instruction": "Extract all factual instances of {schema_name}.",
                        "variables": {},
                        "prompt": extract_prompt_template,
                    }
            else:
                extract_tmpl_path = Path(extract_prompt_file) if extract_prompt_file else Path("prompts/schema_extract.yaml")
                if not extract_tmpl_path.exists():
                    pkg_fallback = Path(__file__).resolve().parent.parent / extract_tmpl_path
                    if pkg_fallback.exists():
                        extract_tmpl_path = pkg_fallback
                if extract_tmpl_path.exists():
                    with open(extract_tmpl_path, "r", encoding="utf-8") as f:
                        extract_tmpl = yaml.safe_load(f) or {}

            for schema_name, tasks in tasks_by_schema.items():
                meta = tasks[0]["schema_meta"]
                compiled_schema = compile_schema_to_prompt(meta.schema_cls)
                schema_directive = compiled_schema.render_prompt_block(as_array=True)

                payloads: List[Dict[str, Any]] = []
                for t in tasks:
                    ch: ChunkRecord = t["chunk"]
                    if extract_tmpl:
                        raw_tmpl = extract_tmpl.get("prompt", "")
                        vars_dict = dict(extract_tmpl.get("variables", {}))
                        if variables:
                            vars_dict.update(variables)
                        raw_inst = extract_tmpl.get("instruction", "Extract all factual instances of {schema_name}.")
                        vars_dict["instruction"] = raw_inst.replace("{schema_name}", meta.name)
                        vars_dict["schema_name"] = meta.name
                        vars_dict["schema_description"] = meta.description
                        vars_dict["schema_block"] = schema_directive

                        prompt_str = raw_tmpl
                        for vk, vv in vars_dict.items():
                            prompt_str = prompt_str.replace(f"{{{vk}}}", str(vv))
                        prompt_str = prompt_str.replace("{{ text }}", ch.text).replace("{{ text}}", ch.text)
                    else:
                        prompt_str = (
                            f"Extract all factual instances of {meta.name} from the following text.\n"
                            f"Scope: {meta.description}\n\n"
                            f'Source Content:\n"""\n{ch.text}\n"""\n\n'
                            f"{schema_directive}"
                        )

                    payloads.append({
                        "record_id": ch.record_id,
                        "chunk_index": ch.chunk_index,
                        "schema_name": meta.name,
                        "prompt": prompt_str,
                    })

                sys_prompt = (
                    system_prompt
                    or extract_tmpl.get("system_prompt")
                    or "You are an exact, verbatim data extraction engine. Output valid JSON array only."
                )
                raw_results = engine.run_all(
                    items=payloads,
                    system_prompt=sys_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    prompt_key="prompt",
                    **(generation_kwargs or {}),
                )

                for res in raw_results:
                    parsed = UniversalOutputParser.parse(res.get("raw_output", ""), model_cls=meta.schema_cls)
                    valid_data = [item.model_dump() for item in parsed.items] if parsed.success else []

                    if tracker:
                        tracker.mark_completed(
                            record_id=res["record_id"],
                            schema_name=meta.name,
                            chunk_index=res["chunk_index"],
                            result_payload=valid_data,
                        )

                    extraction_records.append({
                        "source_id": res["record_id"],
                        "schema_name": meta.name,
                        "data": valid_data,
                        "chunk_index": res["chunk_index"],
                    })

            if tracker:
                tracker.save()

        # 8. Relational Normalization
        console.print("\n[bold green]Normalizing into Relational Tables...[/bold green]")
        relational_tables = RelationalNormalizer.normalize_dataset(
            extraction_records,
            id_column_name=id_column_name,
            child_separator=wide_list_separator,
        )

        # 9. Output Exports
        formats = export_formats or ["sqlite", "csv", "wide"]
        outputs_summary: Dict[str, Any] = {
            "tables_generated": list(relational_tables.keys()),
            "total_records": len(records),
        }

        # Export Relational CSVs
        if "csv" in formats:
            csv_dir = out_dir / relational_csv_dir
            csv_dir.mkdir(parents=True, exist_ok=True)
            for t_name, rows in relational_tables.items():
                df = pd.DataFrame(rows)
                df.to_csv(csv_dir / f"{t_name}.csv", index=False)
            outputs_summary["relational_csv_dir"] = str(csv_dir)

        # Export SQLite DB
        if "sqlite" in formats:
            db_path = out_dir / sqlite_db_name
            SQLiteExporter.export(relational_tables, db_path, id_column_name=id_column_name)
            outputs_summary["sqlite_db"] = str(db_path)

        # Export Consolidated Wide Table
        if "wide" in formats:
            wide_base = out_dir / wide_table_name
            wide_parquet = wide_base.with_suffix(".parquet") if not str(wide_base).endswith(".parquet") else wide_base
            wide_csv = wide_base.with_suffix(".csv") if not str(wide_base).endswith(".csv") else wide_base

            WideTableBuilder.export(
                relational_tables,
                wide_parquet,
                source_texts=source_texts,
                output_format="parquet",
                list_join_char=wide_list_separator,
                id_column_name=id_column_name,
            )
            WideTableBuilder.export(
                relational_tables,
                wide_csv,
                source_texts=source_texts,
                output_format="csv",
                list_join_char=wide_list_separator,
                id_column_name=id_column_name,
            )
            outputs_summary["wide_table"] = str(wide_parquet)

        console.print(f"[bold green]SchemaMap Extraction & Relational Aggregation Complete![/bold green]")
        console.print(f"Generated {len(relational_tables)} relational tables across {len(records)} documents.")
        return outputs_summary


# Convenient alias
SchemaMapPipeline = SchemaMappingPipeline

