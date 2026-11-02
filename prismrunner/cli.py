"""PrismRunner Command Line Interface."""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path when cli.py is executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
from rich.console import Console
from rich.table import Table

from prismrunner.pipeline import PrismPipeline
from prismrunner.engine.device import get_optimal_device, get_memory_info
from prismrunner.engine.context import generate_hardware_advisory

console = Console()
logger = logging.getLogger("prismrunner")


def _parse_vars(var_tuples: Tuple[str, ...]) -> dict:
    """Parse dynamic --var KEY=VALUE CLI options into a dictionary."""
    parsed = {}
    for item in var_tuples:
        if "=" in item:
            k, v = item.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def apply_options(options):
    """Apply a list of Click option decorators in reverse order."""
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


SHARED_ENGINE_OPTIONS = [
    click.option("--model", "-m", type=str, help="Hugging Face model repository ID or local path."),
    click.option("--device", "-d", type=str, help="Compute device (auto, mps, cuda, cpu)."),
    click.option("--batch-size", "-b", type=int, default=4, help="Inference batch size."),
    click.option("--max-tokens", type=int, default=512, help="Maximum new tokens to generate."),
    click.option("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic)."),
    click.option("--top-p", type=float, default=None, help="Top-p nucleus sampling probability threshold."),
    click.option("--top-k", type=int, default=None, help="Top-k sampling threshold."),
    click.option("--repetition-penalty", type=float, default=None, help="Repetition penalty parameter."),
    click.option("--cache-dir", type=click.Path(), help="Directory to store/load cached model weights."),
    click.option("--local-files-only", is_flag=True, help="Only load model from local cache; do not check Hugging Face online."),
    click.option("--config", "-c", type=click.Path(exists=True), help="Pipeline config YAML path."),
]

SHARED_IO_OPTIONS = [
    click.option("--column", "--text-column", "text_column", type=str, help="Name of column containing input text."),
    click.option("--id-column", type=str, help="Name of column containing unique record ID."),
    click.option("--metadata-columns", type=str, help="Comma-separated list of metadata columns to include."),
    click.option("--exclude-columns", type=str, help="Comma-separated list of columns to exclude from metadata."),
    click.option("--limit", "-l", type=int, help="Maximum number of records to process."),
    click.option("--dry-run", is_flag=True, help="Render prompt and inspect pipeline without loading model."),
]

SHARED_CHUNKING_OPTIONS = [
    click.option("--chunk-size", type=int, default=1024, help="Tokens per chunk window."),
    click.option("--chunk-overlap", type=int, default=64, help="Tokens overlap between chunks."),
    click.option("--chunk-strategy", type=click.Choice(["sentence", "token"], case_sensitive=False), default="sentence", help="Boundary strategy."),
]

TASK_IO_OPTIONS = [
    click.option("--input-path", "-i", type=click.Path(exists=False), help="Path to input dataset (.parquet, .csv, .json, .txt)."),
    click.option("--input-text", "-t", type=str, help="Single raw text string to process directly."),
    click.option("--output-path", "-o", type=click.Path(), help="Output file destination."),
    click.option("--output-format", "-f", type=click.Choice(["json", "csv", "parquet"], case_sensitive=False), default="json", help="Output format."),
    click.option("--json", "json_stdout", is_flag=True, help="Output raw JSON to stdout."),
]

PROMPT_OPTIONS = [
    click.option("--schema", "-s", "schema_spec", type=str, help="Custom schema class spec: 'path/to/file.py:ClassName'."),
    click.option("--prompt-file", type=click.Path(exists=True), help="Path to custom prompt template YAML."),
    click.option("--instruction", type=str, help="Instruction override for prompt template."),
    click.option("--system-prompt", type=str, help="System prompt override for chat format."),
    click.option("--var", "variables", multiple=True, help="Inject prompt variable as KEY=VALUE."),
]

TASK_SPECIFIC_OPTIONS = [
    click.option("--chunk/--no-chunk", default=False, help="Enable token-aware sliding window chunking."),
    click.option("--as-array/--single-object", "as_array", default=None, help="Instruct model to output JSON array vs single object."),
    click.option("--dedup/--no-dedup", default=True, help="Enable deduplication of extracted items across chunks."),
    click.option("--dedup-key", type=str, help="Field or key name used to identify unique extracted items."),
    click.option("--category-field", type=str, help="Field name containing category for classification."),
    click.option("--confidence-field", type=str, help="Field name containing confidence for classification."),
    click.option("--reasoning-field", type=str, help="Field name containing reasoning for classification."),
    click.option("--summary-field", type=str, help="Field name containing summary text for summarization."),
    click.option("--points-field", type=str, help="Field name containing bullet points for summarization."),
]

MAP_OPTIONS = [
    click.option("--input-path", "-i", required=True, type=click.Path(exists=True), help="Input dataset path (.parquet, .csv, .json, .txt)."),
    click.option("--schemas-dir", "-s", required=True, type=click.Path(exists=True), help="Directory containing Pydantic schema files."),
    click.option("--output-dir", "-o", default="outputs", type=click.Path(), help="Output directory for relational tables and databases."),
    click.option("--export", default="sqlite,csv,wide", help="Comma-separated exports: sqlite,csv,wide."),
    click.option("--triage/--no-triage", default=True, help="Enable fast schema relevance triage filter."),
    click.option("--triage-max-tokens", type=int, default=256, help="Maximum tokens for triage classification."),
    click.option("--resume/--no-resume", default=True, help="Enable fault-tolerant checkpoint resume."),
    click.option("--id-column-name", default="source_id", help="Foreign key column name linking parent records in normalized tables."),
    click.option("--wide-table-name", default="consolidated_wide_table", help="Base file name for wide denormalized export."),
    click.option("--sqlite-db-name", default="extracted_database.db", help="SQLite database file name."),
    click.option("--wide-separator", default="; ", help="Separator used when concatenating multi-value fields in wide tables."),
]


def common_options(f):
    """Reusable CLI options across standard task commands."""
    all_task_options = (
        TASK_IO_OPTIONS
        + SHARED_IO_OPTIONS
        + SHARED_ENGINE_OPTIONS
        + SHARED_CHUNKING_OPTIONS
        + PROMPT_OPTIONS
        + TASK_SPECIFIC_OPTIONS
    )
    return apply_options(all_task_options)(f)


def map_options(f):
    """Composable CLI options for multi-schema map command."""
    all_map_options = (
        MAP_OPTIONS
        + SHARED_IO_OPTIONS
        + SHARED_ENGINE_OPTIONS
        + SHARED_CHUNKING_OPTIONS
    )
    return apply_options(all_map_options)(f)


@click.group()
@click.version_option(version="0.1.0", prog_name="prismrunner")
def main():
    """PrismRunner: High-throughput local-first LLM inference engine."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@main.command()
@common_options
def classify(**kwargs):
    """Classify text items or tabular dataset rows into categories."""
    _run_task_cli("classify", kwargs)


@main.command()
@common_options
def extract(**kwargs):
    """Extract entities, structured fields, or action items from text."""
    _run_task_cli("extract", kwargs)


@main.command()
@common_options
def summarize(**kwargs):
    """Generate structured summaries and key takeaways."""
    _run_task_cli("summarize", kwargs)


@main.command("map")
@map_options
def map_cmd(**kwargs):
    """Multi-schema extraction with relevance triage and relational aggregation."""
    from prismrunner.mapping_pipeline import SchemaMappingPipeline
    pipeline = SchemaMappingPipeline(schemas_dir=kwargs["schemas_dir"])
    export_formats = [f.strip().lower() for f in kwargs["export"].split(",") if f.strip()]
    meta_cols = [c.strip() for c in kwargs["metadata_columns"].split(",") if c.strip()] if kwargs.get("metadata_columns") else None
    excl_cols = [c.strip() for c in kwargs["exclude_columns"].split(",") if c.strip()] if kwargs.get("exclude_columns") else None
    pipeline.run(
        input_data=kwargs["input_path"],
        output_dir=kwargs["output_dir"],
        text_column=kwargs.get("text_column"),
        id_column=kwargs.get("id_column"),
        metadata_columns=meta_cols,
        exclude_columns=excl_cols,
        limit=kwargs.get("limit"),
        batch_size=kwargs.get("batch_size"),
        max_new_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        repetition_penalty=kwargs.get("repetition_penalty"),
        use_triage=kwargs.get("triage", True),
        triage_max_tokens=kwargs.get("triage_max_tokens", 256),
        resume=kwargs.get("resume", True),
        export_formats=export_formats,
        dry_run=kwargs.get("dry_run", False),
        chunk_size=kwargs.get("chunk_size", 1024),
        chunk_overlap=kwargs.get("chunk_overlap", 64),
        chunk_strategy=kwargs.get("chunk_strategy", "sentence"),
        id_column_name=kwargs.get("id_column_name", "source_id"),
        sqlite_db_name=kwargs.get("sqlite_db_name", "extracted_database.db"),
        wide_table_name=kwargs.get("wide_table_name", "consolidated_wide_table"),
        wide_list_separator=kwargs.get("wide_separator", "; "),
    )


@main.command("inspect-model")
@click.option("--model", "-m", default="Qwen/Qwen2.5-3B-Instruct", help="Model name or local directory.")
@click.option("--device", "-d", default="auto", help="Target device (auto, mps, cuda, cpu).")
def inspect_model(model: str, device: str):
    """Inspect model architecture, context window, token budgets, and hardware advisor."""
    console.print(f"[bold cyan]Inspecting model:[/bold cyan] {model}")
    target_device = get_optimal_device(None if device == "auto" else device)
    mem_info = get_memory_info(target_device)

    console.print(f"[bold green]Target Device:[/bold green] {target_device} ({mem_info.get('device_name', '')})")

    # Load tokenizer and lightweight config
    try:
        from transformers import AutoTokenizer, AutoConfig
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)

        ctx_limit = 4096
        for attr in ("max_position_embeddings", "seq_length", "n_positions", "max_sequence_length"):
            val = getattr(cfg, attr, None)
            if val and isinstance(val, int) and 512 <= val <= 1_000_000:
                ctx_limit = val
                break

        advisory = generate_hardware_advisory(model, ctx_limit, target_device)

        table = Table(title="PrismRunner Model & Hardware Advisory", border_style="bright_blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Model Repository", advisory["model_name"])
        table.add_row("Active Device", advisory["device"])
        table.add_row("Detected Context Window", f"{advisory['context_limit']:,} tokens")
        table.add_row("Recommended Dtype", advisory["recommended_dtype"])
        table.add_row("Recommended Batch Size", str(advisory["recommended_batch_size"]))
        table.add_row("Recommended Chunk Window", f"{advisory['recommended_chunk_size']} tokens")
        table.add_row("Max Safe Input Tokens", f"{advisory['max_safe_input_tokens']} tokens")
        table.add_row("Reserved Budget (Prompt+Gen)", f"{advisory['reserved_budget_tokens']} tokens")

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to inspect model {model}:[/bold red] {e}")


def _run_task_cli(task: str, kwargs: dict):
    """Helper orchestrating CLI invocations for classify, extract, and summarize."""
    input_source = kwargs.get("input_path") or kwargs.get("input_text")
    if not input_source:
        console.print("[bold red]Error:[/bold red] You must provide either --input-path (-i) or --input-text (-t).")
        raise SystemExit(1)

    overrides = {}
    if kwargs.get("model"):
        overrides.setdefault("model", {})["name"] = kwargs["model"]
    if kwargs.get("device"):
        overrides.setdefault("model", {})["device"] = kwargs["device"]
    if kwargs.get("batch_size"):
        overrides.setdefault("inference", {})["batch_size"] = kwargs["batch_size"]
    if kwargs.get("max_tokens"):
        overrides.setdefault("inference", {})["max_new_tokens"] = kwargs["max_tokens"]
    if kwargs.get("temperature") is not None:
        overrides.setdefault("inference", {})["temperature"] = kwargs["temperature"]
    if kwargs.get("top_p") is not None:
        overrides.setdefault("inference", {})["top_p"] = kwargs["top_p"]
    if kwargs.get("top_k") is not None:
        overrides.setdefault("inference", {})["top_k"] = kwargs["top_k"]
    if kwargs.get("repetition_penalty") is not None:
        overrides.setdefault("inference", {})["repetition_penalty"] = kwargs["repetition_penalty"]
    if kwargs.get("cache_dir"):
        overrides.setdefault("model", {})["cache_dir"] = kwargs["cache_dir"]
    if kwargs.get("local_files_only"):
        overrides.setdefault("model", {})["local_files_only"] = kwargs["local_files_only"]

    pipeline = PrismPipeline(
        config_path=kwargs.get("config"),
        config=overrides if overrides else None,
    )

    vars_dict = _parse_vars(kwargs.get("variables") or ())

    schema_cls = None
    if kwargs.get("schema_spec"):
        from prismrunner.schemas.compiler import load_schema_from_string
        schema_cls = load_schema_from_string(kwargs["schema_spec"])

    meta_cols = [c.strip() for c in kwargs["metadata_columns"].split(",") if c.strip()] if kwargs.get("metadata_columns") else None
    excl_cols = [c.strip() for c in kwargs["exclude_columns"].split(",") if c.strip()] if kwargs.get("exclude_columns") else None

    results = pipeline.run(
        task=task,
        input_data=input_source,
        schema_cls=schema_cls,
        text_column=kwargs.get("text_column"),
        id_column=kwargs.get("id_column"),
        metadata_columns=meta_cols,
        exclude_columns=excl_cols,
        limit=kwargs.get("limit"),
        batch_size=kwargs.get("batch_size"),
        max_new_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        top_k=kwargs.get("top_k"),
        repetition_penalty=kwargs.get("repetition_penalty"),
        chunk=kwargs.get("chunk"),
        chunk_size=kwargs.get("chunk_size"),
        chunk_overlap=kwargs.get("chunk_overlap"),
        chunk_strategy=kwargs.get("chunk_strategy"),
        prompt_file=kwargs.get("prompt_file"),
        system_prompt=kwargs.get("system_prompt"),
        as_array=kwargs.get("as_array"),
        dedup=kwargs.get("dedup", True),
        dedup_key=kwargs.get("dedup_key"),
        category_field=kwargs.get("category_field"),
        confidence_field=kwargs.get("confidence_field"),
        reasoning_field=kwargs.get("reasoning_field"),
        summary_field=kwargs.get("summary_field"),
        points_field=kwargs.get("points_field"),
        variables=vars_dict,
        instruction=kwargs.get("instruction"),
        output_path=kwargs.get("output_path"),
        output_format=kwargs.get("output_format"),
        dry_run=kwargs.get("dry_run", False),
    )

    if kwargs.get("json_stdout"):
        print(json.dumps(results, indent=2, default=str))
    elif not kwargs.get("dry_run"):
        console.print(f"\n[bold green]Success:[/bold green] Processed {len(results)} records.")
        if results and len(results) <= 5:
            console.print("\n[bold cyan]Output Preview:[/bold cyan]")
            console.print(json.dumps(results, indent=2, default=str))
        elif results:
            console.print(f"[dim]Output containing {len(results)} records was generated. Pass --json to view full output or check destination file.[/dim]")


if __name__ == "__main__":
    main()
