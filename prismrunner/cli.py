"""PrismRunner Command Line Interface."""

from __future__ import annotations
import json
import logging
from typing import Tuple

import click
from rich.console import Console
from rich.table import Table

from prismrunner.engine import (
    DEFAULT_MODEL_NAME,
    generate_hardware_advisory,
    get_context_window,
    get_memory_info,
    get_optimal_device,
)
from prismrunner.pipeline import PrismPipeline
from prismrunner.schema import load_schema_from_string

console = Console()


def _parse_vars(var_tuples: Tuple[str, ...]) -> dict:
    parsed = {}
    for item in var_tuples:
        if "=" in item:
            k, v = item.split("=", 1)
            parsed[k.strip()] = v.strip()
    return parsed


def common_options(f):
    options = [
        click.option("--input-path", "-i", type=click.Path(exists=False), help="Path to input dataset (.parquet, .csv, .json, .txt)."),
        click.option("--input-text", "-t", type=str, help="Single raw text string to process directly."),
        click.option("--output-path", "-o", type=click.Path(), help="Output file destination."),
        click.option("--output-format", "-f", type=click.Choice(["json", "csv", "parquet"], case_sensitive=False), default="json"),
        click.option("--json", "json_stdout", is_flag=True, help="Output raw JSON to stdout."),
        click.option("--column", "--text-column", "text_column", type=str, help="Column containing input text."),
        click.option("--id-column", type=str, help="Column containing unique record ID."),
        click.option("--limit", "-l", type=int, help="Maximum number of records to process."),
        click.option("--dry-run", is_flag=True, help="Render prompt without loading model."),
        click.option("--model", "-m", type=str, default=None, help="Hugging Face model ID or local path."),
        click.option("--device", "-d", type=str, default=None, help="Compute device (auto, mps, cuda, cpu)."),
        click.option("--batch-size", "-b", type=int, default=None, help="Inference batch size (defaults to hardware-aware auto)."),
        click.option("--max-tokens", type=int, default=None, help="Maximum new tokens to generate."),
        click.option("--temperature", type=float, default=None, help="Sampling temperature (0.0 for deterministic)."),
        click.option("--schema", "-s", "schema_spec", type=str, help="Custom schema spec: 'path/to/file.py:ClassName'."),
        click.option("--prompt-file", type=click.Path(exists=True), help="Path to custom prompt template YAML."),
        click.option("--instruction", type=str, help="Instruction override for prompt template."),
        click.option("--system-prompt", type=str, help="System prompt override."),
        click.option("--var", "variables", multiple=True, help="Inject prompt variable as KEY=VALUE."),
        click.option("--chunk/--no-chunk", default=None, help="Enable/disable token-aware chunking (defaults to context-aware auto)."),
        click.option("--chunk-size", type=int, default=None, help="Tokens per chunk window (defaults to context-aware auto)."),
        click.option("--chunk-overlap", type=int, default=None, help="Tokens overlap between chunks."),
    ]
    for opt in reversed(options):
        f = opt(f)
    return f


@click.group()
@click.version_option(version="0.1.0", prog_name="prismrunner")
def main():
    """PrismRunner: Local-first LLM batch inference engine for Pydantic schemas."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


@main.command()
@common_options
def classify(**kwargs):
    """Classify text items or dataset rows into structured categories."""
    _run_task_cli("classify", kwargs)


@main.command()
@common_options
def extract(**kwargs):
    """Extract entities or custom Pydantic schemas from text."""
    _run_task_cli("extract", kwargs)


@main.command()
@common_options
def summarize(**kwargs):
    """Generate structured summaries and key takeaways."""
    _run_task_cli("summarize", kwargs)


@main.command("inspect-model")
@click.option("--model", "-m", default=DEFAULT_MODEL_NAME, help="Model name or local directory.")
@click.option("--device", "-d", default="auto", help="Target device (auto, mps, cuda, cpu).")
def inspect_model(model: str, device: str):
    """Inspect model context window, token budgets, and hardware recommendations."""
    target_device = get_optimal_device(device)
    mem_info = get_memory_info(target_device)
    console.print(f"[bold green]Target Device:[/bold green] {target_device} ({mem_info.get('device_name', '')})")

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        ctx_limit = get_context_window(cfg)
        advisory = generate_hardware_advisory(model, ctx_limit, target_device)

        table = Table(title="PrismRunner Model & Hardware Advisory", border_style="bright_blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        for k, v in advisory.items():
            table.add_row(k, str(v))
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Failed to inspect model {model}:[/bold red] {e}")


def _run_task_cli(task: str, kwargs: dict):
    input_source = kwargs.get("input_path") or kwargs.get("input_text")
    if not input_source:
        console.print("[bold red]Error:[/bold red] Provide either --input-path (-i) or --input-text (-t).")
        raise SystemExit(1)

    pipeline = PrismPipeline(
        model_name=kwargs.get("model"),
        device=kwargs.get("device"),
        batch_size=kwargs.get("batch_size"),
    )
    schema_cls = load_schema_from_string(kwargs["schema_spec"]) if kwargs.get("schema_spec") else None

    results = pipeline.run(
        task=task,
        input_data=input_source,
        schema_cls=schema_cls,
        text_column=kwargs.get("text_column"),
        id_column=kwargs.get("id_column"),
        limit=kwargs.get("limit"),
        batch_size=kwargs.get("batch_size"),
        max_new_tokens=kwargs.get("max_tokens"),
        temperature=kwargs.get("temperature"),
        chunk=kwargs.get("chunk"),
        chunk_size=kwargs.get("chunk_size"),
        chunk_overlap=kwargs.get("chunk_overlap"),
        prompt_file=kwargs.get("prompt_file"),
        system_prompt=kwargs.get("system_prompt"),
        variables=_parse_vars(kwargs.get("variables") or ()),
        instruction=kwargs.get("instruction"),
        output_path=kwargs.get("output_path"),
        output_format=kwargs.get("output_format"),
        dry_run=kwargs.get("dry_run", False),
    )

    if kwargs.get("json_stdout") or kwargs.get("dry_run"):
        print(json.dumps(results, indent=2, default=str))
    else:
        console.print(f"[bold green]Processed {len(results)} records.[/bold green]")
        if results and len(results) <= 5:
            console.print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
