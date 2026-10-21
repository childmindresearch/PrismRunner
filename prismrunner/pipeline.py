"""Unified PrismRunner Pipeline orchestrating loading, chunking, inference, parsing, and stitching."""

from __future__ import annotations
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import yaml
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from prismrunner.engine.device import get_optimal_device, clear_device_memory
from prismrunner.engine.model import load_model_and_tokenizer
from prismrunner.engine.inference import BatchInferenceEngine
from prismrunner.engine.context import ContextBudget
from prismrunner.chunking.chunker import TokenAwareChunker, ChunkRecord
from prismrunner.loaders.dataset import DatasetLoader, Record, export_results
from prismrunner.schemas.compiler import compile_schema_to_prompt, CompiledPromptSchema
from prismrunner.schemas.parser import UniversalOutputParser, ParseResult
from prismrunner.schemas.base import ClassificationItem, EntityItem, SummaryItem
from prismrunner.stitcher.stitcher import DeterministicStitcher

logger = logging.getLogger(__name__)
console = Console()


TASK_DEFAULT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "classify": ClassificationItem,
    "extract": EntityItem,
    "summarize": SummaryItem,
}


class PrismPipeline:
    """End-to-end task execution pipeline."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.config = self._load_config(config_path, config)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.context_window = 4096
        self.engine: Optional[BatchInferenceEngine] = None

    def _load_config(
        self,
        config_path: Optional[Union[str, Path]],
        override_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge default configuration, YAML configuration, and explicit overrides."""
        base_cfg: Dict[str, Any] = {
            "model": {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "device": "auto",
                "torch_dtype": "auto",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "trust_remote_code": True,
            },
            "inference": {
                "batch_size": 4,
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.95,
                "system_prompt": "You are PrismRunner, an accurate local inference engine. Output structured JSON strictly adhering to schema.",
            },
            "chunking": {
                "enabled": False,
                "chunk_size": 1024,
                "chunk_overlap": 64,
                "strategy": "sentence",
            },
            "prompts": {
                "classify": "prompts/classify.yaml",
                "extract": "prompts/extract.yaml",
                "summarize": "prompts/summarize.yaml",
            },
            "io": {
                "default_id_column": "id",
                "default_text_column": "text",
                "output_format": "json",
                "output_dir": "outputs",
            },
        }

        # Check for config file
        target_path = Path(config_path) if config_path else Path("config/pipeline_config.yaml")
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f) or {}
                for section, vals in file_cfg.items():
                    if isinstance(vals, dict) and section in base_cfg:
                        base_cfg[section].update(vals)
                    else:
                        base_cfg[section] = vals
            except Exception as e:
                logger.warning(f"Could not load config file {target_path}: {e}")

        if override_config:
            for section, vals in override_config.items():
                if isinstance(vals, dict) and section in base_cfg:
                    base_cfg[section].update(vals)
                else:
                    base_cfg[section] = vals

        return base_cfg

    def init_engine(
        self,
        model_name: Optional[str] = None,
        device_name: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> BatchInferenceEngine:
        """Initialize or reuse the batch inference engine."""
        if self.engine is not None:
            return self.engine

        m_cfg = self.config["model"]
        i_cfg = self.config["inference"]

        m_name = model_name or m_cfg["name"]
        d_pref = device_name or (None if m_cfg.get("device") == "auto" else m_cfg.get("device"))
        dtype_pref = torch_dtype or (None if m_cfg.get("torch_dtype") == "auto" else m_cfg.get("torch_dtype"))
        b_size = batch_size or i_cfg["batch_size"]
        m_tokens = max_new_tokens or i_cfg["max_new_tokens"]
        temp = temperature if temperature is not None else i_cfg["temperature"]

        target_device = get_optimal_device(d_pref)

        self.model, self.tokenizer, self.device, self.context_window = load_model_and_tokenizer(
            model_name_or_path=m_name,
            device=target_device,
            torch_dtype=dtype_pref,
            load_in_4bit=m_cfg.get("load_in_4bit", False),
            load_in_8bit=m_cfg.get("load_in_8bit", False),
            trust_remote_code=m_cfg.get("trust_remote_code", True),
        )

        self.engine = BatchInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=b_size,
            max_new_tokens=m_tokens,
            temperature=temp,
            top_p=i_cfg.get("top_p", 0.95),
        )

        return self.engine

    def load_prompt_template(
        self,
        task: str,
        custom_prompt_path: Optional[Union[str, Path]] = None,
        prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Load YAML prompt template for a specific task or use provided in-memory template."""
        if prompt_template is not None:
            if isinstance(prompt_template, dict):
                return dict(prompt_template)
            elif isinstance(prompt_template, str):
                return {
                    "version": "custom",
                    "task": task,
                    "instruction": "",
                    "variables": {},
                    "prompt": prompt_template,
                }

        prompt_path = None
        if custom_prompt_path:
            prompt_path = Path(custom_prompt_path)
        else:
            default_rel = self.config["prompts"].get(task)
            if default_rel:
                prompt_path = Path(default_rel)

        if prompt_path and prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        # Fallback inline templates
        return {
            "version": "1.0",
            "task": task,
            "instruction": f"Perform {task} accurately on the text below.",
            "variables": {},
            "prompt": "Source Content:\n\"\"\"\n{{ text }}\n\"\"\"\n\n{schema_block}",
        }

    def render_prompt(
        self,
        template_yaml: Dict[str, Any],
        text: str,
        compiled_schema: Optional[CompiledPromptSchema] = None,
        variable_overrides: Optional[Dict[str, str]] = None,
        instruction_override: Optional[str] = None,
    ) -> str:
        """Render prompt text substituting variables, instructions, schema directives, and text."""
        raw_tmpl = template_yaml.get("prompt", "{{ text }}")
        variables = dict(template_yaml.get("variables", {}))
        if variable_overrides:
            variables.update(variable_overrides)

        instruction = instruction_override or template_yaml.get("instruction", "")
        variables["instruction"] = instruction

        schema_block = compiled_schema.render_prompt_block(as_array=True) if compiled_schema else ""
        variables["schema_block"] = schema_block

        # Interpolate {variables}
        rendered = raw_tmpl
        for var_name, var_val in variables.items():
            rendered = rendered.replace(f"{{{var_name}}}", str(var_val))

        # Interpolate {{ text }} / {{ item }} slot
        rendered = rendered.replace("{{ text }}", text).replace("{{ text}}", text)
        rendered = rendered.replace("{{ item }}", text).replace("{{ item}}", text)

        return rendered

    def run(
        self,
        task: str,
        input_data: Union[str, Path, List[Record]],
        schema_cls: Optional[Type[BaseModel]] = None,
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        limit: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        chunk: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_strategy: Optional[str] = None,
        prompt_file: Optional[Union[str, Path]] = None,
        prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        instruction: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        output_format: Optional[str] = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute the end-to-end pipeline."""
        # 1. Load input dataset
        if isinstance(input_data, list) and all(isinstance(r, Record) for r in input_data):
            records = input_data[:limit] if limit else input_data
        else:
            records = DatasetLoader.load(
                file_path_or_text=input_data,
                text_column=text_column or self.config["io"].get("default_text_column"),
                id_column=id_column or self.config["io"].get("default_id_column"),
                limit=limit,
            )

        if not records:
            logger.warning("No records found to process.")
            return []

        # 2. Compile schema
        target_schema = schema_cls or TASK_DEFAULT_SCHEMAS.get(task)
        compiled_schema = compile_schema_to_prompt(target_schema) if target_schema else None

        # 3. Load prompt template
        tmpl = self.load_prompt_template(task, prompt_file, prompt_template=prompt_template)

        # 4. Handle chunking
        do_chunk = chunk if chunk is not None else self.config["chunking"].get("enabled", False)
        c_size = chunk_size or self.config["chunking"].get("chunk_size", 1024)
        c_overlap = chunk_overlap or self.config["chunking"].get("chunk_overlap", 64)
        c_strat = chunk_strategy or self.config["chunking"].get("strategy", "sentence")

        if do_chunk:
            tok = self.tokenizer if self.tokenizer else None
            chunker = TokenAwareChunker(
                tokenizer=tok,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                strategy=c_strat,
            )
            chunks: List[ChunkRecord] = chunker.chunk_dataset(records)
        else:
            # 1-to-1 mapping
            chunks = [
                ChunkRecord(
                    record_id=r.id,
                    record_index=r.index,
                    chunk_index=0,
                    total_chunks=1,
                    text=r.text,
                    token_count=len(r.text) // 4,
                    metadata=r.metadata,
                )
                for r in records
            ]

        # 5. Build prompt payloads
        payloads: List[Dict[str, Any]] = []
        for ch in chunks:
            rendered_prompt = self.render_prompt(
                template_yaml=tmpl,
                text=ch.text,
                compiled_schema=compiled_schema,
                variable_overrides=variables,
                instruction_override=instruction,
            )
            payloads.append({
                "record_id": ch.record_id,
                "record_index": ch.record_index,
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.total_chunks,
                "text": ch.text,
                "prompt": rendered_prompt,
                "metadata": ch.metadata,
            })

        # Dry run inspection mode
        if dry_run:
            console.print("[bold yellow]*** DRY RUN MODE ***[/bold yellow]")
            console.print(f"Total parent records: {len(records)}")
            console.print(f"Total chunks: {len(payloads)}")
            if payloads:
                console.print("\n[bold cyan]Sample Rendered Prompt (Chunk 1):[/bold cyan]")
                console.print(payloads[0]["prompt"])
            return payloads

        # 6. Initialize engine & run inference
        engine = self.init_engine(
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        sys_prompt = system_prompt or tmpl.get("system_prompt") or self.config["inference"].get("system_prompt")
        raw_results = engine.run_all(
            items=payloads,
            system_prompt=sys_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_key="prompt",
        )

        # 7. Parse outputs with UniversalOutputParser
        for res in raw_results:
            parse_res = UniversalOutputParser.parse(
                text=res.get("raw_output", ""),
                model_cls=target_schema,
            )
            res["parsed_items"] = parse_res.items
            res["parse_success"] = parse_res.success
            res["parsing_strategy"] = parse_res.parsing_strategy

        # 8. Deterministic Stitching
        if task == "classify":
            stitched = DeterministicStitcher.stitch_classification(raw_results)
        elif task == "extract":
            stitched = DeterministicStitcher.stitch_extraction(raw_results)
        elif task == "summarize":
            stitched = DeterministicStitcher.stitch_summarization(raw_results)
        else:
            stitched = raw_results

        # 9. Output export
        if output_path:
            export_results(stitched, output_path, output_format)

        return stitched
