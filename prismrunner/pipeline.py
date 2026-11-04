"""Unified PrismRunner Pipeline orchestrating loading, chunking, inference, parsing, and stitching."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel
import yaml

from prismrunner.data import (
    ChunkRecord,
    DatasetLoader,
    DeterministicStitcher,
    TokenAwareChunker,
    export_results,
)
from prismrunner.engine import (
    DEFAULT_MODEL_NAME,
    BatchInferenceEngine,
    ContextBudget,
    generate_hardware_advisory,
    get_optimal_device,
    load_model_and_tokenizer,
)
from prismrunner.prompt import PromptTemplate
from prismrunner.schema import (
    ClassificationItem,
    CompiledPromptSchema,
    EntityItem,
    SummaryItem,
    UniversalOutputParser,
    compile_schema_to_prompt,
)

logger = logging.getLogger(__name__)

TASK_DEFAULT_SCHEMAS: Dict[str, Type[BaseModel]] = {
    "classify": ClassificationItem,
    "extract": EntityItem,
    "summarize": SummaryItem,
}


class PrismPipeline:
    """End-to-end local batch LLM execution pipeline for structured Pydantic extraction."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.config = self._load_config(config_path, config)
        if model_name:
            self.config["model"]["name"] = model_name
        if device:
            self.config["model"]["device"] = device
        if batch_size is not None:
            self.config["inference"]["batch_size"] = batch_size

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
        """Merge default configuration, optional YAML file, and explicit overrides."""
        base_cfg: Dict[str, Any] = {
            "model": {
                "name": DEFAULT_MODEL_NAME,
                "device": "auto",
                "torch_dtype": "auto",
                "load_in_4bit": False,
                "load_in_8bit": False,
                "trust_remote_code": True,
                "attn_implementation": "auto",
                "compile_model": False,
            },
            "inference": {
                "batch_size": "auto",
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.95,
                "bucket_by_length": True,
                "constrain_json": True,
                "system_prompt": "You are PrismRunner, an accurate local inference engine. Output structured JSON strictly adhering to schema.",
            },
            "chunking": {
                "enabled": "auto",
                "chunk_size": "auto",
                "chunk_overlap": 64,
                "strategy": "sentence",
            },
            "io": {
                "default_id_column": "id",
                "default_text_column": "text",
                "output_format": "json",
            },
        }

        if config_path:
            target_path = Path(config_path)
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
        batch_size: Optional[Union[int, str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BatchInferenceEngine:
        """Initialize or reuse the batch inference engine with hardware-aware batch sizing."""
        if self.engine is not None:
            if isinstance(batch_size, int) and batch_size > 0:
                self.engine.batch_size = batch_size
            return self.engine

        m_cfg = self.config["model"]
        i_cfg = self.config["inference"]

        m_name = model_name or m_cfg["name"]
        d_pref = device_name or (None if m_cfg.get("device") == "auto" else m_cfg.get("device"))
        dtype_pref = torch_dtype or (None if m_cfg.get("torch_dtype") == "auto" else m_cfg.get("torch_dtype"))
        m_tokens = max_new_tokens or i_cfg["max_new_tokens"]
        temp = temperature if temperature is not None else i_cfg["temperature"]
        tp = top_p if top_p is not None else i_cfg.get("top_p", 0.95)
        tk = top_k if top_k is not None else i_cfg.get("top_k", None)
        rp = repetition_penalty if repetition_penalty is not None else i_cfg.get("repetition_penalty", None)
        gk = generation_kwargs or i_cfg.get("generation_kwargs", {})

        target_device = get_optimal_device(d_pref)

        self.model, self.tokenizer, self.device, self.context_window = load_model_and_tokenizer(
            model_name_or_path=m_name,
            device=target_device,
            torch_dtype=dtype_pref,
            load_in_4bit=m_cfg.get("load_in_4bit", False),
            load_in_8bit=m_cfg.get("load_in_8bit", False),
            trust_remote_code=m_cfg.get("trust_remote_code", True),
            token=m_cfg.get("token"),
            cache_dir=m_cfg.get("cache_dir"),
            local_files_only=m_cfg.get("local_files_only", False),
            attn_implementation=m_cfg.get("attn_implementation", "auto"),
            compile_model=m_cfg.get("compile_model", False),
        )

        # Dynamically resolve batch_size from hardware advisory if set to "auto"
        raw_b_size = batch_size if batch_size is not None else i_cfg.get("batch_size", "auto")
        if raw_b_size == "auto" or not isinstance(raw_b_size, int):
            advisory = generate_hardware_advisory(
                model_name=m_name,
                context_limit=self.context_window,
                device=self.device,
                max_new_tokens=m_tokens,
            )
            resolved_b_size = advisory["recommended_batch_size"]
        else:
            resolved_b_size = raw_b_size

        self.engine = BatchInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=resolved_b_size,
            max_new_tokens=m_tokens,
            temperature=temp,
            top_p=tp,
            top_k=tk,
            repetition_penalty=rp,
            context_window=self.context_window,
            generation_kwargs=gk,
        )
        return self.engine

    def render_prompt(
        self,
        template_yaml: Dict[str, Any],
        text: str,
        compiled_schema: Optional[CompiledPromptSchema] = None,
        variable_overrides: Optional[Dict[str, str]] = None,
        instruction_override: Optional[str] = None,
        as_array: Optional[bool] = None,
    ) -> str:
        """Render prompt text substituting variables, instructions, schema directives, and text."""
        target_as_array = as_array if as_array is not None else True
        schema_block = compiled_schema.render_prompt_block(as_array=target_as_array) if compiled_schema else ""
        return PromptTemplate.render(
            template=template_yaml,
            text=text,
            schema_block=schema_block,
            variables=variable_overrides,
            instruction=instruction_override,
        )

    def _resolve_dynamic_chunk_budget(
        self,
        tmpl: Dict[str, Any],
        compiled_schema: Optional[CompiledPromptSchema],
        sys_prompt: Optional[str],
        variables: Optional[Dict[str, str]],
        instruction: Optional[str],
        as_array: Optional[bool],
        max_new_tokens: int,
        requested_chunk_size: Optional[Union[int, str]],
    ) -> int:
        """Compute exact safe input token budget from context window, prompt template, and schema."""
        empty_prompt = self.render_prompt(
            template_yaml=tmpl,
            text="",
            compiled_schema=compiled_schema,
            variable_overrides=variables,
            instruction_override=instruction,
            as_array=as_array,
        )
        try:
            budget = ContextBudget.from_tokenizer(
                tokenizer=self.tokenizer,
                context_limit=self.context_window,
                system_prompt=sys_prompt,
                template_text=empty_prompt,
                max_new_tokens=max_new_tokens,
            )
            max_safe = budget.max_input_tokens
        except ValueError:
            max_safe = max(256, self.context_window // 2)

        if isinstance(requested_chunk_size, int) and requested_chunk_size > 0:
            return min(requested_chunk_size, max_safe)

        if self.device is not None:
            advisory = generate_hardware_advisory(
                model_name=self.config["model"]["name"],
                context_limit=self.context_window,
                device=self.device,
                max_new_tokens=max_new_tokens,
            )
            return min(max_safe, advisory["recommended_chunk_size"])

        return min(max_safe, max(512, self.context_window // 2))

    def run(
        self,
        task: str = "extract",
        input_data: Optional[Union[str, Path, Any]] = None,
        schema_cls: Optional[Type[BaseModel]] = None,
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        batch_size: Optional[Union[int, str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        chunk: Optional[Union[bool, str]] = None,
        chunk_size: Optional[Union[int, str]] = None,
        chunk_overlap: Optional[int] = None,
        chunk_strategy: Optional[str] = None,
        prompt_file: Optional[Union[str, Path]] = None,
        prompt_template: Optional[Union[str, Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        instruction: Optional[str] = None,
        as_array: Optional[bool] = None,
        constrain_json: Optional[bool] = None,
        dedup: bool = True,
        dedup_key: Optional[Union[str, Callable]] = None,
        category_field: Optional[str] = None,
        confidence_field: Optional[str] = None,
        reasoning_field: Optional[str] = None,
        tie_breaker: str = "earliest",
        summary_field: Optional[str] = None,
        points_field: Optional[str] = None,
        part_header_template: Optional[str] = "--- [Part {index}/{total}] ---",
        output_path: Optional[Union[str, Path]] = None,
        output_format: Optional[str] = None,
        dry_run: bool = False,
        on_chunk_processed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute the end-to-end pipeline on strings, lists, DataFrames, or dataset files."""
        target_input = input_data if input_data is not None else self.config.get("io", {}).get("input_path")
        if target_input is None:
            raise ValueError("No input_data provided.")

        records = DatasetLoader.load(
            file_path_or_data=target_input,
            text_column=text_column or self.config["io"].get("default_text_column"),
            id_column=id_column or self.config["io"].get("default_id_column"),
            metadata_columns=metadata_columns,
            exclude_columns=exclude_columns,
            limit=limit,
        )
        if not records:
            return []

        target_schema = schema_cls or TASK_DEFAULT_SCHEMAS.get(task)
        compiled_schema = compile_schema_to_prompt(target_schema) if target_schema else None
        tmpl = PromptTemplate.load(source=prompt_template or prompt_file, default_task=task)
        sys_prompt = system_prompt or tmpl.get("system_prompt") or self.config["inference"].get("system_prompt")
        m_tokens = max_new_tokens or self.config["inference"].get("max_new_tokens", 512)

        # Initialize engine before chunking (unless dry_run) so actual model context window & tokenizer are used
        if not dry_run and self.engine is None:
            self.init_engine(
                batch_size=batch_size,
                max_new_tokens=m_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generation_kwargs=generation_kwargs,
            )

        raw_chunk_flag = chunk if chunk is not None else self.config["chunking"].get("enabled", "auto")
        req_c_size = chunk_size if chunk_size is not None else self.config["chunking"].get("chunk_size", "auto")
        c_size = self._resolve_dynamic_chunk_budget(
            tmpl=tmpl,
            compiled_schema=compiled_schema,
            sys_prompt=sys_prompt,
            variables=variables,
            instruction=instruction,
            as_array=as_array,
            max_new_tokens=m_tokens,
            requested_chunk_size=req_c_size,
        )
        c_overlap = chunk_overlap if chunk_overlap is not None else self.config["chunking"].get("chunk_overlap", 64)
        c_strat = chunk_strategy or self.config["chunking"].get("strategy", "sentence")

        # When chunk="auto" or True, TokenAwareChunker keeps short records as 1 chunk and splits only records > c_size
        if raw_chunk_flag is True or raw_chunk_flag == "auto":
            chunker = TokenAwareChunker(
                tokenizer=self.tokenizer,
                chunk_size=c_size,
                chunk_overlap=c_overlap,
                strategy=c_strat,
            )
            chunks: List[ChunkRecord] = chunker.chunk_dataset(records)
        else:
            chunker_counter = TokenAwareChunker(tokenizer=self.tokenizer)
            chunks = [
                ChunkRecord(
                    record_id=r.id,
                    record_index=r.index,
                    chunk_index=0,
                    total_chunks=1,
                    text=r.text,
                    token_count=chunker_counter.count_tokens(r.text),
                    metadata=r.metadata,
                )
                for r in records
            ]

        payloads: List[Dict[str, Any]] = [
            {
                "record_id": ch.record_id,
                "record_index": ch.record_index,
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.total_chunks,
                "text": ch.text,
                "prompt": self.render_prompt(
                    template_yaml=tmpl,
                    text=ch.text,
                    compiled_schema=compiled_schema,
                    variable_overrides=variables,
                    instruction_override=instruction,
                    as_array=as_array,
                ),
                "metadata": ch.metadata,
            }
            for ch in chunks
        ]

        if dry_run:
            return payloads

        engine = self.init_engine(
            batch_size=batch_size,
            max_new_tokens=m_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            generation_kwargs=generation_kwargs,
        )

        def _parse_batch_items(batch_completions: List[str], batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            batch_annotated = []
            for item, comp in zip(batch_items, batch_completions):
                res = dict(item, raw_output=comp)
                parse_res = UniversalOutputParser.parse(text=comp, model_cls=target_schema)
                res["parsed_items"] = parse_res.items
                res["parse_success"] = parse_res.success
                res["parsing_strategy"] = parse_res.parsing_strategy
                res["parse_error"] = parse_res.error
                if on_chunk_processed:
                    try:
                        on_chunk_processed(res)
                    except Exception as e:
                        logger.warning(f"Error in on_chunk_processed callback: {e}")
                batch_annotated.append(res)
            return batch_annotated

        use_constrain = (
            constrain_json
            if constrain_json is not None
            else self.config["inference"].get("constrain_json", True)
        )
        target_as_array = as_array if as_array is not None else True
        response_prefix = ("[" if target_as_array else "{") if (use_constrain and target_schema is not None) else None

        raw_results = engine.run_all(
            items=payloads,
            system_prompt=sys_prompt,
            max_new_tokens=m_tokens,
            temperature=temperature,
            prompt_key="prompt",
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            bucket_by_length=self.config["inference"].get("bucket_by_length", True),
            on_batch_complete=_parse_batch_items,
            response_prefix=response_prefix,
            stop_on_json_close=bool(use_constrain),
            **(generation_kwargs or {}),
        )

        for res in raw_results:
            if "parsed_items" not in res:
                parse_res = UniversalOutputParser.parse(text=res.get("raw_output", ""), model_cls=target_schema)
                res["parsed_items"] = parse_res.items
                res["parse_success"] = parse_res.success
                res["parsing_strategy"] = parse_res.parsing_strategy
                res["parse_error"] = parse_res.error

        if task == "classify":
            stitched = DeterministicStitcher.stitch_classification(
                raw_results,
                category_field=category_field,
                confidence_field=confidence_field,
                reasoning_field=reasoning_field,
                tie_breaker=tie_breaker,
            )
        elif task == "extract":
            stitched = DeterministicStitcher.stitch_extraction(
                raw_results,
                dedup=dedup,
                dedup_key=dedup_key,
            )
        elif task == "summarize":
            stitched = DeterministicStitcher.stitch_summarization(
                raw_results,
                summary_field=summary_field,
                points_field=points_field,
                part_header_template=part_header_template,
            )
        else:
            stitched = raw_results

        if output_path:
            export_results(stitched, output_path, output_format)

        return stitched
