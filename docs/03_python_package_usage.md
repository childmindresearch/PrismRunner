# Python Package API Reference

PrismRunner is designed as a modular Python package that can be imported into your custom data pipelines, ETL systems, background workers, and web services.

---

## 1. Primary High-Level Pipelines

### `PrismPipeline`
The primary single-task orchestrator for classification, extraction, and summarization.

```python
from prismrunner import PrismPipeline

pipeline = PrismPipeline(
    config_path="config/pipeline_config.yaml",  # Optional custom config
    model_name="Qwen/Qwen2.5-3B-Instruct",      # Override model
    device="auto",                             # "auto", "mps", "cuda", "cpu"
)
```

#### Method: `pipeline.run(...)`
```python
results = pipeline.run(
    task="extract",                            # "classify", "extract", or "summarize"
    input_data="data/corpus.parquet",          # File path, raw string, or List[Record]
    schema_cls=CustomModel,                    # Target Pydantic model class
    text_column="body",                        # Column containing document text
    id_column="doc_id",                        # Column containing unique ID
    limit=100,                                 # Optional row limit
    batch_size=8,                              # Batch size forward pass
    max_new_tokens=512,                        # Max tokens generated per prompt
    temperature=0.0,                           # Sampling temperature (0.0 for deterministic)
    chunk=True,                                # Enable token sliding-window chunker
    chunk_size=1024,                           # Max tokens per chunk
    chunk_overlap=64,                          # Overlapping tokens between chunks
    chunk_strategy="sentence",                 # "sentence" or "token"
    prompt_file="prompts/custom.yaml",         # Optional path to YAML template
    prompt_template="...",                     # In-memory template string or dict
    system_prompt="...",                       # Runtime system prompt override
    instruction="...",                         # Task instruction override
    variables={"key": "val"},                  # Dictionary of template variable overrides
    output_path="outputs/results.json",        # Path to export output
    output_format="json",                      # "json", "csv", or "parquet"
    dry_run=False,                             # If True, returns rendered payloads without running LLM
)
```

---

### `SchemaMappingPipeline` (Alias: `SchemaMapPipeline`)
Orchestrates multi-schema catalog extraction, vectorized triage pre-screening, checkpoint resume, and relational normalization into SQLite, normalized CSVs, and wide tables.

```python
from prismrunner import SchemaMappingPipeline, SchemaCatalog

catalog = SchemaCatalog(schemas_dir="schemas/domain/")
mapper = SchemaMappingPipeline(catalog=catalog)

summary = mapper.run(
    input_data="data/clinical_corpus.parquet",
    output_dir="outputs/run_01",
    text_column="text",
    id_column="note_id",
    use_triage=True,                           # Fast pre-screening ChatML filter
    resume=True,                               # Granular checkpoint resume
    export_formats=["sqlite", "csv", "wide"],  # Relational export formats
    batch_size=8,
    extract_prompt_template="...",             # In-memory prompt override
    system_prompt="...",                       # System prompt override
)
```

---

## 2. Low-Level Modular Components

PrismRunner's components can be imported individually for custom architectures.

### `DatasetLoader` & `Record`
Load datasets deterministically across Parquet, CSV, JSON, TXT, and Markdown files:

```python
from prismrunner import DatasetLoader, Record

records = DatasetLoader.load(
    file_path_or_data="data/corpus.parquet",
    text_column="notes",
    id_column="patient_id",
    limit=50,
)

print(f"Loaded {len(records)} records. First ID: {records[0].id}")
```

---

### `TokenAwareChunker`
Splits documents on sentence or token boundaries while tracking chunk indices:

```python
from prismrunner import TokenAwareChunker

chunker = TokenAwareChunker(
    tokenizer=pipeline.tokenizer,
    chunk_size=1024,
    chunk_overlap=64,
    strategy="sentence",  # Preserves sentence boundaries
)

chunks = chunker.chunk_dataset(records)
print(f"Generated {len(chunks)} token-budgeted chunks.")
```

---

### `compile_schema_to_prompt`
Inspects Pydantic models and generates human-readable prompt blocks and sample JSONs without OpenAPI leaks:

```python
from prismrunner import compile_schema_to_prompt
from pydantic import BaseModel, Field

class DiagnosticFinding(BaseModel):
    condition: str = Field(..., description="Condition or disease name")
    status: str = Field(..., description="active, resolved, or rule-out")

compiled = compile_schema_to_prompt(DiagnosticFinding)
prompt_directive = compiled.render_prompt_block(as_array=True)
print(prompt_directive)
```

---

### `UniversalOutputParser`
Robustly strips markdown fences, repairs trailing commas, extracts JSON arrays/objects, filters phantom `None` objects, and validates against Pydantic models:

```python
from prismrunner import UniversalOutputParser

raw_llm_output = """
```json
[
  {"condition": "Type 2 Diabetes", "status": "active"},
  {"condition": "Hypertension", "status": "active"}
]
```
"""

parse_result = UniversalOutputParser.parse(
    text=raw_llm_output,
    model_cls=DiagnosticFinding,
)

print(f"Success: {parse_result.success}")
print(f"Parsed {len(parse_result.items)} items: {parse_result.items}")
```

---

### `DeterministicStitcher`
Consolidates chunk-level predictions across parent documents:

```python
from prismrunner import DeterministicStitcher

# For extraction: deduplicates entities case-insensitively across chunks
stitched = DeterministicStitcher.stitch_extraction(raw_chunk_results)

# For classification: majority voting with chronological tie-breaks
voted = DeterministicStitcher.stitch_classification(raw_chunk_results)

# For summarization: chronological section assembly
summaries = DeterministicStitcher.stitch_summarization(raw_chunk_results)
```

---

### Relational Aggregator & Exporters
Turns multi-schema extraction outputs into relational tables, SQLite `.db` databases, and wide tables:

```python
from prismrunner import RelationalNormalizer, SQLiteExporter, WideTableBuilder

# 1. Normalize into relational dictionaries
relational_tables = RelationalNormalizer.normalize_dataset(extraction_records)

# 2. Export to SQLite with foreign keys and indexes
SQLiteExporter.export(relational_tables, "outputs/database.db")

# 3. Export to 1-row-per-document Wide Table (Parquet and CSV)
WideTableBuilder.export(relational_tables, "outputs/wide_table.parquet")
```
