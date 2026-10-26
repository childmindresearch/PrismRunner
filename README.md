# PrismRunner

> **High-throughput, local-first LLM batch execution runtime for structured datasets and Pydantic schemas.**  
> Run completely offline across Apple Silicon (MPS), NVIDIA GPUs (CUDA), and CPU—zero API keys, zero cloud costs, and zero data leakage.

---

## What is PrismRunner?

PrismRunner bridges the gap between raw data files and typed Python data structures. It transforms unstructured text and tabular datasets (**Parquet**, **CSV**, **JSON**, **JSONL**, **TXT**, or **Pandas DataFrames**) into strongly-typed **Pydantic v2 schemas** using local open-weight language models (e.g., Qwen, Llama, Mistral).

Instead of writing fragile custom glue code, prompt engineering hacks, and regex cleaners, PrismRunner provides a production-grade runtime that handles the entire lifecycle:

* **Data Ingestion**: Loads tables, files, or DataFrames with deterministic ID tracking and metadata preservation.
* **Token-Aware Chunking**: Intelligently segments long documents preserving sentence boundaries within your model's safe token budget.
* **Schema-to-Prompt Compiler**: Automatically turns Pydantic models into clear bulleted rules and sample JSON—never leaking raw OpenAPI `$defs` or `$ref` blobs that confuse local models.
* **Native ChatML Batching**: Automatically applies left-padded chat templates with generation prompt alignment for high-throughput GPU/MPS inference.
* **Resilient Output Parsing**: Strips markdown fences, repairs trailing commas, validates models, and purges empty "phantom" hallucinations.
* **Deterministic Stitching**: Recombines multi-chunk outputs into single record-level results via majority voting, entity deduplication, or chronological section assembly.
* **Multi-Schema Database Mapping**: Pre-screens chunks against entire catalogs using lightweight triage and exports relational tables to SQLite, normalized CSVs, and consolidated wide tables.

---

## Quick Start (Under 60 Seconds)

### 1. Installation

```bash
git clone https://github.com/childmindresearch/PrismRunner.git
cd PrismRunner
pip install -e .
```

### 2. Classify text in one command

```bash
prismrunner classify \
  --input-text "Apple unveils new M4 chip with enhanced neural engine tensor cores." \
  --json
```

### 3. Test prompts instantly (Dry Run - No Weights Loaded)

Preview how your schema, variables, and text render into the model prompt without downloading or loading weights into memory:

```bash
prismrunner classify \
  --input-text "Clinical trial demonstrates high efficacy for new hypertension drug." \
  --dry-run
```

---

## How It Works

| Step | What PrismRunner Does | Why It Matters |
| :--- | :--- | :--- |
| **1. Data Ingestion** | Ingests Parquet, CSV, JSON, TXT, or Pandas DataFrames with automatic ID tracking. | No conversion scripts needed; connects directly to your existing data pipelines. |
| **2. Context Chunking** | Sliding-window chunker respecting sentence boundaries and token budgets. | Prevents silent truncation and context window overflow on long documents. |
| **3. Schema Compilation** | Inspects Pydantic v2 classes to generate clean human-readable bullets and sample JSON. | Local models follow clear bulleted constraints far better than complex raw JSON schemas. |
| **4. Left-Padded Batching** | Left-pads tokens and aligns ChatML assistant generation prompts. | Maximizes batch throughput on GPUs/Apple Silicon while eliminating prompt echoing. |
| **5. Robust Output Parser** | Strips markdown fences, fixes trailing commas, and filters phantom objects. | Guarantees valid Pydantic instances or actionable validation error diagnostics. |
| **6. Deterministic Stitcher** | Reassembles multi-chunk outputs via majority voting, deduplication, or section headers. | You get exactly 1 clean, consolidated result per parent record. |

---

## Python API

PrismRunner natively accepts Pandas DataFrames, lists of dictionaries, raw text strings, or file paths with zero boilerplate:

```python
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from prismrunner import PrismPipeline

# 1. Define any Pydantic schema
class TechAssessment(BaseModel):
    architecture: str = Field(description="Primary system architecture")
    maturity_level: Literal["experimental", "beta", "production"] = Field(description="Deployment maturity")
    score: float = Field(ge=0.0, le=10.0, description="Readiness score from 0 to 10")

# 2. Ingest directly from Pandas DataFrame or in-memory data
df = pd.DataFrame({
    "service_id": ["SRV-01", "SRV-02"],
    "system_report": [
        "Migrated monolith to event-driven microservices using Kafka. Production ready.",
        "Prototyping serverless edge functions for latency-critical routing.",
    ]
})

# 3. Run pipeline (auto-detects Apple Silicon MPS, NVIDIA CUDA, or CPU)
pipeline = PrismPipeline()
results = pipeline.run(
    task="extract",
    input_data=df,
    text_column="system_report",
    id_column="service_id",
    schema_cls=TechAssessment,
    batch_size=4,
)

print(results)
```

---

## Core Capabilities

### 1. Document Classification (`prismrunner classify`)
Classify tabular rows or text records into designated categories with confidence scoring and reasoning extraction. When records span multiple chunks, PrismRunner deterministically aggregates predictions via majority voting:

```bash
prismrunner classify \
  --input-path data/articles.parquet \
  --column text \
  --id-column article_id \
  --batch-size 8 \
  --output-path outputs/classified.json
```

### 2. Entity & Item Extraction (`prismrunner extract`)
Extract entities, action items, or custom structured models with automatic cross-chunk deduplication:

```bash
prismrunner extract \
  --input-path data/transcripts.csv \
  --column conversation \
  --schema "schemas/clinical_records.py:Medication" \
  --chunk \
  --chunk-size 1024 \
  --output-path outputs/medications.json
```

### 3. Document Summarization (`prismrunner summarize`)
Generate structured summaries and key takeaways across lengthy documents. Single-chunk documents return clean text; multi-chunk documents are chronologically ordered under section headers:

```bash
prismrunner summarize \
  --input-path data/reports.json \
  --column content \
  --chunk \
  --chunk-size 1500 \
  --output-format csv \
  --output-path outputs/summaries.csv
```

### 4. Multi-Schema Database Mapping (`prismrunner map`)
Extract across an entire catalog of Pydantic models simultaneously. PrismRunner runs fast relevance triage to skip irrelevant schemas, tracks progress with fault-tolerant checkpointing, and exports relational outputs into:
* A queryable **SQLite database** with foreign keys (`source_id`)
* Normalized **relational CSVs**
* A consolidated **1-row-per-document Wide Table** (Parquet & CSV)

```bash
prismrunner map \
  --input-path data/clinical_notes.parquet \
  --schemas-dir schemas/ \
  --output-dir outputs/clinical_run/ \
  --export sqlite,csv,wide \
  --batch-size 8
```

### 5. Hardware & Token Budget Advisor (`prismrunner inspect-model`)
Inspect any Hugging Face model repository to see its context limit, available VRAM, and recommended safe chunk sizes:

```bash
prismrunner inspect-model --model Qwen/Qwen2.5-3B-Instruct --device auto
```

---

## 📚 User Documentation & Guides

Comprehensive guides and interactive recipes are available in the [**`docs/`**](docs/README.md) directory:

* [**01. Prompt Customization Guide**](docs/01_prompt_customization.md) — Overriding templates, prompt variables, and system prompts.
* [**02. Jupyter Notebook Guide**](docs/02_jupyter_notebooks.md) — Interactive workflows, in-cell Pydantic schemas, and DataFrame recipes.
* [**03. Python Package API Reference**](docs/03_python_package_usage.md) — Complete SDK reference for `PrismPipeline` and `SchemaMappingPipeline`.
* [**04. Standalone & CLI Reference**](docs/04_standalone_and_cli.md) — Command flags, recipes, and standalone `main.py` execution.
* [**05. Custom Schemas & Catalogs**](docs/05_custom_schemas_and_catalogs.md) — Defining schemas, multi-schema catalogs, and SQLite/wide-table exports.
* [**06. Configuration & Hardware Tuning**](docs/06_configuration_and_hardware.md) — Optimizing MPS, CUDA, CPU, VRAM batch sizing, and token budgets.

---

## Configuration (`config/pipeline_config.yaml`)

Default pipeline settings can be customized in `config/pipeline_config.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  device: "auto"             # auto, mps, cuda, cpu
  torch_dtype: "auto"        # auto, float16, bfloat16, float32
  load_in_4bit: false
  load_in_8bit: false

inference:
  batch_size: 4
  max_new_tokens: 512
  temperature: 0.0
  system_prompt: "You are PrismRunner, an accurate local inference engine. Follow structured format instructions strictly."

chunking:
  enabled: false
  chunk_size: 1024
  chunk_overlap: 64
  strategy: "sentence"       # sentence or token
```

---

## License

MIT
