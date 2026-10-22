# Standalone Script & CLI Reference

PrismRunner can be executed directly from your terminal as a standalone script or command-line utility for automated batch jobs, cron tasks, and shell scripts.

---

## 1. Running as a Standalone Script or CLI

You can invoke PrismRunner via any of the following entrypoints:

```bash
# Direct standalone execution
python main.py --help

# Python module execution
python -m prismrunner --help

# Installed CLI binary
prismrunner --help
```

---

## 2. CLI Commands Overview

```text
Usage: prismrunner [OPTIONS] COMMAND [ARGS]...

  PrismRunner: High-throughput local LLM batch inference runtime.

Commands:
  inspect-model  Inspect model architecture, context window, and safe chunking budget.
  classify       Classify dataset rows or text into designated categories.
  extract        Extract structured items and entities conforming to Pydantic schemas.
  summarize      Generate faithful summaries across single or multi-chunk documents.
  map            Execute multi-schema triage, batch extraction, and relational export.
```

---

## 3. Command Reference

### `inspect-model`
Calculates context capacity and hardware utilization:
```bash
prismrunner inspect-model \
  --model Qwen/Qwen2.5-3B-Instruct \
  --device auto
```
**Output:**
- Detected hardware accelerator (MPS, CUDA, or CPU)
- Native model context limit (e.g. 32,768 tokens)
- Recommended safe chunking budget (accounting for system prompts, schema overhead, and generation tokens)

---

### `classify`
Classify tabular datasets or ad-hoc strings:
```bash
# Tabular batch classification
prismrunner classify \
  --input-path data/articles.parquet \
  --column body \
  --id-column article_id \
  --var categories="Finance, Technology, Health, Politics" \
  --batch-size 8 \
  --output-path outputs/classified.json

# Ad-hoc text classification with JSON console output
prismrunner classify \
  --input-text "Apple unveils new M4 Max chip with enhanced neural engine." \
  --var categories="Hardware, Software, Markets" \
  --json
```

---

### `extract`
Extract structured entities into schemas:
```bash
# Extract with custom Pydantic schema file and chunking enabled
prismrunner extract \
  --input-path data/clinical_notes.csv \
  --column note_text \
  --id-column note_id \
  --schema schemas/clinical_records.py:Medication \
  --chunk \
  --chunk-size 1024 \
  --chunk-overlap 64 \
  --batch-size 4 \
  --output-path outputs/medications.json
```

#### Custom Instruction and Template Overrides:
```bash
prismrunner extract \
  --input-path data/reports.parquet \
  --instruction "Extract only active high-priority action items." \
  --var guidelines="Include assigned owner and estimated completion date." \
  --output-path outputs/action_items.json
```

---

### `summarize`
Summarize single or multi-part documents:
```bash
prismrunner summarize \
  --input-path data/transcripts.json \
  --column dialogue \
  --chunk \
  --chunk-size 1500 \
  --output-format csv \
  --output-path outputs/transcripts_summary.csv
```

---

### `map` (SchemaMap Mode)
Execute multi-schema catalog extraction with fast ChatML triage, checkpoint resume, and relational export:
```bash
prismrunner map \
  --input-path data/clinical_corpus.parquet \
  --column notes \
  --id-column patient_id \
  --schemas-dir schemas/clinical/ \
  --output-dir outputs/clinical_run/ \
  --export sqlite,csv,wide \
  --batch-size 8
```

**Generated Outputs in `outputs/clinical_run/`:**
- `extracted_database.db`: SQLite database with typed tables, foreign keys, and indexes.
- `relational_csv/`: Folder containing normalized CSV tables (`Diagnosis.csv`, `Medication.csv`, `LabResult.csv`).
- `consolidated_wide_table.parquet`: 1-row-per-document denormalized wide table.
- `.checkpoint_state.json`: Granular checkpoint record. If interrupted, re-running the same command resumes automatically.

---

## 4. Common CLI Options & Flags

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--input-path`, `-i` | File Path | Path to Parquet, CSV, JSON, TXT, or Markdown dataset. |
| `--input-text`, `-t` | String | Raw text string for quick ad-hoc testing. |
| `--column`, `--text-column` | String | Name of the text column to process. |
| `--id-column` | String | Name of the unique identifier column. |
| `--metadata-columns` | String | Comma-separated list of metadata columns to preserve. |
| `--exclude-columns` | String | Comma-separated list of columns to exclude from metadata. |
| `--schema`, `-s` | String | Path to custom schema (`file.py:ClassName`). |
| `--prompt-file` | File Path | Custom YAML prompt template path. |
| `--instruction` | String | Task instruction override. |
| `--system-prompt` | String | Custom system prompt override for chatML format. |
| `--as-array / --single-object` | Flag | Toggle whether model outputs JSON array or single object. |
| `--dedup / --no-dedup` | Flag | Enable/disable post-inference deduplication across chunks. |
| `--dedup-key` | String | Key or field name for deduplicating extracted items. |
| `--category-field` | String | Custom field name for classification category/label. |
| `--confidence-field` | String | Custom field name for classification confidence score. |
| `--reasoning-field` | String | Custom field name for classification justification. |
| `--summary-field` | String | Custom field name for narrative summary text. |
| `--points-field` | String | Custom field name for bulleted takeaways list. |
| `--var` | Key=Val | Custom variable override (can be repeated). |
| `--batch-size`, `-b` | Integer | Inference batch size (default: 4). |
| `--max-tokens` | Integer | Maximum new tokens to generate (default: 512). |
| `--temperature` | Float | Sampling temperature (0.0 for deterministic). |
| `--top-p` | Float | Nucleus sampling probability threshold. |
| `--top-k` | Integer | Top-k sampling threshold. |
| `--repetition-penalty` | Float | Penalize repetitive token generation. |
| `--chunk / --no-chunk` | Boolean | Enable/disable token sliding-window chunking. |
| `--chunk-size` | Integer | Max token size per chunk (default: 1024). |
| `--chunk-overlap` | Integer | Token overlap between chunks (default: 64). |
| `--chunk-strategy` | Choice | `sentence` or `token` boundary strategy. |
| `--dry-run` | Flag | Preview rendered prompts without loading model. |
| `--output-path`, `-o` | File Path | Destination file for results. |
| `--output-format` | String | `json`, `csv`, or `parquet`. |
| `--device` | String | `auto`, `mps`, `cuda`, or `cpu`. |
| `--json` | Flag | Output raw JSON directly to stdout for Unix pipes. |

### SchemaMap `map` Options

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--id-column-name` | String | Foreign key column name linking parent records (default: `source_id`). |
| `--wide-table-name` | String | Base filename for denormalized wide table (default: `consolidated_wide_table`). |
| `--sqlite-db-name` | String | SQLite database filename (default: `extracted_database.db`). |
| `--wide-separator` | String | Delimiter for joined multi-value lists in wide table (default: `; `). |
| `--triage-max-tokens` | Integer | Maximum tokens generated during relevance triage pass (default: 256). |


---

## 5. Dry-Run Verification

Before submitting long-running batch jobs, verify prompt rendering and schema compilation using `--dry-run`:

```bash
prismrunner extract \
  --input-text "Patient prescribed 50mg Lisinopril daily." \
  --schema schemas/clinical_records.py:Medication \
  --instruction "Extract active cardiovascular medications." \
  --dry-run
```
