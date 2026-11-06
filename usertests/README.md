# PrismRunner User Tests (`usertests/`)

Hands-on scripts and an interactive Jupyter notebook for testing PrismRunner's features.

## 1. Python Scripts (`.py`)

All scripts run in **instant dry-run mode** by default (no model download required) and support `--live` (or `DRY_RUN = False` in `SemanticPrism1.py`) to run full local LLM inference on your NVIDIA CUDA GPU, Apple Silicon MPS, or CPU:

```bash
# Test hardware detection, schema compiler, JSON parser, and quickstart pipeline
python usertests/01_quickstart_and_hardware.py
python usertests/01_quickstart_and_hardware.py --live

# Test Pandas DataFrame extraction, custom classification variables, and sliding-window chunking
python usertests/02_custom_schemas_and_dataframes.py
python usertests/02_custom_schemas_and_dataframes.py --live

# Run the SemanticPrism Stage 1 Extraction pipeline (Theme Discovery, Master Synthesis, and SVO Triples)
python usertests/SemanticPrism1.py

# Run the SemanticPrism Stage 2 Refinement pipeline (Two-Tier SVO Clustering, Taxonomic Hypernym Lifting, and Contextual Fidelity)
python usertests/SemanticPrism2.py
python usertests/SemanticPrism2.py --fidelity-only

# Run the SemanticPrism Stage 4 Synthesis pipeline (Target Resolution, Enum Clustering, Community Schema Synthesis, and Master Ontology Consolidation)
python usertests/SemanticPrism4.py
python usertests/SemanticPrism4.py --dry-run
```

### `SemanticPrism1.py` Highlights

- **100% Local `PrismRunner` Execution**: Runs Phase 1 (Global Theme Discovery), Phase 1.5 (Theme Aggregation), Phase 2 (Empirical Graph Clustering + Map-Reduce Master Theme Synthesis), and Phase 3 (Master-Guided SVO Triple Extraction) without any external server dependencies.
- **Continuous Device-Relative Auto-Tuning**:
  - `DEVICE = "auto"`, `TORCH_DTYPE = "auto"`, `ATTN_IMPLEMENTATION = "auto"`, `BATCH_SIZE = "auto"`, and `CONTEXT_WINDOW_CAP = "auto"` adapt automatically to any GPU, Apple Silicon Mac, or CPU host.
  - Dynamically bounds stage chunk word counts to the loaded model's context window (`_resolve_stage_max_words`), offloads the `SentenceTransformer` embedding model immediately after Phase 2 clustering to reclaim 100% of VRAM for the LLM, and automatically splits batches on OOM without losing progress.
- **Incremental Checkpointing**: Saves per-document `{doc}_themes.json` and `{doc}_triples.json` files as soon as each document's chunks finish and logs timestamped progress to the console (`PRINT_CHECKPOINTS = True`).

## 2. Jupyter Notebook (`.ipynb`)

Open [`usertests/03_interactive_walkthrough.ipynb`](03_interactive_walkthrough.ipynb) in Jupyter Lab, Jupyter Notebook, or VS Code / Antigravity to step through hardware advisories, custom Pydantic schemas, parser sanitization, DataFrame extraction, and chunking cell-by-cell.
