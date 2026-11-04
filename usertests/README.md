# PrismRunner User Tests (`usertests/`)

Hands-on scripts and an interactive Jupyter notebook for testing PrismRunner's features.

## 1. Python Scripts (`.py`)

Both scripts run in **instant dry-run mode** by default (no model download required) and support `--live` to run full local LLM inference on your GPU, Apple Silicon MPS, or CPU:

```bash
# Test hardware detection, schema compiler, JSON parser, and quickstart pipeline
python usertests/01_quickstart_and_hardware.py
python usertests/01_quickstart_and_hardware.py --live

# Test Pandas DataFrame extraction, custom classification variables, and sliding-window chunking
python usertests/02_custom_schemas_and_dataframes.py
python usertests/02_custom_schemas_and_dataframes.py --live
```

## 2. Jupyter Notebook (`.ipynb`)

Open [`usertests/03_interactive_walkthrough.ipynb`](03_interactive_walkthrough.ipynb) in Jupyter Lab, Jupyter Notebook, or VS Code / Antigravity to step through hardware advisories, custom Pydantic schemas, parser sanitization, DataFrame extraction, and chunking cell-by-cell.

