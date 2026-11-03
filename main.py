#!/usr/bin/env python3
"""PrismRunner standalone runfile.

Can be run directly via:
    python3 main.py --help
    python3 main.py classify --input-text "..." --dry-run
    python3 main.py map -i data/clinical_notes.parquet -s schemas/ --dry-run
"""

import sys
from pathlib import Path

# Add project root to sys.path so it works out-of-the-box without prior pip install
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prismrunner.cli import main

if __name__ == "__main__":
    main()
