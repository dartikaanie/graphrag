"""conftest.py -- adds the repo root AND llm/d_lightrag/ to sys.path so tests can
`import d_lightrag` the bare way the CLI does, and `from llm... import ...`."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
