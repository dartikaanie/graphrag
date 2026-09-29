"""conftest.py -- adds the repo root to sys.path so tests can
`import llm.prompts` / `from llm.citations import ...` the same way every
condition script does (sys.path.insert(0, repo_root) at import time).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
