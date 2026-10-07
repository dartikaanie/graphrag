"""conftest.py -- adds the repo root AND this directory (llm/c_graphrag/)
to sys.path, so tests can `import c_graphrag`/`import embedding_cache`
the same bare way c_graphrag.py's own CLI usage expects (run from within
this directory), while also being able to `from llm.manifest import ...`.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
