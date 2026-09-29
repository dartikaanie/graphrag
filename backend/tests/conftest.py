"""conftest.py -- adds backend/ to sys.path so tests can `import app.*`
directly, matching how uvicorn imports it (backend/ as the working dir /
package root). No LLM calls, no live Neo4j/DuckDB connections in any test
here -- everything is fixture JSON/JSONL read from tmp_path.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
