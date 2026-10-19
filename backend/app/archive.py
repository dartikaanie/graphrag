"""Bridge to llm/archive_paths.py, the resolver for the pre-official archive
(2026-10-19; results/_archive/2026-10-19_pre-official/ARCHIVE_INDEX.md).
The dashboard shows official runs only; archived data is read through this
module, and only behind an explicit include_archive / "show archive" toggle
(the Process page's stage-1 record is the one labelled exception)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm import archive_paths  # noqa: E402,F401
