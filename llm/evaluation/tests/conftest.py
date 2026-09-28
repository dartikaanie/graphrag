"""
conftest.py -- tambahkan llm/evaluation/ ke sys.path supaya test bisa
`import llm_judge_context_relevance`, `import _judge_common`, dst.
langsung sebagai modul top-level (sama seperti saat script-script itu
dijalankan langsung dari cwd=llm/evaluation/, konsisten dgn cara run.py
memanggilnya). Semua test di sini OFFLINE -- TIDAK ADA panggilan LLM
sungguhan, memakai stub call_fn.
"""

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
