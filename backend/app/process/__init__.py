"""Read-only "Process & Parameters" feature (docs/agent_prompt_ui_process_page.md).

Nothing in this package writes to Neo4j, starts a run, or calls an LLM/judge
API. Values are read from the code, config, logs and reports that actually
use them; rationale text is quoted from existing documents (see registry.py).
"""
