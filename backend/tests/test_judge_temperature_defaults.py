"""settings_service's judge_temperature default mirrors
llm/evaluation/judge_clients.JUDGE_TEMPERATURE. Read with `ast`, not
imported: judge_clients calls load_dotenv() on import."""

import ast
from pathlib import Path

from app.services import settings_service

JUDGE_CLIENTS = Path(__file__).resolve().parents[2] / "llm" / "evaluation" / "judge_clients.py"


def _judge_temperature_constant() -> float:
    tree = ast.parse(JUDGE_CLIENTS.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "JUDGE_TEMPERATURE" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("JUDGE_TEMPERATURE not found in judge_clients.py")


def test_settings_default_matches_judge_temperature():
    assert settings_service.DEFAULTS["judge_temperature"] == _judge_temperature_constant() == 0.0
