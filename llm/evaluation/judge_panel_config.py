"""
judge_panel_config.py
---------------------
The fixed LLM-as-Judge panel (docs/DECISION_C_SCORING.md, Amendment 12).
Three judges from different model families, all served by DeepInfra's
OpenAI-compatible API. gpt-4o-mini is the generator only and is not a judge
(self-preference bias, Verga et al. 2024). The composition is fixed before
any judge run on calibration items and is never changed based on results.

Every judge uses the same frozen judge prompt (judge_prompt_v2_2:
PROMPT_VERSION, temperature 0, max_tokens), the same input template and the
same parser. No reasoning/thinking mode.

Model verification (2026-10-19, `client.models.list()` + live test calls):
  - deepseek-ai/DeepSeek-V3.2 and Qwen/Qwen2.5-72B-Instruct are listed under
    these exact ids.
  - meta-llama/Llama-3.3-70B-Instruct is NOT listed; DeepInfra lists only
    meta-llama/Llama-3.3-70B-Instruct-Turbo (FP8). The panel requests that
    explicit id (decision 2026-10-19); every record is checked against
    expected_served_model and flagged if another model answers.
  - DeepSeek-V3.2 thinking: DeepInfra controls reasoning with the request
    parameter `reasoning_effort` (docs.deepinfra.com/chat/reasoning; "none"
    disables it). Test calls: "high" -> reasoning_content of 1,061 chars;
    "none" and the default -> no reasoning_content, no <think> block. The
    panel sends "none" explicitly and checks every response (judge_panel.py:
    reasoning_present()).

Prices are USD per 1M tokens, from the DeepInfra model listing
(`models.list()` metadata.pricing), checked 2026-10-19.
"""

from dataclasses import dataclass, field

from judge_prompt_v2_2 import PROMPT_VERSION, REQUEST_MAX_TOKENS, REQUEST_TEMPERATURE

PANEL_ID = "panel-v1"
PROVIDER = "deepinfra"
BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY_ENV = "DEEPINFRA_API_KEY"
PRICE_DATE = "2026-10-19"

JUDGE_PROMPT_VERSION = PROMPT_VERSION      # "judge-v2.2", frozen; the same for all three judges
TEMPERATURE = REQUEST_TEMPERATURE          # 0.0
MAX_TOKENS = REQUEST_MAX_TOKENS            # 2000, the current judge value

# Retries (Amendment 12 task spec): up to 2 retries on a network error or a parse failure.
MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 6


@dataclass(frozen=True)
class PanelJudge:
    judge_id: str                  # short id stored on every record
    model: str                     # the id sent in the request
    family: str
    price_per_m_input: float
    price_per_m_output: float
    expected_served_model: str     # response.model the provider is expected to return
    extra_body: dict = field(default_factory=dict)  # request parameters beyond the OpenAI ones
    thinking_disabled: bool = True
    precision_note: str = ""
    provider: str = PROVIDER
    base_url: str = BASE_URL
    api_key_env: str = API_KEY_ENV
    price_date: str = PRICE_DATE


PANEL_JUDGES: tuple[PanelJudge, ...] = (
    PanelJudge(
        judge_id="llama", model="meta-llama/Llama-3.3-70B-Instruct-Turbo", family="Llama (Meta)",
        price_per_m_input=0.10, price_per_m_output=0.32,
        expected_served_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        precision_note="served by DeepInfra in FP8 quantization (-Turbo; the only Llama-3.3-70B variant listed)",
    ),
    PanelJudge(
        judge_id="deepseek", model="deepseek-ai/DeepSeek-V3.2", family="DeepSeek",
        price_per_m_input=0.26, price_per_m_output=0.38,
        expected_served_model="deepseek-ai/DeepSeek-V3.2",
        extra_body={"reasoning_effort": "none"},  # non-thinking mode, see the module docstring
    ),
    PanelJudge(
        judge_id="qwen", model="Qwen/Qwen2.5-72B-Instruct", family="Qwen (Alibaba)",
        price_per_m_input=0.36, price_per_m_output=0.40,
        expected_served_model="Qwen/Qwen2.5-72B-Instruct",
    ),
)

JUDGES_BY_ID = {j.judge_id: j for j in PANEL_JUDGES}
JUDGES_BY_MODEL = {j.model: j for j in PANEL_JUDGES}


def judge_cost_usd(judge: PanelJudge, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    return round(prompt_tokens / 1e6 * judge.price_per_m_input + completion_tokens / 1e6 * judge.price_per_m_output, 8)


def panel_public_view() -> dict:
    """The panel configuration as recorded in manifests (no keys)."""
    return {
        "panel_id": PANEL_ID, "provider": PROVIDER, "base_url": BASE_URL, "prompt_version": JUDGE_PROMPT_VERSION,
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "max_retries": MAX_RETRIES, "price_date": PRICE_DATE,
        "judges": [{"judge_id": j.judge_id, "model": j.model, "family": j.family,
                    "expected_served_model": j.expected_served_model, "extra_body": j.extra_body,
                    "thinking_disabled": j.thinking_disabled, "price_per_m_input": j.price_per_m_input,
                    "price_per_m_output": j.price_per_m_output, "precision_note": j.precision_note}
                   for j in PANEL_JUDGES],
    }
