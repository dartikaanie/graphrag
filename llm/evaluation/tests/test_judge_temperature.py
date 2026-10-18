"""Every judge uses judge_clients.JUDGE_TEMPERATURE (0.0); generators send
no temperature at all."""

import inspect
from types import SimpleNamespace

import judge_clients
import judge_prompt_v2_1
import llm_judge_context_relevance_v1
import llm_judge_hallucination_v1
import llm_judge_hallucination_v2
from _judge_common import call_judge_llm_with_retry

from llm.client_factory import call_llm_openai_with_meta


def test_judge_temperature_is_zero():
    assert judge_clients.JUDGE_TEMPERATURE == 0.0


def test_official_judges_use_judge_temperature():
    for mod in (llm_judge_hallucination_v1, llm_judge_hallucination_v2, llm_judge_context_relevance_v1):
        assert mod.REQUEST_TEMPERATURE == judge_clients.JUDGE_TEMPERATURE, mod.__name__
    # pre-registered literal (Amendment 4 rule 7) -- must not drift from the constant
    assert judge_prompt_v2_1.REQUEST_TEMPERATURE == judge_clients.JUDGE_TEMPERATURE


def test_shared_judge_call_defaults_to_judge_temperature():
    default = inspect.signature(call_judge_llm_with_retry).parameters["request_temperature"].default
    assert default == judge_clients.JUDGE_TEMPERATURE


def test_legacy_judges_default_to_judge_temperature(monkeypatch):
    import llm_judge_answer_relevance
    import llm_judge_context_relevance
    import llm_judge_hallucination

    for mod in (llm_judge_hallucination, llm_judge_answer_relevance, llm_judge_context_relevance):
        assert mod.JUDGE_TEMPERATURE == judge_clients.JUDGE_TEMPERATURE, mod.__name__
        assert 'os.getenv("JUDGE_TEMPERATURE", JUDGE_TEMPERATURE)' in inspect.getsource(mod.main), mod.__name__


class _FakeOpenAI:
    def __init__(self):
        self.sent = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent = kwargs
        return SimpleNamespace(
            id="r1", model="m", created=0, system_fingerprint=None, usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        )


def test_zero_temperature_is_sent_not_dropped():
    # legacy judges go through client_factory: 0.0 is falsy, must still be sent
    client = _FakeOpenAI()
    result = call_llm_openai_with_meta(client, [{"role": "user", "content": "x"}], "m", temperature=0.0)
    assert client.sent["temperature"] == 0.0
    assert result["request_params"]["temperature"] == 0.0


def test_generator_call_sends_no_temperature():
    client = _FakeOpenAI()
    result = call_llm_openai_with_meta(client, [{"role": "user", "content": "x"}], "m")
    assert "temperature" not in client.sent
    assert result["request_params"]["temperature"] is None  # recorded explicitly as "not sent"
