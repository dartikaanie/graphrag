"""Offline tests (mocked client objects, no real API calls) for
llm/client_factory.py's _with_meta variants (Step 5 data retention,
docs/Agent prompt grounding factor ui.md): each must return
content+metadata without crashing when a provider's response object is
missing some fields, and the OLD plain call_llm_*() wrappers must keep
returning exactly the content string, unaffected.
"""

from llm.client_factory import (
    call_llm_anthropic,
    call_llm_anthropic_with_meta,
    call_llm_local,
    call_llm_local_with_meta,
    call_llm_ollama,
    call_llm_ollama_with_meta,
    call_llm_openai,
    call_llm_openai_with_meta,
)

MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hello"},
]


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def model_dump(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class _FakeOpenAIChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeOpenAIResponse:
    def __init__(self, content, with_usage=True, with_fingerprint=True):
        self.id = "resp_abc"
        self.model = "gpt-4o-mini-2024-07-18"
        self.created = 1700000000
        self.system_fingerprint = "fp_xyz" if with_fingerprint else None
        self.choices = [_FakeOpenAIChoice(content)]
        self.usage = _FakeOpenAIUsage() if with_usage else None


class _FakeOpenAIClient:
    def __init__(self, response):
        self._response = response
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def test_openai_with_meta_full_fields():
    client = _FakeOpenAIClient(_FakeOpenAIResponse("hello world"))
    result = call_llm_openai_with_meta(client, MESSAGES, "gpt-4o-mini", temperature=0.1)
    assert result["content"] == "hello world"
    assert result["response_id"] == "resp_abc"
    assert result["response_model"] == "gpt-4o-mini-2024-07-18"
    assert result["response_created"] == 1700000000
    assert result["system_fingerprint"] == "fp_xyz"
    assert result["finish_reason"] == "stop"
    assert result["usage_full"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert result["request_params"] == {"model": "gpt-4o-mini", "temperature": 0.1}


def test_openai_with_meta_missing_usage_and_fingerprint_is_none_not_crash():
    client = _FakeOpenAIClient(_FakeOpenAIResponse("x", with_usage=False, with_fingerprint=False))
    result = call_llm_openai_with_meta(client, MESSAGES, "gpt-4o-mini")
    assert result["usage_full"] is None
    assert result["system_fingerprint"] is None


def test_openai_old_wrapper_returns_only_content_string():
    client = _FakeOpenAIClient(_FakeOpenAIResponse("plain content"))
    result = call_llm_openai(client, MESSAGES, "gpt-4o-mini")
    assert result == "plain content"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _FakeAnthropicUsage:
    def model_dump(self):
        return {"input_tokens": 8, "output_tokens": 4}


class _FakeAnthropicResponse:
    def __init__(self, text, with_usage=True, with_id=True):
        self.content = [type("Block", (), {"text": text})()]
        self.id = "msg_abc" if with_id else None
        self.model = "claude-3-5-sonnet"
        self.stop_reason = "end_turn"
        self.usage = _FakeAnthropicUsage() if with_usage else None


class _FakeAnthropicClient:
    def __init__(self, response):
        self._response = response
        self.messages = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def test_anthropic_with_meta_full_fields():
    client = _FakeAnthropicClient(_FakeAnthropicResponse("claude says hi"))
    result = call_llm_anthropic_with_meta(client, MESSAGES, "claude-3-5-sonnet", temperature=0.2)
    assert result["content"] == "claude says hi"
    assert result["response_id"] == "msg_abc"
    assert result["response_model"] == "claude-3-5-sonnet"
    assert result["response_created"] is None  # Anthropic has no such field, never guessed
    assert result["system_fingerprint"] is None  # OpenAI-only field
    assert result["finish_reason"] == "end_turn"
    assert result["usage_full"] == {"input_tokens": 8, "output_tokens": 4}
    assert result["request_params"]["model"] == "claude-3-5-sonnet"
    assert result["request_params"]["max_tokens"] == 1024


def test_anthropic_with_meta_missing_usage_is_none_not_crash():
    client = _FakeAnthropicClient(_FakeAnthropicResponse("x", with_usage=False))
    result = call_llm_anthropic_with_meta(client, MESSAGES, "claude-3-5-sonnet")
    assert result["usage_full"] is None


def test_anthropic_old_wrapper_returns_only_content_string():
    client = _FakeAnthropicClient(_FakeAnthropicResponse("plain"))
    result = call_llm_anthropic(client, MESSAGES, "claude-3-5-sonnet")
    assert result == "plain"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Local (HF pipeline) -- essentially no structured metadata available at all
# ---------------------------------------------------------------------------

class _FakePipe:
    tokenizer = None  # forces the fallback manual-join path

    def __call__(self, prompt_text, max_new_tokens, do_sample):
        return [{"generated_text": prompt_text + "assistant reply here"}]


def test_local_with_meta_has_only_content_and_model_rest_none():
    pipe = _FakePipe()
    result = call_llm_local_with_meta(pipe, MESSAGES, "llama-2-7b-chat-hf")
    assert result["content"] == "assistant reply here"
    assert result["response_id"] is None
    assert result["response_model"] == "llama-2-7b-chat-hf"
    assert result["response_created"] is None
    assert result["system_fingerprint"] is None
    assert result["finish_reason"] is None
    assert result["usage_full"] is None


def test_local_old_wrapper_returns_only_content_string():
    pipe = _FakePipe()
    result = call_llm_local(pipe, MESSAGES, "llama-2-7b-chat-hf")
    assert result == "assistant reply here"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Ollama -- dev/testing provider, has its own field names for usage
# ---------------------------------------------------------------------------

class _FakeOllamaClient:
    def __init__(self, response):
        self._response = response

    def chat(self, model, messages, options):
        self.last_call = {"model": model, "messages": messages, "options": options}
        return self._response


def test_ollama_with_meta_full_fields():
    client = _FakeOllamaClient({
        "message": {"content": "ollama reply"},
        "model": "phi3:mini",
        "created_at": "2026-10-05T00:00:00Z",
        "done_reason": "stop",
        "prompt_eval_count": 20,
        "eval_count": 15,
    })
    result = call_llm_ollama_with_meta(client, MESSAGES, "phi3:mini", temperature=0.5)
    assert result["content"] == "ollama reply"
    assert result["response_id"] is None  # Ollama has no response id concept
    assert result["response_model"] == "phi3:mini"
    assert result["response_created"] == "2026-10-05T00:00:00Z"
    assert result["system_fingerprint"] is None  # OpenAI-only field
    assert result["finish_reason"] == "stop"
    assert result["usage_full"] == {"prompt_eval_count": 20, "eval_count": 15}
    assert result["request_params"]["options"]["num_ctx"] == 8192


def test_ollama_with_meta_missing_eval_counts_usage_is_none_not_crash():
    client = _FakeOllamaClient({"message": {"content": "x"}, "model": "phi3:mini"})
    result = call_llm_ollama_with_meta(client, MESSAGES, "phi3:mini")
    assert result["usage_full"] is None
    assert result["response_created"] is None
    assert result["finish_reason"] is None


def test_ollama_old_wrapper_returns_only_content_string():
    client = _FakeOllamaClient({"message": {"content": "plain"}, "model": "phi3:mini"})
    result = call_llm_ollama(client, MESSAGES, "phi3:mini")
    assert result == "plain"
    assert isinstance(result, str)
