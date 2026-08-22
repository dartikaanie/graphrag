"""
TEST KONEKSI MODEL LLM
========================
    python test_llm_connection.py                          # pakai config dari .env
    python test_llm_connection.py --provider anthropic --model claude-sonnet-4-5
    python test_llm_connection.py --n-test 3                # kirim 3 prompt uji berbeda
"""

import argparse
import os
import sys
import anthropic
import ollama
import ollama as ollama_sdk

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TEST_PROMPTS = [
    "Reply with exactly one word: OK",
    "What is 2+2? Reply with just the number.",
    "Name one popular Python web framework in one word.",
]


def call_openai(prompt: str, model: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens


def call_anthropic(prompt: str, model: str):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


def call_ollama(prompt: str, model: str):
    #qwen2.5:1.5b
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    reply = response["message"]["content"]
    tin = response.get("prompt_eval_count", 0)
    tout = response.get("eval_count", 0)
    return reply, tin, tout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "ollama"),
                         choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "all-MiniLM-L6-v2"))
    parser.add_argument("--n-test", type=int, default=1, help="Jumlah prompt uji dikirim ")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST KONEKSI MODEL LLM")
    print("=" * 70)
    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")

    if args.provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        key_env_name = "OPENAI_API_KEY"
        call_fn = call_openai
        if not key:
            print(f"\n[FAIL] {key_env_name} tidak ditemukan di .env")
            sys.exit(1)
    elif args.provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        key_env_name = "ANTHROPIC_API_KEY"
        call_fn = call_anthropic
        if not key:
            print(f"\n[FAIL] {key_env_name} tidak ditemukan di .env")
            sys.exit(1)
    elif args.provider == "ollama":
        call_fn = call_ollama
        try:
            ollama_sdk.list()
            print("Ollama server: [OK] terdeteksi di localhost:11434")
        except Exception as e:
            print(f"\n[FAIL] Tidak bisa konek ke Ollama -- pastikan sudah jalan "
                  f"'ollama serve' dan model '{args.model}' sudah di-pull "
                  f"(ollama pull {args.model}). Detail: {e}")
            sys.exit(1)
        key = None
    else:
        print(f"\n[FAIL] Tidak bisa konek ke provider {args.provider} -- hanya support openai, anthropic, ollama")
        sys.exit(1)

    n = min(args.n_test, len(TEST_PROMPTS))
    print(f"\nMengirim {n} prompt uji...\n")

    total_in, total_out = 0, 0
    n_success = 0

    for i, prompt in enumerate(TEST_PROMPTS[:n], 1):
        print(f"[{i}/{n}] Prompt: {prompt!r}")
        try:
            reply, tin, tout = call_fn(prompt, args.model)
            total_in += tin
            total_out += tout
            n_success += 1
            print(f"      [OK] Response: {reply!r}")
            print(f"      Token: {tin} in / {tout} out")
        except Exception as e:
            print(f"      [FAIL] {e}")
        print()

    print("=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"Berhasil: {n_success}/{n}")
    if n_success > 0:
        print(f"Total token terpakai: {total_in} in / {total_out} out")
        print(f"\n[OK] Model LLM berfungsi.")
    else:
        print(f"\n[FAIL] Semua percobaan gagal")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())