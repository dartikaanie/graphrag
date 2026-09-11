"""
llm/
=====================================
Package bersama untuk semua urusan "memanggil LLM" yang dipakai oleh
SELURUH kondisi eksperimen (Kondisi A, B, C):

  - client_factory.py -- get_llm_client() + call_llm_openai/anthropic/
    local/ollama. Satu-satunya tempat yang tahu cara bicara ke tiap
    provider LLM.
  - prompts.py -- build_base_messages() (4-turn chat dialog, replikasi
    persis metodologi paper baseline) dan build_rag_messages() (versi
    Kondisi B/C yang menambahkan konteks retrieval).

Kenapa dipisah ke sini (bukan diduplikasi per kondisi seperti sebelumnya):
  - Tambah provider LLM baru / perbaiki bug cara panggil provider tertentu
    cukup dilakukan 1x di client_factory.py, otomatis kepakai di semua
    kondisi tanpa perlu disinkronkan manual ke tiap file kondisi.
  - Struktur prompt dasar (4-turn) dijamin konsisten di semua kondisi
    karena berasal dari satu fungsi yang sama (build_base_messages),
    bukan disalin-tempel dan berisiko diam-diam berbeda dari waktu ke
    waktu.

Cara pakai dari script kondisi (mis. llm/a_pure_llm/a_*.py):

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from llm.client_factory import get_llm_client
    from llm.prompts import build_base_messages
"""
