# IMPLEMENTATION_STATUS — Implementasi GraphRAG saat ini

**Judul tesis:** *Pengembangan GraphRAG Framework Berbasis Pengetahuan Komunitas Stack Overflow untuk Mereduksi Halusinasi pada LLM* (ITB, Informatika, S2)

Dokumen ini menjelaskan kondisi implementasi sekarang, diverifikasi ke kode dan dokumen keputusan di repo (2026-10-25, branch `master`, HEAD `00ad31c` ditambah perubahan working tree yang belum di-commit). Sumber ditulis `path` atau `path:baris`. Tidak ada API key atau secret di dokumen ini.

---

## 1. Ringkasan

1. **Empat kondisi:** A (LLM murni), B (RAG dense retrieval), C (GraphRAG trust-weighted, kontribusi utama), D (dual-level retrieval, adaptasi LightRAG). B, C, dan D memakai prompt 4-turn yang sama (`llm/prompts.py`), model embedding `all-MiniLM-L6-v2`, generator `gpt-4o-mini`, dan sampel yang sama.
2. **Aturan leakage aktif: mode `thread`** (Amendment 15, status **DRAFT**): hanya pertanyaan evaluasi dan semua jawabannya yang dikecualikan. Duplicate, Linked, dan near-duplicate **tidak** dikecualikan. Alasan desain di Amendment 15 masih `TODO(author)`.
3. **Status eksperimen:** stage 1 (pemilihan α, retrieval-only, dev set) selesai. **Stage 2 dan run resmi R1–R9 belum dijalankan.** `results/run_registry.json` berisi 133 entri, semuanya `archived`; tidak ada `official` atau `dev-stage2`.
4. **Dua pagar yang sengaja menahan run resmi:** Kondisi C pada test sample berhenti (exit 2) selama `official_params.json` berstatus `pending_selection`; judging resmi ditolak selama `AMENDMENT14_APPROVED = False` dan `OFFICIAL_ALLOWED = ()` (`llm/evaluation/judge_versions.py`). Amendment 14 masih draf yang belum di-commit (`reports/amendment14_draft.md`).
5. **Parameter dibekukan lewat tabel yang dihasilkan skrip** (`llm/frozen_parameters.py` → `docs/FROZEN_PARAMETERS.md`, `results/frozen_parameters.json`, hash tabel `8e84ee49913d`). Status "frozen (Amendment 14 draft)" baru mengikat setelah Amendment 14 disetujui.

---

## 2. Konfigurasi resmi (R1–R9)

Sumber: `llm/official_run_matrix.py`, `docs/OFFICIAL_RUN_MATRIX.md`.

| Parameter | Nilai |
|---|---|
| Sampel uji | n = 384, seed 42, pool 1536 (1471 setelah filter token 2048), posisi 1–384; daftar ID beku `llm/sample_ids/test_ids_v1.json` |
| Dev set | n = 50, posisi 385–434; `llm/sample_ids/dev_ids_v1.json` |
| Generator | `openai` / `gpt-4o-mini` |
| top-k, chunk | 5, 400 token (tiktoken `cl100k_base`) |
| C | `n_anchor` 3, `n_semantic_expansion` 3, retrieval v3, `fusion_mode trust_weighted`, expansion answerable-only |
| D | `n_low_level` 3, `n_high_level` 3 |
| Leakage | `thread` |

Matriks: R1 A · R2 B-plain · R3 B-grounded · R4 C-primary-plain · R5 C-primary-grounded · R6 C-ablation-plain · R7 C-ablation-grounded · R8 D-plain · R9 D-grounded. "Grounded" berarti instruksi grounding + sitasi; "plain" hanya sitasi. C-primary dan C-ablation berbeda hanya di α (Amendment 10).

---

## 3. Arsitektur per kondisi

### 3.1 Kondisi A — LLM murni
`llm/a_pure_llm/a_baseline_replication.py`. Tanpa retrieval; replikasi protokol Da Silva, Samhi & Khomh (2025). Dasar pembanding.

### 3.2 Kondisi B — RAG dense retrieval
`llm/b_rag/b_condition_b_rag.py`, `llm/b_rag/kg_corpus.py`, `llm/b_rag/build_kg_answer_index.py`.
- Korpus `kg_answers`: seluruh 725.795 Answer di KG, satu vektor per jawaban dengan teks `Q: <judul pertanyaan>\nA: <body jawaban>`, dipotong ke 400 token (teks yang sama dengan yang dilihat C per kandidat).
- Index FAISS `IndexFlatIP` (exact) atas embedding ternormalisasi (= cosine). Tanpa graf, relasi, atau trust.
- Query = `Title + " " + Body[:1000]` (`llm/common/text.py`), sama dengan C.
- Neo4j hanya dipakai offline untuk ekspor korpus; saat runtime B membaca index FAISS dan `kg_answers.parquet`.
- Mode `--corpus legacy_sample` hanya untuk reproduksi run lama.

### 3.3 Kondisi C — GraphRAG trust-weighted (retrieval v3)
`llm/c_graphrag/c_graphrag.py`. Membaca KG Neo4j dan index FAISS yang sudah ada; tidak membangun apa pun.
1. **Entity anchoring:** vector search atas embedding Title+Body pertanyaan, top-`n_anchor` Question (id pertanyaan evaluasi dibuang).
2. **Graph traversal:** 1-hop (jawaban langsung anchor via `HAS_ACCEPTED_ANSWER`/`HAS_ANSWER`) dan 2-hop (anchor → Question terkait via `IS_RELATED_TO`/`TAG_COOCCUR`/`EMBED_SIM` → jawabannya, `LIMIT 50`). Tie-break `a.id ASC`.
3. **Semantic expansion** (Amendment 9): per seed, `n_semantic_expansion` Question paling mirip yang masih punya ≥ 1 Answer yang tidak dikecualikan (over-fetch faktor 20, `EXPANSION_OVERFETCH`). Setting `EXPANSION_REQUIRE_ANSWER` default True.
4. **Fusi dan ranking v3:** dedup per `answer_id`, `final = (1 − α)·sim_norm + α·trust`; `sim` = cosine antara query dan teks kandidat yang dilihat LLM, `sim_norm` = min-max per pertanyaan, `trust` = 0,7·path trust + 0,3·trustScore Answer (kandidat expansion dibatasi 0,4). Urut `final` menurun, tie-break `answer_id` naik, ambil top-5 (`docs/DECISION_C_SCORING.md`).
5. **Switch eksploratif** (tidak dipakai memilih α; resmi memakai default): `max_hops = 2`, kelima tipe edge, `use_author_trust = false`, `accepted_only = false` (`llm/c_graphrag/official_params.json`).

**trustScore v3-logscore** (Amendment 5–6, terpasang di Neo4j): Answer = 0,5·`norm_log(score; P99)` + 0,3·`is_accepted` + 0,2·`w_auth` (bobot reputasi `AUTHOR_TRUST`); Question = 0,7·`norm_log(score)` + 0,3·`minmax(log1p(views))`. Run C mencatat `trustScoreVersion` dari graf; `stage2_report.py` menolak run yang tidak mencatat `v3-logscore`.

**KG:** 2.685.814 Question, 725.795 Answer, 602.900 Question dengan ≥ 1 Answer.

**α:** kandidat stage 2 adalah **0,25 dan 0** (`STAGE2_ALPHAS = (0.25, 0.0)`, `llm/c_graphrag/stage2_report.py`), dari stage 1 yang dihitung ulang di Amendment 9 (% pertanyaan dengan ≥ 1 konteks RELEVANT: 48,0 untuk α 0,25 dan 50,0 untuk α 0). `alpha_primary` dan `alpha_ablation` di `official_params.json` masih `null`; keduanya diisi oleh hasil stage 2 (Amendment 10).

### 3.4 Kondisi D — dual-level retrieval
`llm/d_lightrag/d_lightrag.py` (`D_RETRIEVAL_VERSION = "v3"`). Memakai KG dan FAISS yang sama dengan C; bukan LightRAG asli (tanpa ekstraksi entitas oleh LLM).
- **Low-level:** anchor yang sama dengan C (`n_low_level`), jawaban 1-hop; skor = cosine anchor.
- **High-level:** dari anchor yang sama, 2-hop via `IS_RELATED_TO`/`TAG_COOCCUR`/`EMBED_SIM` ke Question lain yang punya jawaban; skor = bobot edge mentah.
- **Fusi:** dedup per `answer_id` (skor relevansi tertinggi), urut menurun, top-5. trustScore tidak ikut menentukan skor. C vs D adalah ablasi mekanisme retrieval pada graf yang sama.

### 3.5 Provenance per item
Record B/C/D menyimpan jalur retrieval tiap item (`provenance_version = "v3"`, `llm/common/item_provenance.py`), ikut dalam `config_hash`. Pencarian atas permintaan lewat `llm/provenance_lookup.py` dan tombol "Show provenance" di dashboard. Latensi retrieval dan total per pertanyaan ditulis ke log run (`llm/latency_report.py`).

---

## 4. Pencegahan leakage — mode `thread` (Amendment 15, DRAFT)

Satu definisi untuk B, C, dan D: `llm/common/leakage.py`.

**Dikecualikan untuk pertanyaan evaluasi q:**
- q itu sendiri (sebagai anchor, sebagai Question terkait pada 2-hop/expansion, dan sebagai pemilik jawaban di korpus B);
- semua jawaban thread q, termasuk accepted answer (disaring per `answer_id`).

**TIDAK dikecualikan:** pertanyaan Duplicate, pertanyaan Linked, dan pertanyaan near-duplicate (cosine ≥ 0,95), beserta jawabannya. Jawaban mereka boleh masuk konteks lewat vector search, `IS_RELATED_TO`, atau low-level retrieval. Eksklusi hanya berdasarkan identitas pertanyaan, tidak pernah berdasarkan isi accepted answer.

**Penegakan:** `assert_no_leakage` per pertanyaan (retrieved ∩ exclusion = ∅, `LeakageError` jika tidak). Buffer over-fetch mengikuti ukuran exclusion set (`top_n + |exclude| + 5` untuk anchor dan expansion; `k + |A| + 5` untuk B). Setiap record menyimpan `leakage = {mode, n_excluded_q, n_excluded_a, n_hits_filtered, final_k}`.

**Pengukuran, bukan jaminan:** ID-level self-retrieval harus 0. Cosine ≥ 0,95 antara tiap item konteks dan accepted answer **diukur dan dilaporkan**, tidak dijamin 0 dan tidak pernah dipakai sebagai filter.

**Dampak yang diketahui:** di sampel uji (n = 384) ada 2 pertanyaan dengan Duplicate dan 1 dengan near-duplicate (hanya hitungan agregat; tidak ada retrieval yang dijalankan pada sampel uji). Dev set: 0 Duplicate, 0 near-duplicate, 6 Linked. Stage 1 tidak terpengaruh (0 dari 50 exclusion set dev berbeda; retrieval C identik 50/50 pada α 0,5 dan 0,25).

**Mode sensitivitas** (opsional, bukan resmi): `dup`, `dup+neardup0.95`, `dup+linked+neardup0.95`, dan `off` (hanya audit). Dipilih lewat `--leakage-mode`, atau `LEAKAGE_EXCLUDE_DUP`, `LEAKAGE_EXCLUDE_LINKED`, `LEAKAGE_NEAR_DUP_COSINE` (default pada `LeakageSettings`: semuanya mati, jadi mode = `thread`). Eksklusi tambahan dibaca dari `results/leakage/exclusion_sets.parquet` (dibangun `llm/leakage/build_exclusion_sets.py`, sha256 di `.meta.json`); mode `thread` tidak memerlukan tabel itu. Audit retrieval-only: `llm/leakage/audit.py`, laporan `reports/leakage_audit.md`.

---

## 5. Skema config dan hash

- `config_hash` = hash dari dict config setiap kondisi (`build_config` di B, C, D; `llm/manifest.py::compute_config_hash`). Hash menentukan nama file output, sehingga resume/append tidak pernah mencampur run dengan config berbeda.
- **Leakage:** `config_leakage_key(mode)` menulis `"leakage": "<mode>"` untuk setiap mode bernama, termasuk `"leakage": "thread"`. Hanya `leakage_mode=None` (nilai bawaan builder; run yang dibuat sebelum ada pencegahan leakage per-query) tidak menulis key itu, sehingga hash run lama tidak berubah dan run `thread` baru tidak pernah berbagi hash atau nama file dengan run lama. Tes: `llm/c_graphrag/tests/test_config_hash_backward_compatibility.py`, `llm/tests/test_leakage_exclusion.py`.
- CLI B/C/D, dashboard (`backend/app/services/engine_service.py::_leakage_mode`), dan matriks resmi semuanya menghasilkan `thread` secara bawaan; matriks juga mem-*pin* `--leakage-mode thread` di setiap sel supaya isi `.env` tidak mengubah run resmi.
- Key lain yang masuk config hanya bila berbeda dari default: `alpha` (selalu untuk C v3), `sample_split` (bukan `test`), `max_hops`, `edge_types`, `use_author_trust`, `accepted_only`, `expansion_require_answer` + `expansion_overfetch`, `corpus` (B), `provenance_version`.
- `run_registry.expected_generation_config`, `stage2_report.py`, dan `trust_ablation_report.py` memeriksa `config["leakage"] == LEAKAGE_MODE` (`"thread"`); run dengan mode lain ditolak.

---

## 6. Evaluasi dan judge

- **Metrik:** cosine similarity jawaban vs accepted answer (`all-MiniLM-L6-v2`), hallucination rate biner (LLM-as-Judge), NF2 (jawaban dengan sitasi valid), NF3 (latensi, p95 ≤ 15 detik untuk C), dan context relevance (ctxrel-v2, hanya dev).
- **Panel resmi `panel-v2`** (`llm/evaluation/judge_panel_config.py`): Llama-3.3-70B-Instruct-Turbo, DeepSeek-V3.2, Qwen3-235B-A22B-Instruct-2507, via DeepInfra. Prompt `judge-v2.3`, parser `pv2.3-1`, blinding `blind-v3`, temperature 0, agregasi 2-dari-3. gpt-4o-mini tidak dipakai sebagai judge.
- **Status:** Amendment 14 (draf) membekukan panel tanpa gerbang kesepakatan manusia. Sampai disetujui dan di-commit, setiap judging resmi ditolak. Angka halusinasi harus ditulis sebagai keluaran panel LLM, bukan label manusia.
- **Anggaran:** DeepInfra $9,00 (`results/run_registry.json`). Proyeksi biaya di `results/cost_projection.md`: judging stage 2 sekitar $0,76; judging run resmi stage 1 + stage 2 sekitar $13,10 (sudah termasuk buffer 10%), di atas anggaran saat ini. Generasi gpt-4o-mini untuk run resmi sekitar $2,05.
- **Pelaporan:** `llm/evaluation/official_report.py`, `trust_ablation_report.py` (C-primary vs C-ablation), `compare_conditions_stats.py`, `analyze_error_attribution.py`.

---

## 7. Frozen parameters

Dibuat oleh `llm/frozen_parameters.py` dari file yang mendefinisikan tiap nilai; jangan diedit tangan. Isi: 62 parameter dalam grup C retrieval, Judge, trustScore, Alpha selection, Retrieval (B, C, D), Sampling, Leakage exclusion, Generation, Reporting. Grup leakage: mode resmi `thread` (frozen, Amendment 15), aturan eksklusi (frozen), serta ambang near-dup 0,95 dan tabel eksklusi sebagai informasi untuk mode sensitivitas saja.

- Regenerasi: `python3 -m llm.frozen_parameters`; pemeriksaan: `python3 -m llm.frozen_parameters --check` (dijalankan juga oleh `llm/tests/test_frozen_parameters.py`).
- `llm/run_registry.py` membaca hash tabel dari `results/frozen_parameters.json` dan menolak run yang config-nya menyimpang dari nilai beku.

---

## 8. Amendment yang berlaku

Sumber: `docs/DECISION_C_SCORING.md` kecuali disebut lain.

| No. | Isi |
|---|---|
| 1–4 | Kontrol context relevance, ctxrel-v2, hard negative, judge-v2 dan metrik biner |
| 5–6 | Reputasi dan skor log-ternormalisasi di trustScore (v3-logscore) |
| 7 | Paritas korpus dan teks query B dengan C |
| 9 | Semantic expansion hanya memilih Question yang punya jawaban |
| 10 | C-primary dan C-ablation (kedua kandidat α dijalankan pada sampel uji) |
| 11–13 | Judge-v2.2, validasi satu anotator, set konfirmasi 63 item, kriteria penerimaan κ |
| 12 | Panel LLM-as-Judge |
| **15** | **Aturan leakage aktif: mode `thread`. Status DRAFT; rationale `TODO(author)`** |
| 14 (draf, di luar file ini) | `reports/amendment14_draft.md`: panel beku tanpa gerbang kesepakatan manusia; belum di-commit. Penomoran 15 perlu dicek ulang bila 14 dimasukkan ke `DECISION_C_SCORING.md` |

Catatan: Amendment 8 (aturan eksklusi sebelumnya) tetap tercatat sebagai dokumen keputusan, tetapi aturan eksklusinya digantikan oleh Amendment 15 untuk evaluasi resmi.

---

## 9. Cara menjalankan

**Prasyarat lingkungan (`.env`):** `QUESTIONS_PARQUET`, `ANSWERS_PARQUET`, dan `KG_WORKSPACE_DIR` menunjuk ke volume eksternal ("T7 Shield"); volume harus ter-mount. C dan D membutuhkan Neo4j yang berjalan dan index FAISS di `KG_WORKSPACE_DIR`; B membutuhkan index `kg_answers` di `llm/b_rag/index_cache`.

**Satu run (contoh B-grounded, dari root repo):** perintah lengkap tiap sel R1–R9 ada di `docs/OFFICIAL_RUN_MATRIX.md`; menu interaktif lewat `python3 run.py`. Untuk dev: `--sample-split dev --n-sample 50`.

**Urutan menuju run resmi (belum dijalankan; tiap langkah menunggu persetujuan eksplisit penulis):**
1. Amendment 14 disetujui dan di-commit; Amendment 15 dikonfirmasi dan rationale diisi.
2. Stage 2: empat run C dev (α ∈ {0,25; 0} × grounding), perintah di `reports/alpha_stage2_plan.md`; daftarkan sebagai `dev-stage2`, nilai dengan panel-v2, laporan `llm/c_graphrag/stage2_report.py`.
3. Isi `alpha_primary` dan `alpha_ablation` di `official_params.json` dan blok `official_params` di `DECISION_C_SCORING.md` (harus byte-identik; dijaga tes).
4. Run resmi R1–R9 pada sampel uji, daftarkan sebagai `official`, lalu nilai dan laporkan.

**Tes:**
```bash
python3 -m pytest llm backend -q -p no:cacheprovider    # seluruh suite, sekitar 6 menit
python3 -m pytest llm/tests -q                           # hanya tes tingkat repo (subset)
```
Sebagian tes (`*_live.py`) membutuhkan Neo4j dan volume data; tes lain berjalan offline.

---

## 10. Hal yang belum atau tidak bisa dipastikan

- Rationale Amendment 15 belum ditulis (`TODO(author)`); amendment belum dikonfirmasi penulis.
- Stage 2 dan run resmi belum ada, sehingga belum ada hasil C-primary, C-ablation, maupun perbandingan kondisi.
- Judging resmi diblokir sampai Amendment 14 disetujui; proyeksi biaya judging resmi melebihi anggaran DeepInfra saat ini.
- File `run_history.jsonl` yang belum di-track di `llm/*/logs/` membuat dua tes lama gagal (`test_all_9_real_pilot_run_config_hashes_unchanged`, `test_calibration_generate_plan_and_registry`); keduanya tidak terkait leakage.
- Statistik graf (derajat, persentase node terisolasi, jangkauan 2-hop) tidak tercatat di log mana pun.
