# IMPLEMENTATION_STATUS — Implementasi GraphRAG saat ini

**Judul tesis:** *Pengembangan GraphRAG Framework Berbasis Pengetahuan Komunitas Stack Overflow untuk Mereduksi Halusinasi pada LLM* (ITB, Informatika, S2)

Dokumen ini menjelaskan kondisi implementasi sekarang, diverifikasi ke kode dan dokumen keputusan di repo (2026-10-26, branch `master`, HEAD `29f59a1` ditambah perubahan working tree Amendment 17 dan stage 2 yang belum di-commit). Sumber ditulis `path` atau `path:baris`. Tidak ada API key atau secret di dokumen ini.

---

## 1. Ringkasan

1. **Empat kondisi dan empat run resmi (Amendment 16, dikonfirmasi 2026-10-26):** A (LLM murni), B (RAG dense retrieval), C (GraphRAG trust-weighted, kontribusi utama), D (dual-level retrieval, adaptasi LightRAG). Run resmi: A, B-grounded, C-primary-grounded, D-grounded (sel R1, R3, R5, R9; run tesis R1-R4), dengan instruksi grounding dan citation AKTIF untuk B, C, D. B, C, dan D memakai prompt 4-turn yang sama (`llm/prompts.py`), model embedding `all-MiniLM-L6-v2`, generator `gpt-4o-mini`, dan sampel yang sama. Lima sel lain (B-plain, C-primary-plain, C-ablation plain/grounded, D-plain) adalah **legacy**: tetap ada di kode, ditolak pada sampel test tanpa `--allow-legacy-cells`.
2. **Aturan leakage aktif: mode `thread`** (Amendment 15, dikonfirmasi 2026-10-26): hanya pertanyaan evaluasi dan semua jawabannya yang dikecualikan. Duplicate, Linked, dan near-duplicate **tidak** dikecualikan. Alasan desain ada di Amendment 15 (rationale diisi).
3. **Status eksperimen:** stage 1 diulang pada Amendment 17 (C v4): top_two = 0,5 dan 0,25. **Stage 2 sudah dijalankan pada dev set (2026-10-26)**: 6 run C v4 (α 0,5 dan 0,25 sebagai kandidat; α 0 sebagai referensi eksploratif yang tidak ikut seleksi) × grounding on/off, dinilai panel-v2/judge-v2.3, terdaftar sebagai `dev-stage2`; laporan `llm/c_graphrag/results/stage2/stage2_report_20261025T221359Z.{md,json}`. Hasil ada di laporan itu; **α* belum ditulis ke `official_params.json`** (tetap `pending_selection`, menunggu keputusan penulis). Keempat run resmi belum dijalankan. `results/run_registry.json` memuat 12 entri `dev-stage2` selain 133 `archived`; tidak ada `official`.
4. **Gerbang resmi dibuka pada 2026-10-26 (commit persetujuan `approve amendments 14-17 and unlock official judging`):** `official_params.json` berstatus `selected` (α\* = 0,5, α ablasi = 0,25) sehingga Kondisi C boleh berjalan pada test sample; `AMENDMENT14_APPROVED = True` dan `OFFICIAL_ALLOWED = ("judge-v2.3",)` (`llm/evaluation/judge_versions.py`) sehingga judging resmi dengan panel-v2 diizinkan. Amendment 14 (`reports/amendment14_draft.md`, empat run, tanpa gerbang manusia) dikonfirmasi penulis pada 2026-10-26. **Tidak ada run resmi yang pernah dijalankan; setiap run resmi menunggu "go" eksplisit penulis.**
5. **Parameter dibekukan lewat tabel yang dihasilkan skrip** (`llm/frozen_parameters.py` → `docs/FROZEN_PARAMETERS.md`, `results/frozen_parameters.json`, hash tabel `3e29586bf910`). Status "frozen (Amendment 14)" mengikat sejak Amendment 14 dikonfirmasi (2026-10-26).

---

## 2. Konfigurasi resmi (empat run: sel R1, R3, R5, R9)

Sumber: `llm/official_run_matrix.py`, `docs/OFFICIAL_RUN_MATRIX.md`.

| Parameter | Nilai |
|---|---|
| Sampel uji | n = 384, seed 42, pool 1536 (1471 setelah filter token 2048), posisi 1–384; daftar ID beku `llm/sample_ids/test_ids_v1.json` |
| Dev set | n = 50, posisi 385–434; `llm/sample_ids/dev_ids_v1.json` |
| Generator | `openai` / `gpt-4o-mini` |
| top-k, chunk | 5, 400 token (tiktoken `cl100k_base`) |
| C | `n_anchor` 3, `n_semantic_expansion` 3, retrieval v3, `fusion_mode trust_weighted`, expansion answerable-only |
| D | `n_low_level` 3, `n_high_level` 3, retrieval **v5** (satu skor cosine di kedua level, anchor ber-jawaban, high level mengisi sampai 5 jawaban; batas 100 tetangga yang dinilai per kueri) |
| Leakage | `thread` |

Run resmi (Amendment 16): **A** (sel R1), **B-grounded** (R3), **C-primary-grounded** (R5), **D-grounded** (R9); nomor run di tesis R1–R4. "Grounded" berarti instruksi grounding + sitasi. Sel **legacy** (R2 B-plain, R4 C-primary-plain, R6 dan R7 C-ablation, R8 D-plain) masih didefinisikan, tetapi ditolak pada sampel test tanpa `--allow-legacy-cells` (CLI) / `allow_legacy_cells` (API) dan tidak pernah dimulai oleh preset Official, daftar bawaan Factorial Batch, atau "Run all". Tahap 2 pemilihan α tetap menghasilkan kandidat kedua (`alpha_ablation`), tetapi tidak ada run resmi yang memakainya. Perbandingan terdaftar: C vs A, C vs B, C vs D.

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

### 3.3 Kondisi C — GraphRAG trust-weighted (retrieval v4 = ranking v3 + anchor ber-jawaban)
`llm/c_graphrag/c_graphrag.py`. Membaca KG Neo4j dan index FAISS yang sudah ada; tidak membangun apa pun.
1. **Entity anchoring (v4, Amendment 17):** vector search atas embedding Title+Body pertanyaan; anchor hanya Question yang punya ≥ 1 jawaban tak-terkecualikan: yang tanpa jawaban dilewati dan scan berlanjut di daftar FAISS yang sama (kedalaman 400) sampai `n_anchor` = 3 anchor ditemukan (kurang dari itu dicatat). Fungsi yang sama dipakai D v5. Alasan: hanya 28% anchor dev punya jawaban. v3 (anchor biasa) tetap bisa direproduksi.
   - **Kedalaman scan 400 (`ANCHOR_SCAN_DEPTH`, `llm/common/anchors.py:15`):** angka bulat yang ditetapkan sebagai batas aman, bukan persentil dari data; satu-satunya pembenaran tertulis adalah komentar kode ("tested on the dev set: 3 answerable seeds always found"). FAISS diminta `400 + |exclude| + 5` hasil (rumus `N_anchor + |exclude| + 5` hanya untuk jalur lama v3/v4-D); hit yang dikecualikan dibuang dulu, lalu paling banyak 400 kandidat dicek ke Neo4j dalam satu query batch, dan `n_anchor` pertama yang ber-jawaban diambil menurut urutan FAISS. Jika kurang dari 3 yang lolos: tidak error dan tidak mengulang; yang ditemukan dipakai (bisa kosong, C lalu melewati traversal) dan kekurangannya dicatat di record (`anchors_requested`, `anchors_found`, `anchor_candidates_scanned`, `anchors_skipped_unanswered`).
   - **Teramati `anchor_candidates_scanned` (dibaca dari file hasil, 2026-10-27; post hoc):** dev (n = 50, semua α, C dan D): maks 35, P95 30; sampel uji (n = 384, C v4 α 0,5 dan D): maks 53, P95 28. `anchors_found` = 3 pada 50/50 dev dan 384/384 uji. Maksimum 53 ≈ 13% dari batas 400. Angka 400 tidak dipilih dari data sampel uji.
2. **Graph traversal:** 1-hop (jawaban langsung anchor via `HAS_ACCEPTED_ANSWER`/`HAS_ANSWER`) dan 2-hop (anchor → Question terkait via `IS_RELATED_TO`/`TAG_COOCCUR`/`EMBED_SIM` → jawabannya, `LIMIT 50`). Tie-break `a.id ASC`.
3. **Semantic expansion** (Amendment 9): per seed, `n_semantic_expansion` Question paling mirip yang masih punya ≥ 1 Answer yang tidak dikecualikan (over-fetch faktor 20, `EXPANSION_OVERFETCH`). Setting `EXPANSION_REQUIRE_ANSWER` default True.
4. **Fusi dan ranking v3:** dedup per `answer_id`, `final = (1 − α)·sim_norm + α·trust`; `sim` = cosine antara query dan teks kandidat yang dilihat LLM, `sim_norm` = min-max per pertanyaan, `trust` = 0,7·path trust + 0,3·trustScore Answer (kandidat expansion dibatasi 0,4). Urut `final` menurun, tie-break `answer_id` naik, ambil top-5 (`docs/DECISION_C_SCORING.md`).
5. **Switch eksploratif** (tidak dipakai memilih α; resmi memakai default): `max_hops = 2`, kelima tipe edge, `use_author_trust = false`, `accepted_only = false` (`llm/c_graphrag/official_params.json`).

**trustScore v3-logscore** (Amendment 5–6, terpasang di Neo4j): Answer = 0,5·`norm_log(score; P99)` + 0,3·`is_accepted` + 0,2·`w_auth` (bobot reputasi `AUTHOR_TRUST`); Question = 0,7·`norm_log(score)` + 0,3·`minmax(log1p(views))`. Run C mencatat `trustScoreVersion` dari graf; `stage2_report.py` menolak run yang tidak mencatat `v3-logscore`.

**KG:** 2.685.814 Question, 725.795 Answer, 602.900 Question dengan ≥ 1 Answer.

**α:** kandidat stage 2 adalah **0,5 dan 0,25**, diturunkan (bukan ditulis tangan) dari top_two laporan stage 1 of record `stage1_anchors_report_20261025T165734Z.json` oleh `llm/c_graphrag/stage2_candidates.py` (`STAGE2_ALPHAS`). Stage 1 diulang pada Amendment 17 dengan C v4 (anchor hanya dari Question ber-jawaban): % pertanyaan dengan ≥ 1 konteks RELEVANT = 0,0: 56.0, 0,25: 56.0, 0,5: 58.0, 0,75: 56.0, 1,0: 40.0; skor rata-rata = 0,0: 0.376, 0,25: 0.374, 0,5: 0.372, 0,75: 0.332, 1,0: 0.256. Hasil sebelumnya (anchor biasa, C v3) memberi 0,25 dan 0. α 0 dijalankan sebagai **referensi eksploratif** yang dideklarasikan sebelumnya; aturan seleksi tidak pernah melihatnya. `alpha_primary` dan `alpha_ablation` di `official_params.json` masih `null`.

**Hasil stage 2 (dev, n = 50 × grounding on/off = 100 jawaban per α, panel-v2 / judge-v2.3, C v4, `thread`; laporan `llm/c_graphrag/results/stage2/stage2_report_20261025T221359Z.{md,json}`):**

| α | status | halusinasi (ABSTAIN dikeluarkan) | ABSTAIN | tingkat (a) | halusinasi + ABSTAIN | tingkat (b) | penuh / sebagian |
|---|---|---|---|---|---|---|---|
| 0,5 | kandidat | 23 / 91 | 9 | 0,2527 | 32 / 100 | 0,32 | 22 / 1 |
| 0,25 | kandidat | 21 / 90 | 10 | 0,2333 | 31 / 100 | 0,31 | 20 / 1 |
| 0 | eksploratif, tidak ikut seleksi | 21 / 90 | 10 | 0,2333 | 31 / 100 | 0,31 | 19 / 2 |

Per grounding (halusinasi / ABSTAIN): α 0,5 on 11 / 9, off 12 / 0; α 0,25 on 8 / 10, off 13 / 0; α 0 on 8 / 10, off 13 / 0. Rata-rata item konteks per jawaban: 5,0 untuk keenam run. Hitungan per judge (ABSTAIN dikeluarkan, dipool): Llama 1/93, 1/91, 4/92; DeepSeek 26/88, 31/89, 31/90; Qwen3-235B 30/89, 32/88, 35/88 (urutan α 0,5; 0,25; 0). Aturan terdaftar menghasilkan **α = 0,5**, tetapi diputuskan pada langkah **pemutus seri (α lebih besar)**: selisih primer 2 jawaban (≤ margin 2) dan selisih sekunder 1 jawaban (≤ margin 2), jadi bukan keunggulan terukur. Kalimat alasan di laporan pertama ("lowest pooled hallucination rate") menyesatkan; sudah diperbaiki (2026-10-26): `select_stage2_winner` kini menyebut langkah pemutus (`deciding_tier`), dan laporan baru `stage2_report_20261025T232940Z` dibuat dari data yang sama (laporan lama dipertahankan). Catatan: `docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md`; perubahan α\* yang disiapkan (belum diterapkan): `docs/PREPARED_ALPHA_SELECTION_CHANGE.md`. Biaya stage 2: generasi $0,164; judging sekitar $1,08 (estimasi per record; API DeepInfra menunjukkan sekitar $0,95–1,0); total sekitar $1,25. **α\* belum diterapkan** ke `official_params.json` (menunggu keputusan penulis).

**Verifikasi per juri dan tumpang-tindih (dev, baca-saja, 2026-10-26).** Sumber: `llm/c_graphrag/results/stage2/stage2_report_20261025T232940Z.json` (kunci `per_judge_pooled`, `per_judge_per_run`, `pooled`, `exploratory`). Semua angka di tabel hasil dan di paragraf per juri di atas **cocok** dengan laporan; tidak ada selisih.

| Juri | α 0,5 | α 0,25 | α 0 | ABSTAIN (0,5 / 0,25 / 0) |
|---|---|---|---|---|
| Llama | 1 / 93 | 1 / 91 | 4 / 92 | 7 / 9 / 8 |
| DeepSeek | 26 / 88 | 31 / 89 | 31 / 90 | 12 / 11 / 10 |
| Qwen3-235B | 30 / 89 | 32 / 88 | 35 / 88 | 11 / 12 / 12 |
| Panel 2-dari-3 | 23 / 91 | 21 / 90 | 21 / 90 | 9 / 10 / 10 |

Panel, ABSTAIN dihitung gagal: 32 / 100, 31 / 100, 31 / 100.

Per mode prompt, per juri (halusinasi / dinilai, ABSTAIN dikeluarkan):

| Juri | α 0,5 on | α 0,5 off | α 0,25 on | α 0,25 off | α 0 on | α 0 off |
|---|---|---|---|---|---|---|
| Llama | 1 / 43 | 0 / 50 | 1 / 41 | 0 / 50 | 1 / 42 | 3 / 50 |
| DeepSeek | 11 / 38 | 15 / 50 | 13 / 39 | 18 / 50 | 11 / 40 | 20 / 50 |
| Qwen3-235B | 15 / 39 | 15 / 50 | 13 / 38 | 19 / 50 | 17 / 38 | 18 / 50 |

Tumpang-tindih antarjuri (**tidak ada di laporan**; dihitung dari `llm/evaluation/results/panel_stage2_{candidates,exploratory}_judge-v2.3.jsonl` dengan `judge_panel_aggregate.aggregate_labels`, tanpa panggilan API; berkas `*_panel.jsonl` tidak dipakai karena hanya 50 baris dan hanya run tanpa grounding):

| α | ketiga juri | tepat dua juri (pasangan) | tepat satu juri | ditandai ≥ 1 juri |
|---|---|---|---|---|
| 0,5 | 0 | 23 (DeepSeek+Qwen 23) | 11 (Llama 1, DeepSeek 3, Qwen 7) | 34 |
| 0,25 | 0 | 21 (DeepSeek+Qwen 20, Llama+Qwen 1) | 22 (DeepSeek 11, Qwen 11) | 43 |
| 0 | 3 | 18 (DeepSeek+Qwen 17, Llama+DeepSeek 1) | 25 (DeepSeek 10, Qwen 15) | 46 |

Jawaban yang ditandai dua juri atau lebih adalah halusinasi menurut panel; yang ditandai satu juri saja berlabel faktual. Karena itu label panel hampir sama antar-α (23 / 21 / 21), sedangkan angka per juri menurun pada α 0,5 terutama lewat tanda tunggal (11 vs 22). Tanda DeepSeek+Qwen saja: 23 vs 20.

`select_stage2_winner` dijalankan ulang pada metrik laporan: kandidat {0,5; 0,25} dan {0,5; 0,25; 0} sama-sama menghasilkan winner 0,5, `deciding_tier` = tie-break, `keep_current` = false.

### 3.4 Kondisi D — dual-level retrieval
`llm/d_lightrag/d_lightrag.py` (`D_RETRIEVAL_VERSION = "v5"` sejak Amendment 17 = v4 + anchor ber-jawaban yang sama dengan C v4 + high level menambah tetangga sampai ≥ 5 jawaban unik; v4 dan v3 tetap bisa direproduksi lewat `--d-retrieval-version`). Cek retrieval-only dev: 47 dari 50 pertanyaan mendapat 5 item (v4: 20), rata-rata 4,94; C v4: 50 dari 50. Memakai KG dan FAISS yang sama dengan C; bukan LightRAG asli (tanpa ekstraksi entitas oleh LLM).
- **Satu skor untuk kedua level (v4):** cosine antara embedding kueri dan embedding Question pemilik jawaban (anchor untuk low-level, tetangga untuk high-level), dari cache embedding indeks FAISS. Bobot edge tidak lagi dibaca atau dipakai sebagai skor (v3 memberi 0,6 / 0,4 / 1,0 atau cosine, skala yang tidak sebanding).
- **Low-level:** anchor yang sama dengan C (`n_low_level`), jawaban 1-hop.
- **High-level:** dari anchor yang sama, tetangga unik via `IS_RELATED_TO`/`TAG_COOCCUR`/`EMBED_SIM` yang punya jawaban tak-terkecualikan; paling banyak `HIGH_LEVEL_CANDIDATE_CAP = 100` Question dinilai per kueri (bila lebih, id terkecil dipertahankan; tercatat sebagai `high_level_cap_hit`); `n_high_level` terbaik menurut cosine (id naik untuk seri) diambil beserta jawabannya.
- **Fusi:** dedup per `answer_id` (skor tertinggi), urut menurun, seri menurut `answer_id`, top-5. Tanpa trust, tanpa bonus accepted, tanpa semantic expansion. C vs D mengukur efek mekanisme retrieval secara keseluruhan pada graf yang sama.
- Belum ada retrieval v4 yang dijalankan terhadap Neo4j; tes memakai driver palsu (`llm/d_lightrag/tests/test_d_retrieval_v4.py`).

### 3.5 Provenance per item
Record B/C/D menyimpan jalur retrieval tiap item (`provenance_version = "v3"`, `llm/common/item_provenance.py`), ikut dalam `config_hash`. Pencarian atas permintaan lewat `llm/provenance_lookup.py` dan tombol "Show provenance" di dashboard. Latensi retrieval dan total per pertanyaan ditulis ke log run (`llm/latency_report.py`).

---

## 4. Pencegahan leakage — mode `thread` (Amendment 15, dikonfirmasi 2026-10-26)

Satu definisi untuk B, C, dan D: `llm/common/leakage.py`.

**Dikecualikan untuk pertanyaan evaluasi q:**
- q itu sendiri (sebagai anchor, sebagai Question terkait pada 2-hop/expansion, dan sebagai pemilik jawaban di korpus B);
- semua jawaban thread q, termasuk accepted answer (disaring per `answer_id`).

**TIDAK dikecualikan:** pertanyaan Duplicate, pertanyaan Linked, dan pertanyaan near-duplicate (cosine ≥ 0,95), beserta jawabannya. Jawaban mereka boleh masuk konteks lewat vector search, `IS_RELATED_TO`, atau low-level retrieval. Eksklusi hanya berdasarkan identitas pertanyaan, tidak pernah berdasarkan isi accepted answer.

**Penegakan:** `assert_no_leakage` per pertanyaan (retrieved ∩ exclusion = ∅, `LeakageError` jika tidak). Buffer over-fetch mengikuti ukuran exclusion set (`top_n + |exclude| + 5` untuk anchor dan expansion; `k + |A| + 5` untuk B). Setiap record menyimpan `leakage = {mode, n_excluded_q, n_excluded_a, n_hits_filtered, final_k}`.

**Pengukuran, bukan jaminan:** ID-level self-retrieval harus 0. Cosine ≥ 0,95 antara tiap item konteks dan accepted answer **diukur dan dilaporkan**, tidak dijamin 0 dan tidak pernah dipakai sebagai filter.

**Dampak yang diketahui:** di sampel uji (n = 384) ada 2 pertanyaan dengan Duplicate dan 1 dengan near-duplicate (hitungan agregat). Dev set: 0 Duplicate, 0 near-duplicate, 6 Linked. Stage 1 tidak terpengaruh (0 dari 50 exclusion set dev berbeda; retrieval C identik 50/50 pada α 0,5 dan 0,25).

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

- **Metrik:** cosine similarity jawaban vs accepted answer (`all-MiniLM-L6-v2`); hallucination rate biner dalam dua versi, (a) ABSTAIN dikeluarkan (utama) dan (b) ABSTAIN dihitung gagal = (halusinasi + ABSTAIN) / semua jawaban terpakai, plus pembagian keparahan (penuh / sebagian) dan porsi klaim tak-terverifikasi; faithfulness dan citation support (B, C, D; dari label per klaim pada record panel, tanpa panggilan baru; penyebut di `docs/OFFICIAL_RUN_MATRIX.md` dan Amendment 16); NF2 (jawaban dengan sitasi valid); NF3 (p95 latensi end-to-end Kondisi C ≤ 15 detik; p50/p95 retrieval untuk setiap run); metrik sekunder satu judge (Llama-3.3-70B-Instruct, temperature 0): context relevance (B, C, D) dan answer relevance (A–D), `llm/evaluation/secondary_metrics.py` (dry-run mencetak jumlah panggilan; run pada berkas resmi ditolak sampai Amendment 14 disetujui).
- **Panel resmi `panel-v2`** (`llm/evaluation/judge_panel_config.py`): Llama-3.3-70B-Instruct-Turbo, DeepSeek-V3.2, Qwen3-235B-A22B-Instruct-2507, via DeepInfra. Prompt `judge-v2.3`, parser `pv2.3-1`, blinding `blind-v3`, temperature 0, agregasi 2-dari-3. gpt-4o-mini tidak dipakai sebagai judge.
- **Status:** Amendment 14 (dikonfirmasi 2026-10-26) membekukan panel tanpa gerbang kesepakatan manusia; gerbang judging resmi dibuka di commit persetujuan. Angka halusinasi harus ditulis sebagai keluaran panel LLM, bukan label manusia.
- **Anggaran:** DeepInfra $9,00 (`results/run_registry.json`). Proyeksi di `results/cost_projection.md` (dibuat ulang oleh `llm/cost_projection.py` untuk empat run, buffer 10% termasuk): judging stage 2 sekitar $0,76; judging empat run resmi sekitar $5,46; metrik sekunder sekitar $1,3 (batas atas); jumlah sekitar $7,5, di bawah anggaran. Generasi gpt-4o-mini untuk empat run resmi sekitar $0,86. Proyeksi lama untuk sembilan run (sekitar $13,10) tidak berlaku lagi.
- **Pelaporan:** `llm/evaluation/official_report.py` (perbandingan C vs A, C vs B, C vs D: selisih, bendera consistent / inconclusive, hitungan berpasangan per pertanyaan lebih baik / sama / lebih buruk dalam kedua versi, persentase jawaban yang diberi label sama oleh ketiga judge; Fleiss dan Krippendorff tetap), `trust_ablation_report.py` (hanya untuk sel legacy), `compare_conditions_stats.py`, `analyze_error_attribution.py`.

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
| 10 | C-primary dan C-ablation (kedua kandidat α dijalankan pada sampel uji). **Tidak berlaku untuk evaluasi resmi sejak Amendment 16**: C-ablation adalah sel legacy |
| 11–13 | Judge-v2.2, validasi satu anotator, set konfirmasi 63 item, kriteria penerimaan κ |
| 12 | Panel LLM-as-Judge |
| **15** | **Aturan leakage aktif: mode `thread`. Dikonfirmasi 2026-10-26; rationale diisi** |
| **17** | **Anchor hanya dari Question ber-jawaban untuk C (retrieval v4) dan D (v5, satu fungsi bersama `llm/common/anchors.py`, kedalaman scan 400); D v5 mengisi high level sampai k = 5 jawaban; stage 1 diulang (dev, thread): top_two berubah menjadi 0,5 dan 0,25. Dikonfirmasi 2026-10-26** |
| **16** | **Empat run resmi (A, B-grounded, C-primary-grounded, D-grounded); D retrieval v4 (satu skor cosine); pelaporan dua versi hallucination rate, faithfulness / citation support, metrik sekunder, hitungan berpasangan; `C_RETRIEVAL_VERSION` bawaan v3. Dikonfirmasi 2026-10-26** |
| 14 (dikonfirmasi 2026-10-26; salinan di `DECISION_C_SCORING.md` antara Amendment 13 dan 15) | `reports/amendment14_draft.md`: ditulis ulang untuk empat run, panel beku tanpa gerbang manusia; disetujui. Set konfirmasi 63 item (Amendment 11, 7 × 9 run) milik gerbang manusia yang dihapus; kodenya dorman |

Catatan: Amendment 8 (aturan eksklusi sebelumnya) tetap tercatat sebagai dokumen keputusan, tetapi aturan eksklusinya digantikan oleh Amendment 15 untuk evaluasi resmi.

---

## 9. Cara menjalankan

**Prasyarat lingkungan (`.env`):** `QUESTIONS_PARQUET`, `ANSWERS_PARQUET`, dan `KG_WORKSPACE_DIR` menunjuk ke volume eksternal ("T7 Shield"); volume harus ter-mount. C dan D membutuhkan Neo4j yang berjalan dan index FAISS di `KG_WORKSPACE_DIR`; B membutuhkan index `kg_answers` di `llm/b_rag/index_cache`.

**Satu run (contoh B-grounded, dari root repo):** perintah lengkap keempat sel resmi (dan sel legacy) ada di `docs/OFFICIAL_RUN_MATRIX.md`; menu interaktif lewat `python3 run.py`. Untuk dev: `--sample-split dev --n-sample 50`.

**Urutan menuju run resmi (belum dijalankan; tiap langkah menunggu "go" eksplisit penulis):**
1. Amendment 14-17 dikonfirmasi dan perubahan gerbang judging diterapkan (2026-10-26). SELESAI.
2. Stage 2 (SUDAH dijalankan, 2026-10-26): enam run C v4 dev (α 0,5 dan 0,25 + α 0 eksploratif, × grounding), perintah di `reports/alpha_stage2_plan.md`, terdaftar sebagai `dev-stage2`, dinilai panel-v2, laporan di `llm/c_graphrag/results/stage2/`. α\* = 0,5 diterima penulis. SELESAI.
3. `alpha_primary` = 0,5 dan `alpha_ablation` = 0,25 sudah diisi di `official_params.json` dan blok `official_params` di `DECISION_C_SCORING.md` (identik sebagai JSON; dijaga tes). SELESAI.
4. Cek retrieval-only D v5 pada dev set sudah dilakukan (47 dari 50 pertanyaan dengan 5 item). BELUM: empat run resmi pada sampel uji (daftarkan sebagai `official`, nilai dengan panel, jalankan metrik sekunder, buat laporan) — menunggu "go".

**Tes (terakhir, 2026-10-26: 1544 lulus, 27 dilewati (26 integrasi Neo4j/KG yang butuh GRAPHRAG_INTEGRATION=1, 1 tes status pending yang dilewati karena params sudah selected), 0 gagal; `python3 llm/frozen_parameters.py --check` mutakhir):**
```bash
python3 -m pytest llm backend -q -p no:cacheprovider    # seluruh suite, sekitar 6 menit
python3 -m pytest llm/tests -q                           # hanya tes tingkat repo (subset)
```
Sebagian tes (`*_live.py`) membutuhkan Neo4j dan volume data; tes lain berjalan offline.

---

## 10. Hal yang belum atau tidak bisa dipastikan

- Amendment 14-17 dikonfirmasi penulis pada 2026-10-26; kolom commit di Record menyebut subject line commit persetujuan.
- Run resmi belum ada, sehingga belum ada hasil C-primary maupun perbandingan kondisi. Stage 2 (dev) sudah ada; α* belum dipilih secara resmi. Retrieval D v5 sudah dicek retrieval-only terhadap Neo4j pada dev set (tanpa LLM), tetapi belum pernah dijalankan dengan generasi.
- Judging resmi tidak lagi diblokir (gerbang dibuka 2026-10-26); anggaran DeepInfra $9,00, sisa menurut ledger sekitar $7,83 (lihat `reports/amendment14_draft.md` bagian (j)), cukup untuk empat run (lihat bagian 6).
- Dua tes lama yang gagal karena `run_history.jsonl` (`test_all_9_real_pilot_run_config_hashes_unchanged`, `test_calibration_generate_plan_and_registry`) sudah diisolasi: keduanya sekarang membaca riwayat run pra-resmi dari arsip, bukan berkas live (perilaku produksi tidak berubah).
- `llm/evaluation/judge_panel.py` sempat macet berkali-kali pada panggilan terakhir (satu koneksi menggantung; timeout bawaan SDK 600 detik). Perbaikan (2026-10-26): timeout klien 180 detik dan hasil ditulis begitu selesai (`as_completed`, bukan `pool.map` berurutan). Prompt, parser, panggilan, dan agregasi tidak berubah. Saat macet, proses dapat dimatikan dan dijalankan ulang (hasil yang sudah ditulis tidak dibayar lagi). Ledger registri: $0,595 tercatat; sekitar $0,107 dari run yang dibatalkan ditambahkan pada 2026-10-26 sebagai entri penyesuaian eksplisit (total $0,702). Tidak ada hash yang berubah (lihat `docs/STAGE2_RESULT_AND_RUNTIME_NOTE.md`).
- Kalimat `reasoning` di `select_stage2_winner` sudah diperbaiki (2026-10-26) dan menyebut langkah yang memutuskan; laporan stage 2 lama tetap tidak diubah.
- Versi FAISS saat indeks dibangun (Agustus 2026) tidak tercatat; yang terpasang sekarang dan dipakai saat retrieval adalah `faiss-cpu 1.12.0`.

---

## 11. Fakta graf dan lingkungan (dicek 2026-10-25, hanya baca)

**Versi.** Neo4j Kernel **2026.07.0** (enterprise; Cypher 5 dan 25; `CALL dbms.components()`), database `graphrag`; driver Python `neo4j 5.23.1`; `faiss-cpu 1.12.0`; `sentence-transformers 2.2.2`; `numpy 1.26.4`; `torch 2.7.0`; Python 3.10.9.

**Statistik graf.** Log `logs/20261013T145605Z_validate_densification.log` (2026-10-13) ada dan mencatat statistik korpus: 2.685.814 Question; Question tanpa edge keluar (IS_RELATED_TO + TAG_COOCCUR + EMBED_SIM) **620.542 = 23,10%**; derajat keluar rata-rata **8,2411**, median **6**. (Catatan lama "statistik graf tidak tercatat di log mana pun" keliru dan dihapus.)

**Isolasi sebelum densifikasi.** Query baca-saja pada graf saat ini (`MATCH (q:Question) WHERE NOT (q)-[:IS_RELATED_TO]->() RETURN count(q)`), dijalankan 2026-10-25 dengan akses READ:
- tanpa edge **keluar** `IS_RELATED_TO`: **2.548.248** dari 2.685.814 Question = **94,88%** (sebelum densifikasi graf hanya memiliki edge ini di antara Question, jadi ini adalah isolasi sebelum densifikasi menurut definisi derajat keluar yang sama dengan angka "sesudah" 23,10%);
- tanpa edge `IS_RELATED_TO` sama sekali, masuk maupun keluar (`WHERE NOT (q)-[:IS_RELATED_TO]-()`): **2.469.727** = **91,96%**. Ini persis angka "91,96%" di docstring skrip 10 dan 12, jadi angka lama itu adalah ukuran tak-berarah; ukuran berarah (derajat keluar, sama dengan angka sesudah) adalah 94,88%.
- Untuk perbandingan sebelum/sesudah gunakan satu definisi, derajat keluar: **94,88% sebelum → 23,10% sesudah** densifikasi.

**Target validasi tingkat sampel** (≥ 80% node hasil retrieval punya edge keluar; rata-rata jangkauan 2-hop ≥ 5) **bukan kriteria penerimaan**; skrip `01_data_cleaning/12_validate_densification.py` mencetaknya sebagai informasi. Kriteria penerimaan adalah tingkat korpus (isolasi ≤ 30%, derajat rata-rata ≥ 3, median ≥ 1) dan, bila sampel dijalankan, latensi NF3. Validasi tingkat sampel pada log 2026-10-13 berhenti (cache FAISS tidak ditemukan), jadi memang tidak ada angkanya.

**2-hop traversal di C.** `ORDER BY` dari `LIMIT 50` memakai **bobot jalur penuh** `r2.weight * r3.weight` (edge pertama × edge kedua), lalu `a.id` naik sebagai pemutus seri; `LIMIT 50` berlaku **per kueri** (satu query Cypher untuk ketiga anchor sekaligus, bukan per anchor atau per seed) dan sebelum dedup jawaban. 1-hop tidak dibatasi. Sumber: `llm/c_graphrag/c_graphrag.py` (`traverse_graph`, baris `WITH ... (r2.weight * r3.weight) AS edge_weight ORDER BY edge_weight DESC, a.id ASC LIMIT 50`).

**Mode leakage pada stage-1 sweep.** Stage 1 of record (recheck Amendment 9, `expansion_stage1_recheck.py`, 2026-10-18) dijalankan dengan mode leakage **`dup+neardup0.95`** (bawaan saat itu), tercatat di `meta.leakage_mode` pada `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.json` dan di `leakage.mode` setiap record `stage1_expansion_*_alpha*.jsonl` (50 record per α, semuanya `n_excluded_q = 1`). Mode resmi sekarang `thread`. Himpunan eksklusi kedua mode **identik** pada ke-50 pertanyaan dev: `results/leakage/exclusion_sets.parquet` mencatat 0 Duplicate dan 0 near-duplicate untuk semuanya, sehingga tambahan eksklusi `dup+neardup0.95` kosong (kalimat Amendment 15 "stage 1 ran with the thread-only exclusion" kurang tepat secara nama mode, tetapi benar secara isi). Stage-1 Amendment 5–6 (sebelum Amendment 8) memakai eksklusi thread saja. Retrieval tidak dijalankan ulang di sini, jadi kesamaan top-5 (50/50 pada α 0,5 dan 0,25, menurut Amendment 15) tidak diverifikasi ulang. Untuk menjalankan ulang: `python3 llm/c_graphrag/expansion_stage1_recheck.py` (dengan mode bawaan `thread`; butuh Neo4j, cache FAISS dan data parquet; retrieval-only tanpa biaya API, hanya item konteks yang labelnya tidak ada di cache yang perlu dinilai ulang oleh ctxrel-v2/Llama, sekitar $0,014 pada recheck sebelumnya). Tidak dijalankan.


---

## 12. Pembaruan 2026-10-26: generasi resmi selesai, penilaian (judging) tertahan lalu diperbaiki

### 12a. Hasil generasi resmi (empat sel, dari tag `official-run-start2`)

Keempat sel dijalankan satu per satu dari tag `official-run-start2` (commit `e1dd047`), hash parameter beku `fcbee86cbb25`, satu invocation per sel, tanpa resume. Tidak ada kegagalan leakage, tidak ada rekaman yang tidak dapat dipakai, tidak ada galat drive, Neo4j, atau OpenAI.

| | R1 A | R3 B-grounded | R5 C-primary (α 0,5) | R9 D-grounded |
|---|---|---|---|---|
| n / usable | 384 / 384 | 384 / 384 | 384 / 384 | 384 / 384 |
| id unik, urutan beku | ya, ya | ya, ya | ya, ya | ya, ya |
| rata-rata item konteks | 0 | 5,00 | 5,00 | 4,91 (26 pertanyaan < 5, minimum 3) |
| latensi ujung ke ujung rata-rata / p95 / maks (s) | 7,55 / 10,96 / 20,5 | 5,97 / 9,39 / 77,1 | 8,09 / **13,70** / 23,3 | 6,50 / 11,13 / 40,6 |
| retrieval rata-rata / p95 (s) | – | 1,53 / 2,17 | 2,82 / 4,73 | 1,05 / 1,85 |
| kutipan valid (NF2) | – | 72,4% | 68,2% | 67,2% |
| cosine rata-rata ke jawaban diterima | 0,614 | 0,602 | 0,600 | 0,601 |
| waktu dinding | 51,3 menit | 42,2 menit | 55,1 menit | 45,0 menit |
| biaya generasi | $0,190 | $0,211 | $0,207 | $0,201 |

- Total biaya generasi **$0,8085** (proyeksi $0,856 dengan buffer 10%).
- **NF3** (p95 latensi ujung ke ujung Kondisi C): **13,70 s**, di bawah target 15 s. Kondisi pengukuran: Apple M2 dengan RAM 8 GB, Neo4j lokal, embedding pertanyaan dibaca dari SSD eksternal (T7 Shield), indeks FAISS dimuat sekali ke RAM.
- Outlier B: satu panggilan 77 s (hampir pasti penundaan sisi OpenAI); hanya memengaruhi maksimum, bukan p95.
- ABSTAIN belum dapat dihitung (label panel). Proksi: jawaban tanpa kutipan valid = 27,6% (B), 31,8% (C), 32,8% (D).
- Manifes menunjukkan `git_dirty: true` karena berkas keluaran yang belum dilacak dan penambahan ke `run_history.jsonl` oleh run itu sendiri; kode tidak berubah.
- Keempat run didaftarkan sebagai `official` di `results/run_registry.json` (biaya per sel di atas). Ledger DeepInfra tidak berubah oleh pendaftaran generasi.

### 12b. Penghalang penilaian dan perbaikannya

- Konfigurasi run sampel uji **tidak menyimpan** kunci `sample_split` (hanya ditulis bila bukan `test`, supaya hash lama tidak berubah). `judge_versions.run_purpose()` mengklasifikasikan berkas hanya sebagai resmi bila kunci itu bernilai `test`, sehingga keempat berkas resmi dianggap *development* dan `judge_panel.py --official` menolaknya. Tanpa `--official` mereka akan lolos tanpa gerbang resmi.
- Perbaikan: kunci yang tidak ada berarti `test` (`(config.get("sample_split") or "test") == "test"`); rekaman tanpa `config` (keluaran juri, sweep) tetap bukan berkas run. Tes baru di `test_judge_versions.py` (tidak ada → resmi, `dev` → development, `test` → resmi, tanpa config → development).
- Pembaca lain: `annotation_tool.py` (set konfirmasi lama, tidak aktif) membaca kunci yang sama dengan cara yang sama dan diberi perbaikan yang sama. `judge_bias_probes.py` mensyaratkan `== "dev"`; kunci yang tidak ada (= test) memang harus ditolak, jadi tidak diubah.
- Klasifikasi seluruh berkas hasil (hanya baca): dari 73 berkas ber-config, 13 berubah dari *development* menjadi *resmi*: keempat berkas resmi baru, dan sembilan pilot arsip `n10` pra-resmi (`results/_archive/2026-10-19_pre-official/`: 4 C, 2 D, 1 A, 2 B). Pilot arsip itu memang sampel uji; tidak ada yang akan dinilai ulang. 60 berkas lain tetap *development*.

### 12c. Dua entri ledger palsu dan koreksinya

- Tiga tes di `test_judge_panel_resume.py` memanggil `reconcile_ledger` tanpa mengisolasi registry dan menulis dua entri ke `results/run_registry.json` (label "dev-stage2 judging p.jsonl (from the stored records)", $0,01371 + $0,01949 = **$0,0332**, 2026-10-26T08:04:09Z). Tidak ada panggilan DeepInfra. Entri itu ikut ter-commit di `e1dd047`.
- Koreksi: fixture `registry` kini `autouse` (tidak ada tes yang boleh menyentuh registry asli) dan satu entri penyesuaian **−$0,0332** (`adjustment: true`, "ADJUSTMENT 3") membatalkannya; riwayat tidak dihapus. Setelah koreksi: total ledger **$1,174691**, sisa **$7,825309** dari $9,00.

### 12d. Langkah berikutnya

Penilaian per sel dengan `--official` (panel-v2, judge-v2.3, 3 juri, 2-dari-3), batas biaya **$2,00 per sel**, dari tag baru `official-judging-start` (dibuat penulis setelah commit). Dry run R1 (1152 panggilan, proyeksi $1,5358, batas $2,00) sudah dilakukan tanpa panggilan API. Metrik sekunder dan `official_report.py` belum dijalankan.


### 12e. R1 (A): penilaian juri resmi selesai (angka sementara, hanya R1)

Status: run juri resmi R1 (Kondisi A) selesai dan pemeriksaan identitas OK; berhenti sesuai instruksi. R5, metrik sekunder, dan `official_report.py` **belum** dijalankan. Semua angka di bagian ini adalah angka sementara khusus R1 (dari berkas `llm/evaluation/results/panel_official_R1_judge-v2.3{,_panel}.jsonl`), bukan keluaran akhir `official_report.py`. Perintah: `judge_panel.py --official --panel panel-v2 --judge-version judge-v2.3 --max-cost-usd 1.91` dari tag `official-judging-start` (`ec81029`).

- **Identitas:** 1152 rekaman, 1152 pasangan (pertanyaan, juri) unik, id sama dengan daftar uji, 0 id dev, 0 id di luar daftar. Tidak ada berkas kegagalan, tidak ada stall atau galat juri.
- **Rekaman usable:** 384/384 untuk setiap juri; tidak usable 0% (batas 2%). Tidak ada konten reasoning dan tidak ada ketidakcocokan model yang dilayani.

| Juri (FAKTUAL / PENUH / SEBAGIAN) | Label |
|---|---|
| Llama-3.3-70B-Turbo | 361 / 19 / 4 |
| DeepSeek-V3.2 | 245 / 129 / 10 |
| Qwen3-235B-A22B-Instruct-2507 | 241 / 116 / 27 |

- **Panel (2 dari 3):** UNRESOLVED 0, tidak lengkap 0, ABSTAIN 0. Label: 285 FAKTUAL, 87 PENUH, 12 SEBAGIAN. Ketiga juri sepakat pada 206 dari 384.
- **Halusinasi (panel, ABSTAIN dikeluarkan):** 99/384 = **25,8%**. Sama bila ABSTAIN dihitung gagal, karena tidak ada ABSTAIN.
- **Pengamatan:** Llama jauh lebih longgar (361 FAKTUAL) daripada DeepSeek dan Qwen, sehingga ia biasanya juri yang kalah suara. Kesepakatan antarjuri (mis. kappa) perlu dilaporkan di metodologi.
- **Biaya:** aktual **$1,2908** vs dry run $1,5358 (16% di bawah), vs perkiraan $1,36 (5% di bawah), vs batas $1,91. Tidak ada panggilan yang dilewati oleh pengaman anggaran. Ledger dibukukan sekali, ditandai dengan nama berkas. Total ledger **$2,4655**, sisa **$6,5345** dari $9,00.
- **Berikutnya:** R5 (C), dry run $1,7642 untuk 1152 panggilan, batas 1,25x = $2,2053.


### 12f. R5 (C): penilaian juri resmi selesai (angka sementara, hanya R5)

Status: run juri resmi R5 (Kondisi C-primary-grounded, α 0,5) selesai dan pemeriksaan identitas OK; berhenti sesuai instruksi. R3, R9, metrik sekunder, dan `official_report.py` **belum** dijalankan. Angka di bagian ini adalah angka sementara khusus R5 (dari `llm/evaluation/results/panel_official_R5_judge-v2.3{,_panel}.jsonl`), bukan keluaran akhir `official_report.py`. Perintah: `judge_panel.py --official --panel panel-v2 --judge-version judge-v2.3 --max-cost-usd 2.2053` dari commit `ea85648` (tidak ada berkas `.py` yang berubah sejak tag `official-judging-start`).

- **Identitas:** 1152 rekaman, 1152 pasangan (pertanyaan, juri) unik, id sama dengan daftar uji, 0 id dev, 0 id di luar daftar. Tidak ada berkas kegagalan, tidak ada stall atau galat juri.
- **Rekaman usable:** 384/384 untuk setiap juri; tidak usable 0% (batas 2%). Tidak ada konten reasoning dan tidak ada ketidakcocokan model yang dilayani.

| Juri | FAKTUAL | PENUH | SEBAGIAN | ABSTAIN |
|---|---|---|---|---|
| Llama-3.3-70B-Turbo | 305 | 13 | 1 | 65 |
| DeepSeek-V3.2 | 193 | 98 | 9 | 84 |
| Qwen3-235B-A22B-Instruct-2507 | 205 | 88 | 8 | 83 |

- **Panel (2 dari 3):** UNRESOLVED 0, tidak lengkap 0, ABSTAIN 72. Label: 242 FAKTUAL, 66 PENUH, 4 SEBAGIAN, 72 ABSTAIN. Ketiga juri sepakat pada 224 dari 384.
- **Halusinasi (panel):** ABSTAIN dikeluarkan **70/312 = 22,4%**; ABSTAIN dihitung gagal **142/384 = 37,0%**. R1 (A) adalah 25,8% pada kedua versi.
- **Catatan penting:**
  1. Kedua angka berbeda karena prompt grounded dapat menolak menjawab (ABSTAIN), sehingga keduanya harus dilaporkan berdampingan.
  2. Penyebut berbeda (312 vs 384), jadi C vs A bukan perbandingan setara di sini; perbandingan berpasangan menjadi tugas `official_report.py`.
  3. Kenaikan kesepakatan tiga juri (206 → 224) sebagian karena ABSTAIN kategori yang mudah disepakati: 64 dari 224 jawaban yang disepakati ketiga juri adalah ABSTAIN (dihitung dari berkas panel). Tanpa itu, 160 dari 312 jawaban non-ABSTAIN (51,3%) disepakati, dibanding 206 dari 384 (53,6%) pada R1.
- **Biaya:** aktual **$1,3451** vs dry run $1,7642 (24% di bawah), vs batas $2,2053. Tidak ada panggilan yang dilewati oleh pengaman anggaran. Ledger dibukukan sekali, ditandai `panel_official_R5_judge-v2.3.jsonl`. Total ledger **$3,8105**, sisa **$5,1895** dari $9,00.
- **Sel tersisa (dry run):** R3 (B) $1,7741, R9 (D) $1,7439, metrik sekunder $1,318 (termasuk buffer 10%).
