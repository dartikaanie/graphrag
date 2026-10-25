# Matriks run resmi: empat run (test sample, n = 384)

Sumber kebenaran: `llm/official_run_matrix.py` (`build_matrix`). Dokumen ini memuat perintah yang dihasilkan modul itu; `llm/tests/test_official_run_matrix.py` gagal bila keduanya berbeda. Keputusan: `docs/DECISION_C_SCORING.md`, Amendment 16 (menggantikan matriks sembilan sel dari Amendment 10; Amendment 10 tidak ditulis ulang). Nilai α hanya dibaca dari `llm/c_graphrag/official_params.json`.

## Status saat ini

`official_params.json` berstatus **`pending_selection`**: `alpha_primary` dan `alpha_ablation` masih `null` karena tahap 2 belum dijalankan. Selama status bukan `selected`:

- `c_graphrag.py --sample-split test` berhenti dengan pesan galat (exit 2) sebelum memuat data atau Neo4j;
- `POST /api/runs` menolak run C pada sampel test (HTTP 400);
- preset **Official (n=384)** di dashboard terkunci.

Setelah tahap 2: isi *Outcome*, *Record* Amendment 10, dan blok `official_params` (nilai dari draf `stage2_report.py`) dalam satu commit, dengan `status: "selected"`. Placeholder `<alpha_primary>` di bawah lalu diganti nilai dari berkas itu.

## Empat run resmi

Instruksi *grounding* dan *citation* AKTIF untuk B, C, dan D. Tidak ada run tanpa grounding dan tidak ada run ablasi C.

| Run tesis | Sel | Label | Kondisi | Grounding | α (C) | Dashboard (Factorial Batch) |
|---|---|---|---|---|---|---|
| R1 | R1 | A | A | — | — | `A` |
| R2 | R3 | B-grounded | B | on | — | `B-grounded` |
| R3 | R5 | C-primary-grounded | C | on | `alpha_primary` (α\*) | `C-primary-grounded` |
| R4 | R9 | D-grounded | D | on | — | `D-grounded` |

- Perbandingan utama yang didaftarkan: **C vs A, C vs B, C vs D** (R5 vs R1, R5 vs R3, R5 vs R9), pada *hallucination rate* panel (ABSTAIN dikeluarkan).
- C v3 dengan `fusion_mode = trust_weighted` (v3 mengabaikan `fusion_mode` untuk ranking, tetapi nilainya masuk `config_hash`; dengan nilai yang sama, run CLI dan dashboard untuk sel yang sama berbagi `config_hash` dan berkas output).
- Parameter C lain dikunci oleh `official_params.json`: `max_hops = 2`, `edge_types` = kelima tipe, `use_author_trust = false`, `accepted_only = false`. Expansion answerable-only (Amendment 9) aktif.
- Leakage: `thread` (Amendment 15) untuk B, C, D: hanya pertanyaan evaluasi dan semua jawabannya yang dikecualikan.
- Condition D memakai retrieval **v4** (`--d-retrieval-version v4`): satu skor cosine (kueri vs Question pemilik jawaban) untuk kedua level; v3 tetap bisa direproduksi.
- Tahap 2 pemilihan α tetap menghasilkan kandidat kedua (`alpha_ablation`) dan `official_params.json` tetap menyimpannya, **tetapi tidak ada run resmi yang memakainya**.

## Perintah CLI (dari root repo)

Perintah mem-*pin* semua parameter yang bisa ditimpa `.env` (`OVERSAMPLE_POOL`, `TOP_K`, `TOKEN_CHUNK_LIMIT`, `N_ANCHOR`, mode leakage, dll.), sehingga isi `.env` tidak bisa mengubah run resmi diam-diam.

**R1 · A** (run tesis R1)

`cd llm/a_pure_llm && python a_baseline_replication.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini`

**R3 · B-grounded** (run tesis R2)

`cd llm/b_rag && python b_condition_b_rag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --corpus kg_answers --require-citation --require-grounding --leakage-mode thread`

**R5 · C-primary-grounded** (run tesis R3)

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_primary> --fusion-mode trust_weighted --require-grounding --expansion-require-answer --leakage-mode thread`

**R9 · D-grounded** (run tesis R4)

`cd llm/d_lightrag && python d_lightrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-low-level 3 --n-high-level 3 --d-retrieval-version v4 --no-require-grounding --leakage-mode thread --allow-legacy-cells`

## Sel legacy (BUKAN bagian evaluasi resmi)

Lima sel dari matriks sembilan sel Amendment 10 tetap ada di kode supaya data dan hash lama dapat direproduksi: R2 B-plain, R4 C-primary-plain, R6 C-ablation-plain, R7 C-ablation-grounded, R8 D-plain. Pada sampel test, CLI menolaknya (exit 2) dan `POST /api/runs` menjawab HTTP 400, kecuali bendera eksplisit `--allow-legacy-cells` (API: `allow_legacy_cells: true`) diberikan. Preset Official, daftar bawaan Factorial Batch, dan "Run all conditions" tidak pernah memulainya; di Factorial Batch sel ini hanya muncul bila kotak "include legacy cells" dicentang. Laporan resmi tidak membuat perbandingan yang membutuhkannya kecuali kedua run-nya ada.

**R2 · B-plain** (LEGACY)

`cd llm/b_rag && python b_condition_b_rag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --corpus kg_answers --require-citation --no-require-grounding --leakage-mode thread --allow-legacy-cells`

**R4 · C-primary-plain** (LEGACY)

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_primary> --fusion-mode trust_weighted --no-require-grounding --expansion-require-answer --leakage-mode thread --allow-legacy-cells`

**R6 · C-ablation-plain** (LEGACY)

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_ablation> --fusion-mode trust_weighted --no-require-grounding --expansion-require-answer --leakage-mode thread --allow-legacy-cells`

**R7 · C-ablation-grounded** (LEGACY)

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_ablation> --fusion-mode trust_weighted --require-grounding --expansion-require-answer --leakage-mode thread --allow-legacy-cells`

**R8 · D-plain** (LEGACY)

`cd llm/d_lightrag && python d_lightrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-low-level 3 --n-high-level 3 --d-retrieval-version v4 --require-grounding --leakage-mode thread`

## Padanan di dashboard

Halaman **Factorial Batch** → preset **Official (n=384)** → *Confirm & Launch*. Preset membaca `official_params.json` lewat `GET /api/config/defaults` (`official_c_params`, dan `official_run_matrix` = keempat sel dalam bentuk JSON), mengisi `c_retrieval_version = v3`, `alpha (C-primary)`, `n_sample = 384`, `sample_split = test`, lalu menyusun **4 sel** (`A`, `B-grounded`, `C-primary-grounded`, `D-grounded`). Satu sel bisa juga diluncurkan langsung dengan `POST /api/runs` memakai objek `dashboard` dari `official_run_matrix`. Di dashboard, mode leakage dan saklar expansion diambil dari `LEAKAGE_*` / `EXPANSION_REQUIRE_ANSWER` (env); nilai bawaannya sama dengan yang di-*pin* di CLI.

## Catatan P0-4: perbedaan prompt A vs B/C/D adalah bagian dari intervensi

Struktur 4-turn (replikasi Da Silva dkk.) sama di semua kondisi. Untuk B, C, dan D, selain konteks dan instruksi di turn 4, ada dua tambahan teks (`llm/prompts.py`):

1. **System turn (turn 1)** ditambah `GROUNDED_SYSTEM_ADDITION`, berisi aturan mengutip label `[SO-<id>]` dan larangan membuat klaim di luar konteks.
2. **Reminder di akhir turn 4** (`GROUNDED_REMINDER`).

Keduanya identik byte-per-byte di B, C, dan D (`llm/tests/test_prompt_parity_v3.py`) dan merupakan bagian dari intervensi retrieval + constraint, bukan perbedaan tersembunyi. Kondisi A memakai system turn asli tanpa tambahan. Varian tanpa grounding (`NON_GROUNDED_*`) masih ada di kode untuk sel legacy. Prompt dan `PROMPT_VERSION` (`v3`) tidak diubah oleh dokumen ini.
