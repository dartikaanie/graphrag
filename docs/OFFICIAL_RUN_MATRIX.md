# Matriks run resmi R1–R9 (test sample, n = 384)

Sumber kebenaran: `llm/official_run_matrix.py` (`build_matrix`). Dokumen ini memuat perintah yang dihasilkan modul itu; `llm/tests/test_official_run_matrix.py` gagal bila keduanya berbeda. Keputusan: `docs/DECISION_C_SCORING.md`, Amendment 10. Nilai α hanya dibaca dari `llm/c_graphrag/official_params.json`.

## Status saat ini

`official_params.json` berstatus **`pending_selection`**: `alpha_primary` dan `alpha_ablation` masih `null` karena tahap 2 belum dijalankan. Selama status bukan `selected`:

- `c_graphrag.py --sample-split test` berhenti dengan pesan galat (exit 2) sebelum memuat data atau Neo4j;
- `POST /api/runs` menolak run C pada sampel test (HTTP 400);
- preset **Official (n=384)** di dashboard terkunci.

Setelah tahap 2: isi *Outcome*, *Record* Amendment 10, dan blok `official_params` (nilai dari draf `stage2_report.py`) dalam satu commit, dengan `status: "selected"`. Placeholder `<alpha_primary>` / `<alpha_ablation>` di bawah lalu diganti nilai dari berkas itu (jalankan ulang `build_matrix` atau baca berkasnya).

## Matriks

| Run | Label | Kondisi | Grounding | α (C) | Dashboard (sel Factorial Batch) |
|---|---|---|---|---|---|
| R1 | A | A | — | — | `A` |
| R2 | B-plain | B | off (plain) | — | `B-plain` |
| R3 | B-grounded | B | on (grounded) | — | `B-grounded` |
| R4 | C-primary-plain | C | off (plain) | `alpha_primary` (α\*) | `C-primary-plain` |
| R5 | C-primary-grounded | C | on (grounded) | `alpha_primary` (α\*) | `C-primary-grounded` |
| R6 | C-ablation-plain | C | off (plain) | `alpha_ablation` | `C-ablation-plain` |
| R7 | C-ablation-grounded | C | on (grounded) | `alpha_ablation` | `C-ablation-grounded` |
| R8 | D-plain | D | off (plain) | — | `D-plain` |
| R9 | D-grounded | D | on (grounded) | — | `D-grounded` |

- **C-primary** = pemenang tahap 2 (α\*); **C-ablation** = kandidat tahap 2 yang lain. Kandidatnya α ∈ {0,25; 0} (Amendment 9), jadi bila α\* = 0, C-ablation memakai α = 0,25.
- Semua sel C: C retrieval **v3**, `fusion_mode = trust_weighted`. v3 mengabaikan `fusion_mode` untuk ranking, tetapi nilainya masuk `config_hash`; dengan nilai yang sama, run CLI dan dashboard untuk sel yang sama berbagi `config_hash` dan berkas output.
- Parameter C lain dikunci oleh `official_params.json`: `max_hops = 2`, `edge_types` = kelima tipe, `use_author_trust = false`, `accepted_only = false`. Expansion answerable-only (Amendment 9) aktif.
- Leakage: `dup+neardup0.95` (Amendment 8) untuk B, C, D.

## Perintah CLI (dari root repo)

Perintah mem-*pin* semua parameter yang bisa ditimpa `.env` (`OVERSAMPLE_POOL`, `TOP_K`, `TOKEN_CHUNK_LIMIT`, `N_ANCHOR`, mode leakage, dll.), sehingga isi `.env` tidak bisa mengubah run resmi diam-diam.

**R1 · A**

`cd llm/a_pure_llm && python a_baseline_replication.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini`

**R2 · B-plain**

`cd llm/b_rag && python b_condition_b_rag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --corpus kg_answers --require-citation --no-require-grounding --leakage-mode dup+neardup0.95`

**R3 · B-grounded**

`cd llm/b_rag && python b_condition_b_rag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --corpus kg_answers --require-citation --require-grounding --leakage-mode dup+neardup0.95`

**R4 · C-primary-plain**

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_primary> --fusion-mode trust_weighted --no-require-grounding --expansion-require-answer --leakage-mode dup+neardup0.95`

**R5 · C-primary-grounded**

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_primary> --fusion-mode trust_weighted --require-grounding --expansion-require-answer --leakage-mode dup+neardup0.95`

**R6 · C-ablation-plain**

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_ablation> --fusion-mode trust_weighted --no-require-grounding --expansion-require-answer --leakage-mode dup+neardup0.95`

**R7 · C-ablation-grounded**

`cd llm/c_graphrag && python c_graphrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-anchor 3 --n-semantic-expansion 3 --c-retrieval-version v3 --alpha <alpha_ablation> --fusion-mode trust_weighted --require-grounding --expansion-require-answer --leakage-mode dup+neardup0.95`

**R8 · D-plain**

`cd llm/d_lightrag && python d_lightrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-low-level 3 --n-high-level 3 --no-require-grounding --leakage-mode dup+neardup0.95`

**R9 · D-grounded**

`cd llm/d_lightrag && python d_lightrag.py --n-sample 384 --seed 42 --oversample-pool 1536 --sample-split test --provider openai --model gpt-4o-mini --top-k 5 --token-chunk-limit 400 --n-low-level 3 --n-high-level 3 --require-grounding --leakage-mode dup+neardup0.95`

## Padanan di dashboard

Halaman **Factorial Batch** → preset **Official (n=384)** → *Confirm & Launch*. Preset membaca `official_params.json` lewat `GET /api/config/defaults` (`official_c_params`, dan `official_run_matrix` = tabel di atas dalam bentuk JSON), mengisi `c_retrieval_version = v3`, `alpha (C-primary)`, `alpha (C-ablation)`, `n_sample = 384`, `sample_split = test`, lalu menyusun 9 sel dengan label seperti kolom terakhir tabel. Satu sel bisa juga diluncurkan langsung dengan `POST /api/runs` memakai objek `dashboard` dari `official_run_matrix`. Di dashboard, mode leakage dan saklar expansion diambil dari `LEAKAGE_*` / `EXPANSION_REQUIRE_ANSWER` (env); nilai bawaannya sama dengan yang di-*pin* di CLI.

## Catatan P0-4: perbedaan prompt A vs B/C/D adalah bagian dari intervensi

Struktur 4-turn (replikasi Da Silva dkk.) sama di semua kondisi. Untuk B, C, dan D, selain konteks dan instruksi di turn 4, ada dua tambahan teks (`llm/prompts.py`):

1. **System turn (turn 1)** ditambah `GROUNDED_SYSTEM_ADDITION` (run grounded) atau `NON_GROUNDED_SYSTEM_ADDITION` (run plain dengan konteks), berisi aturan mengutip label `[SO-<id>]` dan, untuk grounded, larangan membuat klaim di luar konteks.
2. **Reminder di akhir turn 4** (`GROUNDED_REMINDER` / `NON_GROUNDED_REMINDER`).

Keduanya identik byte-per-byte di B, C, dan D (`llm/tests/test_prompt_parity_v3.py`) dan merupakan bagian dari intervensi retrieval + constraint, bukan perbedaan tersembunyi. Kondisi A memakai system turn asli tanpa tambahan. Prompt dan `PROMPT_VERSION` (`v3`) tidak diubah oleh dokumen ini; Bab III/IV perlu menyebut kedua tambahan ini secara eksplisit.
