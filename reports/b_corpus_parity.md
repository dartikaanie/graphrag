# Paritas korpus dan teks query Kondisi B dengan Kondisi C

Tanggal: 2026-10-17 · Branch: `fix/b-corpus-parity` · Draf amandemen: [`b_corpus_parity_amendment_draft.md`](b_corpus_parity_amendment_draft.md)

## 1. Ringkasan

- Kondisi B kini mengambil konteks dari **seluruh 725.795 Answer node** di KG yang sama dengan Kondisi C. Sebelumnya B hanya memakai indeks 7.257 jawaban, yaitu sampel `INDEX_POOL = 8000` dengan 10.681 chunk.
- Representasi teks jawaban B identik dengan C: satu vektor per jawaban, `Q: <judul>\nA: <body>`, dipotong ke 400 token oleh fungsi yang sama.
- Teks query B identik dengan C: `f"{title} {str(body)[:1000]}"`.
- Exclusion leakage B kini dilakukan per query, sama seperti C.
- B tetap retrieval FAISS murni (`IndexFlatIP`, exact search). Neo4j hanya dibaca sekali, offline, sebagai sumber ekspor teks. Saat runtime B tidak menyentuh Neo4j, dan ini dibuktikan oleh test.
- Perilaku C tidak berubah. Retrieval C pada 12 pertanyaan dev × 4 konfigurasi identik dengan snapshot sebelum refactor.
- Dev run B (n = 35, posisi 385–419):
  - B-plain: cosine rata-rata **0,591** (median 0,602), NF2 74,3%.
  - B-grounded: cosine rata-rata **0,568** (median 0,600), NF2 77,1%.
  - Konteks selalu 5 item, tidak ada yang berasal dari thread pertanyaan itu sendiri.
- Tidak ada run pada test split. Evaluasi resmi n = 384 tidak dijalankan.

## 2. Temuan investigasi (sebelum kode diubah)

### 2.1 Isi KG (`graphrag`, read-only)
| Ukuran | Nilai |
|---|---|
| Answer node | 725.795 |
| Question node | 2.685.814 |
| Answer dengan parent `HAS_ANSWER` / `HAS_ACCEPTED_ANSWER` | 454.807 / 270.988 (jumlah = semua Answer) |
| Answer tanpa parent (yatim) | **0** |
| Answer dengan > 1 parent | 0 |
| `a.body` null / `q.title` null (parent) | 0 / 0 |
| `isAccepted = true` | 270.988, sama persis dengan `HAS_ACCEPTED_ANSWER` |

Karena tidak ada jawaban yatim, tidak ada keputusan perlakuan yang perlu diambil. Skrip ekspor tetap dibuat untuk berhenti (abort) dengan hitungan bila kelak ada jawaban yatim atau jawaban dengan dua parent.

### 2.2 Teks kandidat C
`fuse_and_rank()` membentuk `Q: {via_question_title}\nA: {answer_body}` lalu memotongnya dengan `_truncate_doc_text` (cl100k_base, `token_chunk_limit` = 400). Ini berlaku di cabang v2 maupun v3.

`via_question` selalu parent langsung dari jawaban, karena 1-hop, 2-hop, dan expansion semuanya mengambil jawaban lewat `HAS_ANSWER|HAS_ACCEPTED_ANSWER` dari Question tersebut. Setiap jawaban punya tepat satu parent, sehingga teks per `answer_id` unik dan dapat diekspor.

### 2.3 Format teks query
| Jalur | Format | Sumber teks | Encode |
|---|---|---|---|
| B (lama) | `f"{Title}\n{Body}"` (body lengkap) | parquet | `model.encode` default device (MPS), `faiss.normalize_L2` |
| C, fallback `embed_query` | `f"{Title} {str(Body)[:1000]}"` | parquet | CPU, 1 teks, normalisasi numpy |
| C, jalur utama (precomputed) | `f"{title or ''} {(body or '')[:1000]}"` | Neo4j `q.title`/`q.body` (skrip 11) | MPS, batch 64 |
| `11_densify_embedding_similarity.py` | sama dengan baris di atas | Neo4j | MPS, batch 64 |
| D | sama dengan C (precomputed + fallback `f"{Title} {str(Body)[:1000]}"`) | — | — (tidak diubah) |

Format string C dan skrip 11 identik. Satu-satunya perbedaan adalah penanganan nilai NULL: skrip 11 memetakan NULL ke `""`, sedangkan C ke `"None"`. Perbedaan ini dilaporkan dan tidak "dibetulkan", karena C dipertahankan apa adanya.

Pada 35 pertanyaan dev tidak ada Title atau Body yang NULL. Teks Neo4j sama dengan teks parquet (35/35), dan selisih maksimum vektor precomputed C terhadap encode ulang di CPU adalah **1,8e-7**.

**Format query final B**: `build_query_text(title, body) = f"{title} {str(body)[:1000]}"` (`llm/common/text.py`), di-encode satu teks per panggilan di CPU, dengan normalisasi numpy yang sama seperti `c_graphrag.embed_query`.

### 2.4 Pemanggil yang terdampak
- CLI B: `main()`, `process_sample()`.
- Dashboard: `backend/app/services/engine_service.py::run_condition_b`. Sebelumnya fungsi ini tidak punya leakage check sama sekali.
- `llm/evaluation/calibration_generate.py`: CLI B plus `build_config`/`build_output_path`.
- `llm/c_graphrag/stage1_sweep.py`: memakai `cg._truncate_doc_text`. Nama ini dipertahankan sebagai alias.
- Test lama: `test_sample_split.py`, `test_config_hash_backward_compatibility.py`, `test_engine_service_factorial_e2e.py`.
- `backend/app/process/registry.py`: kutipan docstring B.

## 3. Yang diubah

| File | Perubahan |
|---|---|
| `llm/common/text.py` (baru) | `build_query_text`, `answer_doc_text`, `truncate_doc_text` (dipindah tanpa perubahan dari C) |
| `llm/common/leakage.py` (baru) | `build_exclusion`, `LeakageExclusion`, `filter_excluded`, `assert_no_leakage`. Tanpa Neo4j, input berupa id. |
| `llm/b_rag/build_kg_answer_index.py` (baru) | `--stage export|encode|index|all`. Satu-satunya kode B yang membuka Neo4j (read-only, keyset pagination). Encode dapat di-resume. |
| `llm/b_rag/kg_corpus.py` (baru) | Layout file, cache key, loader runtime, `search()` dengan exclusion per query, lookup teks lewat DuckDB. Tidak meng-import neo4j. |
| `llm/b_rag/b_condition_b_rag.py` | `--corpus kg_answers` (default) / `legacy_sample`; query lewat `build_query_text`; exclusion dan assert per query; `assert_index_has_no_eval_leakage` dihapus; `config["corpus"]`; field `query_text` di setiap record; docstring. |
| `llm/c_graphrag/c_graphrag.py` | Hanya memanggil `build_query_text`, `answer_doc_text`, `truncate_doc_text`, `build_exclusion`. Perilaku identik. |
| `llm/d_lightrag/d_lightrag.py` | Hanya `build_exclusion`; set yang dihasilkan identik. Format query D tidak diubah. |
| `backend/app/services/engine_service.py`, `history_service.py`, `process/registry.py` | Dashboard memuat indeks KG (cache LRU, ikut dibersihkan oleh `release_faiss_cache`); menyimpan `corpus` dan `corpus_cache_key` di riwayat; parameter `cond.b_corpus`. |
| `llm/evaluation/calibration_generate.py` | Dikunci ke `--corpus legacy_sample`, karena batch kalibrasi memang dibuat dengan korpus lama. |
| `.gitignore` | Mengabaikan `llm/b_rag/index_cache/kg_answers_*/` serta `*.parquet`, `*.f32`, dan `*.i64` di bawahnya. |

**Kompatibilitas config hash.** Kunci `"corpus": "kg_answers_v1_tok400"` hanya ditambahkan untuk korpus KG. `config_hash` semua run lama (korpus legacy) tidak berubah, dan dibuktikan oleh `test_config_hash_backward_compatibility.py`. Run baru tidak akan pernah melakukan resume ke file run lama.

**Cache key indeks**: `kg_answers_v1_tok400_all-MiniLM-L6-v2_358aaed6804d`. Bagian akhirnya adalah 12 karakter pertama dari sha256 daftar `answer_id` (int64 little-endian, urut menurut id).

## 4. Korpus lama vs baru

| | Lama (`legacy_sample`) | Baru (`kg_answers`) |
|---|---|---|
| Sumber | sampel acak parquet (`INDEX_POOL` 8.000, seed 42) | semua Answer node KG (ekspor Neo4j) |
| Jawaban | 7.257 | **725.795** (100× lebih banyak) |
| Pertanyaan induk | 7.238 | 602.900 |
| Vektor | 10.681 chunk (multi-chunk) | 725.795 (1 per jawaban; 27,5% terpotong ke 400 token) |
| Teks query | `Title\nBody` | `Title + " " + Body[:1000]` (= C) |
| Leakage | global saat build | per query (= C) |
| Ukuran indeks | 16 MB + meta pickle 11 MB | 1.115 MB + meta parquet 4 MB |

## 5. Build: waktu, ukuran, RAM

| Tahap | Perkiraan | Realisasi | Puncak RSS |
|---|---|---|---|
| Export (Neo4j → parquet) | ±5–6 menit (dari uji 20.000 baris) | **279,6 detik (4,7 menit)**, ±2.600–3.100 baris/detik | 228 MB |
| Encode (MPS, batch 64, unit 4.096) | ±70 menit (benchmark 173 teks/detik pada 2.048 teks awal) | **7.708,7 detik (2 jam 8 menit)**, rata-rata ±94 teks/detik | 500 MB |
| Index (`IndexFlatIP` + meta) | beberapa menit | **27,1 detik** | 1.342 MB (proses encode+index: 1.407 MB) |

Encode lebih lambat dari perkiraan karena jawaban dengan id lebih besar cenderung lebih panjang, dan sebagian waktu berjalan bersamaan dengan test.

**Pilihan device dan batch.** C meng-encode kandidat satu per satu di CPU (batch_size = 1), karena embedding sebuah teks bergeser ±1e-7 tergantung isi batch-nya. Vektor dokumen B tidak pernah dibandingkan dengan vektor C, jadi B memakai batch 64 di MPS (2,2× lebih cepat dari CPU: 173 vs 80 teks/detik).

Determinisme dijamin dengan cara lain:
- Setiap pemanggilan `encode()` selalu menerima unit yang sama persis (4.096 `answer_id` berurutan).
- Resume selalu dimulai di batas unit.
- Checkpoint menolak resume bila batch, device, atau ukuran unit berbeda.

Pengulangan pada MPS dan CPU sama-sama bit-identik. Selisih MPS terhadap CPU 2,7e-7, tidak relevan karena B memakai satu device secara konsisten. Vektor query B dihitung di CPU, sama seperti C.

**Uji resume pada data asli.** Proses dihentikan (kill) di 20.480/725.795, lalu perintah yang sama dijalankan ulang dan melanjutkan dari checkpoint. Setelah selesai, unit 0, 4, 5 (dimulai tepat di titik interupsi), 6, dan 177 (terakhir) di-encode ulang di proses baru. Hasilnya **bit-identik** dengan memmap, dan id-nya sama.

**Ukuran file** (`llm/b_rag/index_cache/kg_answers_v1_tok400/`, tidak di-commit):

| File | Ukuran |
|---|---|
| `kg_answers.parquet` | 694 MB |
| `all-MiniLM-L6-v2/embeddings.f32` | 1.115 MB |
| `all-MiniLM-L6-v2/ids.i64` | 5,8 MB |
| `…_358aaed6804d.faiss` | 1.115 MB |
| `…_meta.parquet` | 4,0 MB |
| Total | ±2,9 GB |

**RAM runtime B** (CLI, `/usr/bin/time -l`): puncak RSS 1,88 GB (B-plain) dan 1,75 GB (B-grounded). Angka ini mencakup indeks flat 1,1 GB, model, torch, dan DuckDB untuk sampling. Masih aman untuk RAM 8 GB, tetapi dashboard backend akan menahan 1,1 GB indeks ini di samping cache FAISS C/D sampai "release FAISS cache" dipanggil.

## 6. Test (pytest)

Seluruh suite: **966 passed, 2 skipped**. Dua test yang di-skip memerlukan `GRAPHRAG_INTEGRATION=1` dan sudah dijalankan terpisah dengan hasil lulus (22 passed).

| Persyaratan | Test | Hasil |
|---|---|---|
| Jumlah vektor = baris parquet = Answer node yang diekspor | `test_b_c_parity::test_vectors_eq_parquet_rows_eq_exported_answer_nodes` | 725.795 = 725.795 = 725.795 |
| Teks kandidat B = C untuk `answer_id` yang sama | `test_candidate_text_identical_between_b_and_c` | semua `answer_id` yang diambil C pada snapshot (12 q × 4 konfigurasi), sha256 sama |
| Teks query B = C, ≥ 20 pertanyaan dev | `test_query_text_identical_b_vs_c`, `test_query_text_matches_script11_neo4j_text` | 35/35 sama dengan ekspresi C; 12/12 sama dengan string yang benar-benar di-embed C sebelum refactor; 35/35 sama dengan teks skrip 11 dari Neo4j |
| Embedding query B = C (toleransi 1e-6) | `test_query_embedding_b_equals_c` | sama dengan `embed_query` C (fallback); terhadap vektor precomputed C maks **1,82e-7** |
| Perilaku C tidak berubah, ≥ 10 pertanyaan dev | `llm/c_graphrag/tests/test_c_behavior_unchanged.py` | 12 q × {v2, v3 α = 0,5} × {precomputed, embed_query}: teks `embed_query`, id, skor, dan sha256 teks konteks identik dengan snapshot sebelum refactor. Snapshot itu sendiri terbukti deterministik (diambil ulang sebelum refactor: identik). |
| ≥ 20 pertanyaan dev: tanpa thread sendiri, jumlah konteks = `top_k` | `test_dev_results_exclude_own_thread_and_keep_top_k` | 35/35 berisi 5 konteks dan bersih. **Tanpa exclusion, 31/35 akan mengambil thread-nya sendiri.** |
| Resume encoding identik | `test_encode_resume_is_bit_identical` (+ uji data asli di §5) | bit-identik; resume dengan parameter berbeda ditolak |
| B tidak menyentuh Neo4j | `test_b_no_neo4j_runtime.py` | AST: tidak ada `import neo4j` di jalur B; import B tidak memuat `neo4j`; `process_sample` berhasil dengan modul `neo4j` diracuni (setiap akses gagal) dan tanpa variabel `NEO4J_*` |

## 7. Dev run (n = 35, posisi 385–419, gpt-4o-mini, top_k = 5)

| | B-plain baru | B-plain lama¹ | B-grounded baru | B-grounded lama¹ |
|---|---|---|---|---|
| Cosine rata-rata / median | **0,591 / 0,602** | 0,604 / 0,605 | **0,568 / 0,600** | 0,555 / 0,564 |
| % cosine > 0,5 | 74,3% | — | 68,6% | — |
| Pertanyaan dengan cosine baru > lama | 14/35 | | 20/35 | |
| NF2 (≥ 1 kutipan valid) | 74,3% | 45,7% | 77,1% | 34,3% |
| Skor retrieval top-1 rata-rata | 0,697 | 0,575 | 0,697 | 0,575 |
| Accepted answer dalam 5 konteks (rata-rata) | 1,94 | n/a | 1,94 | n/a |
| Latency retrieval rata-rata / median | 0,373 / 0,276 detik | 0,106 / 0,061 detik | 0,280 / 0,252 detik | 0,076 / 0,059 detik |
| Konteks = 5 / thread sendiri | 35/35 / 0 | | 35/35 / 0 | |

¹ "Lama" = 35 pertanyaan yang sama, diambil dari run dev n = 50 dengan korpus `legacy_sample` (2026-10-13).

File hasil:
- `results/_archive/2026-10-19_pre-official/llm/b_rag/results/condition_b_openai_gpt-4o-mini_n35_seed42_plain_dev_v3_c8fb5cfbe0.jsonl`
- `results/_archive/2026-10-19_pre-official/llm/b_rag/results/condition_b_openai_gpt-4o-mini_n35_seed42_grounded_dev_v3_0e2be4937f.jsonl`

Catatan pembacaan:
- **Cosine similarity hampir tidak berubah** (±0,01, dalam kisaran noise untuk n = 35).
- Retrieval jauh lebih relevan: skor top-1 naik dari 0,575 ke 0,697, dan rata-rata 1,9 accepted answer per konteks.
- NF2 naik tajam.
- Latency naik karena exact search atas 725 ribu vektor ditambah lookup DuckDB, tetapi tetap jauh di bawah batas NF3. Pertanyaan pertama ±4,2 detik karena cache dingin.
- Ini informasi saja, bukan dasar seleksi.

**Pilot lama** (test items, n = 10, korpus 8.000, 2026-10-05): B-plain cosine 0,653, NF2 20%; B-grounded cosine 0,623, NF2 30%. Angka ini memakai **test items** dan korpus sampel 8.000, sehingga tidak dapat dibandingkan langsung dengan dev run di atas maupun dengan run resmi berikutnya. Test split tidak disentuh dalam tugas ini.

**Contoh 3 konteks teratas** (B-plain, Q 20952138 "Best way to convert RTMP to MP4", cosine jawaban 0,692):

| skor | answer_id | pertanyaan induk | accepted |
|---|---|---|---|
| 0,781 | 29048319 | 29044491 "How to embed RTMP live-stream?" | ya |
| 0,779 | 12101922 | 11928264 "Play RTMP live stream in HTML5" | tidak |
| 0,762 | 11928297 | 11928264 "Play RTMP live stream in HTML5" | tidak |

Contoh kedua (Q 55753657 "How to have useEffect() act like componentDidMount()?", cosine 0,740): tiga konteks teratas (skor 0,678 / 0,672 / 0,665) berasal dari tiga thread lain tentang `useEffect`/lifecycle React.

## 8. Catatan dan keputusan terbuka

1. **Exclusion baru mencakup** q, semua jawaban di thread q, dan accepted answer q. Duplicate dan Linked (`IS_RELATED_TO`) maupun near-duplicate belum dikecualikan di B, C, maupun D.
   - Dev-50 stage 1: 0% punya Duplicate, 12% punya Linked (6% dengan tetangga yang punya jawaban).
   - Sampel uji 384, hanya angka agregat: 0,5% punya Duplicate, 13,8% punya Linked (8,3% dengan tetangga yang punya jawaban).
2. `semantic_expansion()` di C tidak pernah memakai buffer 6× atau verifikasi keberadaan jawaban. Fix tersebut tidak ada di riwayat git mana pun.
3. Bila kandidat B habis (secara praktis mustahil pada 725 ribu vektor), `search()` mengembalikan kurang dari `top_k` tanpa peringatan. Jumlah sebenarnya tetap tercatat di `n_context_items_used`.
4. `llm/c_graphrag/reputation_ablation_check.py` masih punya salinan exclusion dan teks query sendiri. Skrip ini belum memakai modul bersama.
5. Draf amandemen (bahasa Indonesia) ada di `b_corpus_parity_amendment_draft.md` dan **belum** dimasukkan ke `docs/DECISION_C_SCORING.md`.

## 9. Paragraf untuk Bab III

> Untuk memastikan bahwa perbedaan kinerja antara Kondisi B (RAG konvensional) dan Kondisi C (GraphRAG berbobot trust) hanya berasal dari mekanisme retrieval, kedua kondisi memakai kumpulan jawaban, representasi teks jawaban, dan format query yang sama. Korpus Kondisi B adalah seluruh 725.795 simpul Answer pada knowledge graph yang juga ditelusuri Kondisi C. Teks seluruh jawaban diekspor sekali secara offline dari Neo4j, dan Kondisi B tidak mengakses graf sama sekali saat retrieval. Setiap jawaban direpresentasikan sebagai satu dokumen "Q: <judul pertanyaan induk> A: <isi jawaban>" yang dipotong pada 400 token (tokenizer cl100k_base) dengan fungsi yang sama dengan pembentuk konteks Kondisi C. Teks query dibentuk dari judul pertanyaan ditambah 1.000 karakter pertama isinya, persis seperti query Kondisi C, dan di-embed dengan model all-MiniLM-L6-v2 yang sama. Kedua kondisi juga menerapkan aturan pencegahan kebocoran data yang sama pada setiap query: pertanyaan yang sedang dijawab beserta seluruh jawabannya tidak pernah dapat muncul sebagai konteks. Dengan demikian, perbedaan antara kedua kondisi terbatas pada mekanisme retrieval dan ranking. Kondisi B memeringkat jawaban semata-mata berdasarkan kemiripan kosinus melalui pencarian vektor eksak (FAISS), sedangkan Kondisi C menjangkau kandidat melalui anchoring, penelusuran graf, dan semantic expansion, lalu memeringkatnya dengan kombinasi kemiripan dan skor trust.
