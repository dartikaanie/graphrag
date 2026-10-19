# Semantic expansion hanya memilih Question yang punya jawaban (Amendment 9)

Tanggal: 2026-10-18 · Branch: `fix/expansion-answered` · Prompt: `docs/Agent Prompt leakage review and expansion.md` (Bagian 2) · Draf amandemen: `reports/expansion_amendment9_draft_en.md` / `reports/expansion_amendment9_draft.md`

## 1. Ringkasan

- Dengan switch aktif, **slot expansion kosong turun dari 64,1% menjadi 0%** (277 → 0 dari 432 Question expansion pada 50 pertanyaan dev). Ini berlaku di semua α.
- **Item konteks dari expansion di top-5 kira-kira dua kali lipat**: misalnya 36 → 82 pada α = 0,5 dan 37 → 88 pada α = 0,25. Pertanyaan dengan konteks kurang dari 5 item turun dari 5 menjadi 2.
- **Anchor dan kandidat traversal tidak berubah.** Ini dicek per pertanyaan pada 20 pertanyaan dev (test integrasi) dan pada snapshot 12 pertanyaan. Hanya tahap expansion, dan karena itu top-5 hasil fusi, yang berubah.
- **Cek ulang stage-1** dijalankan dengan protokol yang ditetapkan sebelumnya: dev 385–434 (n = 50), α ∈ {0; 0,25; 0,5; 0,75; 1}, ctxrel-v2 primary, temperature 0, `select_stage1_candidates` tanpa perubahan.
  - Switch off mereproduksi hasil Amendment 6 persis, baik top-5 (50 × 5) maupun metriknya.
  - Switch on: **top_two berubah dari [0,5; 0,25] menjadi [0,25; 0,0]**, keep_current = False.
  - Sesuai pra-registrasi, hasil ini diterima apa adanya.
- Biaya judge: 119 item baru, $0,0137 aktual (estimasi $0,0141, batas $0,50).

## 2. Perubahan kode

- `llm/c_graphrag/c_graphrag.py`
  - `_select_answerable_expansion()`, dipanggil dari `semantic_expansion(require_answer=True)`:
    - per seed, mengambil `search_n = 20·top_n + |exclude| + top_n·i` kandidat FAISS;
    - satu round-trip Cypher `EXISTS { (q)-[:fetch]->(a:Answer) WHERE NOT a.id IN $exclude_a }` untuk kandidat semua seed;
    - pemilihan berurutan per seed (similarity turun, tie-break id naik, tanpa duplikat lintas seed).
  - Record mencatat `n_expansion_requested`, `n_expansion_answerable`, `anchor_question_ids`, dan `graph_candidate_answer_ids`.
  - Setting `EXPANSION_REQUIRE_ANSWER` (pydantic-settings, default True).
  - Kunci config `expansion_require_answer: true` dan `expansion_overfetch: 20` hanya ditulis bila setting ini True, jadi config_hash run lama tidak berubah.
  - CLI `--expansion-require-answer`.
- `stage1_sweep.py`, dashboard (`engine_service.py`): mengikuti setting ini.
  - Ikut diperbaiki: `build_output_path` C di dashboard sebelumnya tidak meneruskan `leakage_mode`, sehingga nama filenya tidak sesuai dengan config_hash.
- `reputation_ablation_check.py`, `investigate_zero_context.py`: sengaja tetap memakai expansion lama, dengan catatan di docstring.
- Kondisi D tidak berubah.

### Perbaikan stabilitas (tidak mengubah hasil)

Run integrasi pertama crash dengan `Bus error` di dalam FAISS `search`. Dua penyebab yang mungkin sudah ditutup:

1. **Satu salinan indeks pertanyaan per proses pytest.**
   - Indeksnya `IndexIVFFlat`, 2.685.814 vektor, 4,15 GB, dan dibaca penuh ke RAM.
   - `take_snapshot()` menerima indeks yang sudah dimuat. Fixture `question_faiss` (scope session) di `llm/c_graphrag/tests/conftest.py` dipakai oleh semua test integrasi.
   - Selama sesi integrasi, sebuah spy pada `faiss.read_index` menggagalkan test bila indeks pertanyaan dibaca untuk kedua kalinya.
2. **Vektor query disalin.** Vektor query dari memmap `question_embeddings.f32` sekarang menjadi salinan float32 sendiri (`llm/common/question_index.py: query_vector`), bukan view memmap yang halamannya dibaca dari drive eksternal saat search.
   - Berlaku di C, D, `build_exclusion_sets.py`, dan kedua skrip expansion lama.

Selain itu, `read_question_index()` mengecek bahwa `nprobe` yang tersimpan di file indeks adalah 32 (nilai dari script 11, sudah diverifikasi). Nilai lain menghentikan run, dan `nprobe` tidak di-set otomatis.

Bukti bahwa hasil tidak berubah: semua snapshot C identik setelah perbaikan. Yang dicek adalah fixture sebelum refactor (mode thread dan mode leakage default), expansion OFF terhadap fixture lama, dan expansion ON terhadap fixture Amendment 9.

## 3. Hasil cek ulang stage-1

Laporan mesin: `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.{md,json}`. Run dilakukan pada 2026-10-18T04:29:53Z di atas commit `38c4f6f` ditambah perubahan branch ini yang belum di-commit.

| α | % q ≥1 RELEVANT (lama → baru) | skor rata-rata (lama → baru) | % accepted terpilih (lama → baru) | pertanyaan berubah |
|---|---|---|---|---|
| 0 | 48,0 → 50,0 | 0,3263 → 0,3525 | 37,29 → 38,52 | 31 / 50 |
| 0,25 | 48,0 → 48,0 | 0,3305 → 0,3504 | 41,95 → 47,13 | 34 / 50 |
| 0,5 | 48,0 → 48,0 | 0,3157 → 0,3299 | 53,39 → 61,89 | 38 / 50 |
| 0,75 | 46,0 → 44,0 | 0,2818 → 0,2848 | 66,95 → 76,23 | 35 / 50 |
| 1 | 34,0 → 28,0 | 0,2225 → 0,2029 | 72,88 → 85,66 | 33 / 50 |

- **Peringkat dengan aturan yang sama:** 0,25; 0,0; 0,5; 0,75; 1,0.
  - α = 0, 0,25, dan 0,5 seri pada kriteria utama (selisih ≤ 2 pp).
  - Di antara ketiganya, skor rata-rata 0,5 (0,3299) berada lebih dari 0,02 di bawah 0,0 (0,3525). Akibatnya 0,5 keluar dari dua teratas.
  - 0,25 dan 0,0 seri pada kedua kriteria, dan yang lebih besar (0,25) ditempatkan pertama.
- **top_two = [0,25; 0,0].** keep_current = False, karena α = 1 tidak berperingkat pertama.
- **Validasi reproduksi:**
  - Dengan switch off, top-5 identik dengan hasil Amendment 6 (`results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/logscore_check/stage1_logscore_alpha*.jsonl`, 50 × 5).
  - Metrik yang dihitung ulang identik dengan metrik tercatat, dan top_two lama tetap [0,5; 0,25].
- **Judge:**
  - Label lama dipakai ulang hanya bila (question_id, answer_id, text_hash, model, prompt_version) identik: 405 label dari cache.
  - Ada 119 item baru: 54 / 55 / 57 / 56 / 67 per α, dengan sebagian item muncul di lebih dari satu α.
  - Item baru dinilai oleh `meta-llama/Llama-3.3-70B-Instruct` (DeepInfra, disajikan sebagai `-Turbo` seperti seluruh label cache), `ctxrel-v2`, temperature 0.
  - 0 call gagal, 0 parse error, semuanya selesai dalam 1 percobaan.
  - Label baru: 15 RELEVANT, 21 PARTIAL, 83 IRRELEVANT.
  - Tersimpan di `results/_archive/2026-10-19_pre-official/llm/c_graphrag/results/expansion_check/ctxrel_v2_expansion_primary.jsonl`.

## 4. Retrieval dan latency (50 pertanyaan dev, retrieval-only)

| α | slot expansion kosong | item expansion di top-5 | total item konteks (pertanyaan dengan < 5 item) | latency p95 (detik) |
|---|---|---|---|---|
| 0 | 64,1% → 0% | 43 → 90 | 236 → 244 (5 → 2) | 0,684* → 0,241 |
| 0,25 | 64,1% → 0% | 37 → 88 | 236 → 244 (5 → 2) | 0,100 → 0,102 |
| 0,5 | 64,1% → 0% | 36 → 82 | 236 → 244 (5 → 2) | 0,091 → 0,106 |
| 0,75 | 64,1% → 0% | 37 → 83 | 236 → 244 (5 → 2) | 0,113 → 0,119 |
| 1 | 64,1% → 0% | 42 → 92 | 236 → 244 (5 → 2) | 0,100 → 0,123 |

\* Run pertama dalam proses (cache dingin), jadi angka ini bukan efek dari perubahan. Pada α yang lain, over-fetch dan satu round-trip `EXISTS` menambah latency p95 sekitar 0,01–0,02 detik, jauh di bawah NF3 15 detik.

## 5. Sumber daya

Diukur dengan `/usr/bin/time -l`. Footprint adalah `peak memory footprint` (phys_footprint). RSS di macOS tidak menghitung halaman yang dikompresi, sehingga tidak bisa dipakai sebagai ukuran memori sebenarnya.

| Proses | Peak footprint | RSS maks | Durasi |
|---|---|---|---|
| Test integrasi C (221 test, satu proses) | 6,91 GB | 2,64 GB | 7,0 menit |
| Recheck stage-1, plan-only | 5,50 GB | 1,71 GB | 2,1 menit |
| Recheck stage-1, `--run-judge` | 5,85 GB | 2,32 GB | 3,4 menit |

## 6. Test

- **Offline:** 1.023 test ter-collect tanpa error collection:

  | Direktori | Test |
  |---|---|
  | `llm/tests` | 77 |
  | `llm/c_graphrag/tests` | 221 |
  | `llm/b_rag/tests` | 23 |
  | `llm/evaluation/tests` | 405 |
  | `backend/tests` | 297 |

  Semua ID test dari commit `6a3c956` masih ter-collect.
- **Test baru:**
  - `test_expansion_answered.py` (7, offline): urutan dan tie-break, `search_n`, exclusion membuat Question tidak lagi answerable, switch off sama dengan jalur lama, kandidat kurang dari top_n, kunci config dan nama file, default setting.
  - `test_expansion_integration.py` (5, integrasi): setiap Question expansion punya Answer yang tidak dikecualikan, deterministik, hanya tahap expansion yang berubah, kedua snapshot.
  - `test_question_index_single_load.py` (5): spy, salinan vektor, cek nprobe, dan pemuatan sekali per proses.
- **Integrasi** (`GRAPHRAG_INTEGRATION=1`, dijalankan sendirian): 219 lulus.
  - 2 test cek nprobe sempat gagal karena salah positif spy: file indeks mini di test memakai nama yang sama dengan indeks pertanyaan.
  - Nama filenya sudah diganti, dan test ulang dengan spy aktif lulus.

## 7. Konsekuensi untuk stage 2 (belum dikerjakan)

- Kandidat stage-2 menurut aturan seleksi sekarang adalah **α = 0,25 dan α = 0**. Pada α = 0, `final_score` hanya memakai similarity; bobot trust = 0.
- `STAGE2_ALPHAS` di `stage2_report.py` masih (0,5; 0,25) dan belum diubah. Mengikuti preseden Amendment 6, nilai ini diperbarui ke top_two baru sebelum run stage-2 mana pun, tetapi menunggu keputusan penulis.
- Run stage-2 dan run resmi harus memakai `EXPANSION_REQUIRE_ANSWER=True` (default). Kunci config-nya membedakan config_hash dari run lama.
