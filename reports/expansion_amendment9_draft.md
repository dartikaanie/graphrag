# Amandemen 9 (2026-10-18): semantic expansion hanya memilih Question yang punya jawaban

> Versi bahasa Indonesia dari Amendment 9. Versi bahasa Inggris untuk `docs/DECISION_C_SCORING.md` ada di `reports/expansion_amendment9_draft_en.md`.

**Status: dikonfirmasi penulis. Ditulis sebelum perubahan ini dijalankan sama sekali** (belum ada retrieval, penilaian judge, maupun cek ulang stage-1 dengan perubahan ini). Hanya bagian Hasil yang diisi sesudahnya, dari cek ulang. Ditulis setelah stage 1 dan sebelum stage 2 maupun evaluasi resmi pada test sample (n = 384). Keputusan diambil sebelum stage 2 dan sebelum run resmi n = 384 apa pun. Rincian teknis ada di `reports/expansion_answered.md`.

### Temuan
- Di Kondisi C, tahap 3 (Semantic Expansion) memilih, untuk setiap seed, `n_semantic_expansion` (= 3) Question yang paling mirip dengan seed, lalu mengambil jawabannya. Banyak dari Question itu tidak punya Answer sama sekali di KG: KG memuat 2.685.814 Question tetapi hanya 725.795 Answer (602.900 Question dengan minimal satu Answer).
- Pada audit leakage (Amendment 8, dev set n = 50, α = 0,5), **64% Question expansion (277 dari 432) tidak menyumbang jawaban**, dan tidak satu pun disebabkan oleh exclusion leakage: semuanya memang tidak punya Answer di KG. Tahap expansion jadi mengisi sebagian besar slotnya dengan kandidat yang mustahil menyumbang konteks.
- Di antara 18 kandidat FAISS pertama setiap seed, 32,7% Question punya Answer; hanya 11 dari 144 seed dev yang mendapat 3 Question ber-answer dari top-3 yang sekarang.

### Perubahan
1. **Hanya Semantic Expansion yang berubah.** Entity Anchoring, Graph Traversal, skor trust, α, dan `fuse_and_rank()` tidak berubah. Kondisi D tidak berubah (tahap high-level-nya sudah mensyaratkan jawaban lewat `EXISTS`).
2. **Over-fetch:** per seed, `search_n = 20 · top_n + |exclude_ids|` kandidat FAISS (60 + id yang dikecualikan, untuk top_n = 3). Faktor 20 dipilih dari distribusi keberadaan Answer di KG pada dev-50 — proporsi seed yang mendapat top_n Question ber-answer: 6× → 83%, 12× → 97%, 20× → 100% (peringkat terdalam yang dibutuhkan: 51) — bukan dari hasil hallucination atau relevansi.
3. **Verifikasi:** kandidat seluruh seed dicek dalam satu round-trip Cypher berparameter; sebuah Question hanya dipertahankan bila punya minimal satu Answer yang tidak termasuk exclusion leakage (`EXISTS { (q)-[:HAS_ANSWER|HAS_ACCEPTED_ANSWER]->(a:Answer) WHERE NOT a.id IN $exclude_a }`, mengikuti filter edge type).
4. **Pemilihan:** per seed, berurutan, `top_n` kandidat terverifikasi pertama menurut similarity (tie-break: id pertanyaan menaik) yang belum diambil seed sebelumnya. Bila kandidat terverifikasi kurang dari `top_n`, yang ada tetap dipakai, dan record mencatat `n_expansion_requested` dan `n_expansion_answerable`.
5. **Setting** `EXPANSION_REQUIRE_ANSWER` (default True) menjadi bagian dari config dan config_hash C (`"expansion_require_answer": true, "expansion_overfetch": 20`); kunci ini tidak ada bila False, sehingga config_hash semua run sebelumnya tidak berubah. False mereproduksi expansion lama persis dan hanya untuk audit. CLI, dashboard, dan `stage1_sweep.py` mengikuti setting ini; `reputation_ablation_check.py` dan `investigate_zero_context.py` sengaja tetap memakai expansion lama demi reproduksi Amendment 5.

### Cek ulang stage-1 (protokol ditetapkan sebelum ada hasil)
Karena perubahan ini mengubah retrieval C, stage 1 dicek ulang dengan **protokol yang sama persis**: dev set yang sama (posisi 385–434, n = 50), grid α yang sama {0; 0,25; 0,5; 0,75; 1}, judge context-relevance primary yang sama (ctxrel-v2, temperature 0), metrik yang sama, dan aturan pemilihan yang sama (`select_stage1_candidates`). Label judge dipakai ulang hanya bila question_id, answer_id, text_hash, model judge, dan versi prompt semuanya identik; item lain dinilai ulang. **Hasilnya diterima apa pun itu, termasuk bila α terpilih berubah.**

### Hasil
Cek ulang dijalankan pada 2026-10-18 (laporan `llm/c_graphrag/results/expansion_check/stage1_expansion_report_20261018T042953Z.md`).
- **Reproduksi:** dengan switch off, top-5 identik dengan hasil stage-1 Amendment 6 (50 × 5), metrik yang dihitung ulang sama dengan yang tercatat, dan top_two tetap [0,5; 0,25].
- **Expansion:** slot expansion kosong turun dari 64,1% menjadi 0% (432 Question expansion pada 50 pertanyaan dev, di semua α). Item konteks dari expansion di top-5 kira-kira dua kali lipat (α = 0,5: 36 → 82; α = 0,25: 37 → 88). Pertanyaan dengan konteks kurang dari 5 item turun dari 5 menjadi 2. Anchor dan kandidat traversal tidak berubah di semua pertanyaan.
- **Pertanyaan yang top-5-nya berubah:** 38 / 50 pada α = 0,5 dan 34 / 50 pada α = 0,25.
- **Latency:** p95 retrieval-only C berubah sekitar +0,01–0,02 detik, misalnya α = 0,5 dari 0,091 menjadi 0,106 detik.
- **Judge:** 405 label cache dipakai ulang sesuai aturan identitas, dan 119 item baru dinilai (ctxrel-v2, primary, `meta-llama/Llama-3.3-70B-Instruct`, temperature 0; 0 gagal). Label baru ada di `results/expansion_check/ctxrel_v2_expansion_primary.jsonl`. Biayanya $0,0137.

| α | % q ≥1 RELEVANT (lama → baru) | skor rata-rata (lama → baru) | % accepted (lama → baru) |
|---|---|---|---|
| 0 | 48,0 → 50,0 | 0,3263 → 0,3525 | 37,29 → 38,52 |
| 0,25 | 48,0 → 48,0 | 0,3305 → 0,3504 | 41,95 → 47,13 |
| 0,5 | 48,0 → 48,0 | 0,3157 → 0,3299 | 53,39 → 61,89 |
| 0,75 | 46,0 → 44,0 | 0,2818 → 0,2848 | 66,95 → 76,23 |
| 1 | 34,0 → 28,0 | 0,2225 → 0,2029 | 72,88 → 85,66 |

- **Peringkat** dengan aturan yang sama: 0,25; 0,0; 0,5; 0,75; 1,0.
  - α = 0, 0,25, dan 0,5 seri pada kriteria utama (selisih ≤ 2 pp).
  - Skor rata-rata α = 0,5 lebih dari 0,02 di bawah α = 0.
  - α = 0,25 dan α = 0 seri pada kedua kriteria, sehingga α yang lebih besar ditempatkan pertama.
- **top_two = [0,25; 0,0]** (sebelumnya [0,5; 0,25]). keep_current = False.
- Sesuai pra-registrasi, hasil ini diterima. **Kandidat stage-2 resmi: α = 0,25 dan α = 0.** Pada α = 0, `final_score` hanya memakai similarity.

### Justifikasi
Ini koreksi struktural (slot expansion harus bisa menyumbang konteks), diputuskan berdasarkan struktur KG, bukan pada hasil hallucination. Keputusan diambil sebelum stage 2 dan sebelum evaluasi resmi pada test sample.
