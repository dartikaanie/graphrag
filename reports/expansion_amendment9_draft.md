# Amandemen 9 (2026-10-18): semantic expansion hanya memilih Question yang punya jawaban

> Versi bahasa Indonesia dari Amendment 9. Versi bahasa Inggris untuk `docs/DECISION_C_SCORING.md` ada di `reports/expansion_amendment9_draft_en.md`.

**Status: dikonfirmasi penulis. Ditulis sebelum perubahan ini dijalankan sama sekali** (belum ada retrieval, penilaian judge, maupun cek ulang stage-1 dengan perubahan ini). Ditulis setelah stage 1 dan sebelum stage 2 maupun evaluasi resmi pada test sample (n = 384). Keputusan diambil sebelum stage 2 dan sebelum run resmi n = 384 apa pun. Rincian teknis ada di `reports/expansion_answered.md`.

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
[diisi setelah cek ulang: proporsi slot expansion kosong sebelum/sesudah, item konteks dari expansion, jumlah pertanyaan yang retrieval-nya berubah pada α = 0,5 dan 0,25, latency C p95, tabel stage-1 dan α terpilih (lama vs baru), tanggal]

### Justifikasi
Ini koreksi struktural (slot expansion harus bisa menyumbang konteks), diputuskan berdasarkan struktur KG, bukan pada hasil hallucination. Keputusan diambil sebelum stage 2 dan sebelum evaluasi resmi pada test sample.
