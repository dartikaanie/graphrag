# DRAF — Amandemen 7 (2026-10-17): paritas korpus dan teks query Kondisi B dengan Kondisi C

> **Status: DRAF, belum dikonfirmasi penulis dan belum dimasukkan ke `docs/DECISION_C_SCORING.md`.**
> Ditulis setelah stage 1 dan SEBELUM stage 2 maupun run apa pun pada test sample (n = 384). Test sample belum disentuh oleh perubahan ini. Satu-satunya run baru adalah run Kondisi B pada dev split (posisi 385–419, n = 35). Rincian teknis ada di `reports/b_corpus_parity.md`.

### Temuan
- Kondisi B membangun indeks FAISS dari **sampel acak 8.000 jawaban** (`INDEX_POOL`) dari parquet. Kondisi C menjangkau **seluruh 725.795 Answer node** di KG Neo4j (`graphrag`). Jadi B hanya melihat sekitar 1,1% jawaban yang dapat dijangkau C.
- Teks query B (`f"{Title}\n{Body}"`, body lengkap) berbeda dari C (`f"{Title} {str(Body)[:1000]}"`, sama dengan yang di-encode `11_densify_embedding_similarity.py`).
- Akibatnya, keunggulan C atas B dapat dijelaskan oleh ukuran korpus atau perbedaan input query, bukan oleh struktur graph dan trust-weighting yang hendak diuji. Ini ancaman terhadap validitas perbandingan, bukan temuan dari hasil hallucination.

### Keputusan
1. **Korpus B = seluruh Answer node KG yang sama dengan C** (725.795 jawaban; setiap jawaban punya tepat satu Question induk lewat `HAS_ANSWER`/`HAS_ACCEPTED_ANSWER`; 0 jawaban yatim, 0 dengan dua induk).
2. **Representasi teks jawaban identik dengan C**: satu vektor per jawaban, teks `Q: <judul pertanyaan induk>\nA: <body jawaban>` dipotong ke 400 token cl100k_base dengan fungsi pemotong yang sama dengan C (dipindah tanpa perubahan ke `llm/common/text.py`). Tidak ada multi-chunk.
3. **Teks query identik dengan C**: `build_query_text(title, body) = f"{title} {str(body)[:1000]}"`, satu fungsi bersama untuk B dan C. Vektor query B sama dengan vektor query C dalam toleransi 1e-6 (selisih maksimum terukur 1,8e-7 pada 35 pertanyaan dev).
4. **Neo4j hanya sumber ekspor offline.** `build_kg_answer_index.py --stage export` membaca Neo4j sekali (read-only) dan menulis `kg_answers.parquet`. Saat runtime, Kondisi B **tidak mengakses Neo4j sama sekali**: retrieval tetap FAISS murni (`IndexFlatIP`, exact search, embedding all-MiniLM-L6-v2 ter-normalisasi L2), teks diambil dari parquet. B tidak memakai graph, relasi, maupun trust.
5. **Leakage exclusion B menjadi per query**, sama seperti C: over-fetch, buang pertanyaan yang sedang dijawab beserta semua jawaban di thread-nya (dan accepted answer-nya), lalu potong ke `top_k`. Definisinya satu fungsi bersama untuk B, C, dan D (`llm/common/leakage.py`), dengan assert per query: hasil retrieval ∩ exclusion set = ∅.
6. **Perilaku Kondisi C tidak berubah.** C hanya dialihkan untuk memanggil fungsi bersama. Retrieval C pada 12 pertanyaan dev × 4 konfigurasi identik dengan snapshot sebelum refactor (id, skor, dan hash teks konteks). Bobot trust, nilai α, dan data Neo4j tidak diubah.
7. Korpus lama dipertahankan di belakang `--corpus legacy_sample`, hanya untuk mereproduksi pilot lama. Run korpus KG punya `config_hash` sendiri (`"corpus": "kg_answers_v1_tok400"`), sehingga tidak pernah tercampur dengan run lama.

### Yang tidak berubah
Sampling dan split, prompt (`PROMPT_VERSION` v3), model generator, `top_k = 5`, `TOKEN_CHUNK_LIMIT = 400`, model embedding, metrik evaluasi, serta seluruh perilaku Kondisi A, C, dan D.

### Dampak terhadap hasil pilot
Hasil pilot B lama (n = 10, **test items**, korpus 8.000 jawaban) dan run dev B lama (n = 50, korpus 8.000) dibuat dengan korpus dan teks query yang berbeda. Keduanya **tidak dapat dibandingkan langsung** dengan run B baru maupun dengan C, dan tetap dicatat sebagai hasil korpus `legacy_sample`. Perbandingan run dev baru vs lama pada 35 pertanyaan dev yang sama dilaporkan di `reports/b_corpus_parity.md` §7 sebagai informasi saja, bukan dasar seleksi. Semua run B resmi (n = 384) memakai korpus KG.

### Justifikasi
Ini koreksi validitas desain eksperimen (paritas korpus dan input antara pembanding), diputuskan berdasarkan struktur pipeline, bukan pada hasil hallucination. Perubahan dibuat sebelum test sample disentuh.
