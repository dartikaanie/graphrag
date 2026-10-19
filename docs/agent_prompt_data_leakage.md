# Tugas lanjutan: perluas exclusion leakage (setelah paritas korpus B selesai)

## Prasyarat
- Tugas paritas korpus B sudah selesai: build indeks, test indeks penuh, dev run n=35, laporan.
- Perubahannya sudah di-commit **lokal** di `fix/b-corpus-parity`, dengan `git add` per file (jangan ikutkan file yang bukan bagian tugas itu).
- Buat branch baru dari `fix/b-corpus-parity`: `fix/leakage-filter`. Jangan push, jangan merge.

Prompt ini menggantikan `docs/agent_prompt_data_leakage.md` untuk bagian Duplicate/Linked/near-dup. Jika ada yang bertentangan, ikuti prompt ini.

## Keadaan saat ini (dari investigasi kamu sendiri)
- `llm/common/leakage.py` `build_exclusion()` baru mencakup q, semua answer thread q, dan accepted answer q.
- Belum ada: Duplicate (dua arah), opsi Linked, opsi near-dup.
- Tidak ada filter berbasis isi accepted answer. Pertahankan seperti itu.
- C/D sudah memakai `$exclude_q` / `$exclude_a` di anchor, q2 (2-hop), dan answer. Nama parameter tidak perlu diganti.
- `reputation_ablation_check.py:255-261` masih punya salinan exclusion sendiri.
- B (`kg_corpus.py:179-199`) mengembalikan kurang dari k secara diam-diam bila kandidat habis.
- `semantic_expansion()` tidak memakai buffer 6×. Question hasil expansion tidak dicek apakah masih punya answer yang tidak dikecualikan.

## Keputusan yang sudah diambil
- **Buffer 6× di `semantic_expansion()`:** sudah diinvestigasi. Fix itu tidak pernah di-commit, jadi seluruh pilot dan stage-1 α dijalankan dengan expansion versi sekarang (`top_n + |exclude| + 5`, tanpa verifikasi answer). **Jangan ubah logika `semantic_expansion()`.** Cukup ukur slot expansion yang kosong (lihat Langkah 4).
- **Duplicate:** 0 pertanyaan dev stage-1 punya Duplicate, jadi mode default tidak mengubah retrieval dev. **Stage-1 α tidak diulang** selama Linked tetap `False`. Verifikasi klaim ini di Langkah 4c.
- **Linked:** default `False`. Linked dianggap pengetahuan komunitas yang sah, bukan ground-truth q. Mode `+Linked` hanya untuk analisis sensitivitas.

## Langkah 1 — perluas `build_exclusion()` (satu sumber kebenaran)
Exclusion set untuk q =
- {q} ∪ semua answer thread q (sudah ada);
- ∪ semua Question yang terhubung ke q lewat `IS_RELATED_TO` tipe **Duplicate**, dua arah, beserta semua answer-nya;
- ∪ (jika `LEAKAGE_EXCLUDE_LINKED=True`, default `False`) Question tipe Linked, dua arah, beserta answer-nya;
- ∪ (jika `LEAKAGE_NEAR_DUP_COSINE` bukan `None`, default `0.95`) Question yang cosine embedding Title+Body-nya dengan q ≥ threshold, beserta answer-nya.
  - Pakai indeks embedding pertanyaan yang sudah dipakai `anchor_via_vector_search`.
  - Jangan bangun indeks baru.
  - `SentenceTransformer` harus `device="cpu"`.

Aturan tambahan:
- Exclusion hanya berdasarkan **identitas/kemiripan pertanyaan**. Jangan memfilter berdasarkan isi accepted answer.
- Ambil tetangga `IS_RELATED_TO` lewat query Cypher berparameter. Pola di `stage1_sweep.py:367-369` boleh dipakai ulang, tapi jangan dicampur dengan logika donor hard-negative.
- Tambahkan setting `LEAKAGE_FILTER` (default `True`), `LEAKAGE_EXCLUDE_LINKED`, dan `LEAKAGE_NEAR_DUP_COSINE` di Pydantic Settings. Setting ini harus ikut config hash supaya run lama dan baru tidak tercampur.
- Precompute exclusion set untuk seluruh pool sampel (`MAX_PLANNED_N_SAMPLE = 384*4`, pakai sampler yang ada), lalu simpan ke `results/leakage/exclusion_sets.parquet` per mode. Kolomnya:
  - `question_id`, `mode`;
  - `excluded_q_ids`, `excluded_a_ids`;
  - `n_dup`, `n_linked`, `n_neardup`.

## Langkah 2 — integrasi
- B, C, D, dan `reputation_ablation_check.py` semuanya memakai `build_exclusion()`. Hapus salinan exclusion di `reputation_ablation_check.py`.
- Karena duplikat q sekarang masuk `$exclude_q`, pastikan duplikat tidak bisa menjadi anchor, q2, atau Question hasil expansion.
- Buffer over-fetch (B: `k + |A| + 5`; anchor/expansion: `top_n + |exclude| + 5`) otomatis membesar mengikuti exclusion set. Pastikan tetap berlaku untuk set yang lebih besar.
- **B:** bila hasil < k, tulis warning dan catat di record.
- **Semua kondisi:** log per query berisi `mode`, `n_excluded_q`, `n_excluded_a`, `n_hits_filtered`, dan `final_k`.
- `assert_no_leakage` memakai exclusion set yang sudah diperluas.

## Langkah 3 — test (pytest)
- Untuk ≥ 20 pertanyaan dev, irisan ID hasil retrieval dengan exclusion set harus ∅ di B, C, D. Uji di setiap mode.
- Ada kasus uji dengan edge Duplicate nyata, yaitu pertanyaan dev yang punya duplikat di KG. Duplikat itu tidak boleh muncul sebagai anchor, q2, expansion, maupun answer.
- Exclusion set deterministik antar run (hash parquet identik).
- **Mode default dengan Duplicate=0 ekuivalen dengan perilaku sekarang:** untuk pertanyaan tanpa duplikat, hasil retrieval C harus identik dengan snapshot yang ada.
- Semua test lama tetap lulus.

## Langkah 4 — audit (tanpa memanggil LLM)
Data pilot lama sudah diarsipkan dan tidak dipakai lagi. Karena itu, audit dilakukan **retrieval-only** pada dev set (posisi 385–434) dengan dua kondisi: `LEAKAGE_FILTER=False` dan `True`, di setiap mode.

**a. Self-retrieval.** Per kondisi B, C, D dan per mode, hitung:
- *ID-level self-retrieval rate*: persentase item yang konteksnya memuat q, answer q, atau Question/answer di exclusion set.
- *Text-level (metrik saja, bukan filter)*: substring identik ≥ 50 karakter dari accepted answer di konteks, atau cosine ≥ 0.95 antara accepted answer dan chunk konteks.

**b. Pembanding.** Sebagai pembanding, tugas paritas B mencatat bahwa tanpa exclusion, 31/35 pertanyaan dev mengambil thread-nya sendiri. Laporkan angka yang setara untuk C dan D.

**c. Verifikasi dampak ke stage-1 α.** Pada dev set stage-1 (n=50), dalam mode default (Duplicate + near-dup 0.95, Linked off):
- Berapa pertanyaan yang exclusion set-nya berubah?
- Apakah retrieval C identik dengan sebelumnya?

Near-dup belum pernah diukur dan bisa lebih berpengaruh daripada Duplicate. Kalau retrieval identik, tulis bahwa stage-1 tidak perlu diulang. Kalau ada yang berubah, laporkan jumlah dan contohnya, lalu BERHENTI sebelum dev run dan tunggu keputusan saya. Jangan jalankan ulang pemilihan α.

**d. Slot expansion kosong.** Per kondisi C dan per mode, hitung berapa Question hasil `semantic_expansion()` yang tidak menyumbang answer apa pun karena semua answer-nya dikecualikan. Laporkan juga apakah angkanya naik dibanding exclusion lama.

**e. Linked (sensitivitas).** Untuk pertanyaan yang punya tetangga Linked ber-answer, apakah konteks dari tetangga itu lolos text-level check (substring ≥ 50 karakter / cosine ≥ 0.95 terhadap accepted answer q)? Hasil ini menentukan apakah Linked perlu dikecualikan.

## Langkah 5 — dev run n ≤ 35
Jalankan B, C, D dengan filter aktif dalam mode default. Bandingkan dengan dev run tugas paritas B (untuk B) dan dengan exclusion lama (untuk C/D) pada:
- self-retrieval rate (target 0%);
- text-level leakage;
- cosine similarity;
- jumlah item `final_k < k` dan `final_k = 0`;
- latency (NF3: C ≤ 15 s p95).

## Batasan
- **Jangan jalankan n=384.**
- Jangan sentuh `.env` atau API key, dan jangan tampilkan isinya.
- Jangan ubah bobot trustScore, α, atau edge KG.
- Jangan ubah pipeline judge.
- Jangan ubah logika `semantic_expansion()`, kecuali yang perlu supaya exclusion set yang diperluas ikut dipakai.
- Proses berat dijalankan satu per satu, dalam batch. DuckDB memakai `memory_limit='2GB'`, `threads=2`. Baca `.jsonl` hanya dari path eksplisit (abaikan `results_old/`).

## Deliverable
1. Kode dan test sesuai langkah di atas.
2. `reports/leakage_audit.md` berisi:
   - jumlah yang dikecualikan per mode;
   - audit filter off vs on (ID-level dan text-level, per kondisi dan per mode);
   - hasil dev run;
   - dampak ke latency;
   - dampak ke stage-1 α;
   - slot expansion kosong dan hasil sensitivitas Linked.
3. Draf paragraf Bab III (Bahasa Indonesia), mencakup:
   - prosedur exclusion per-query dan metrik self-retrieval rate;
   - Wu dkk. sebagai **analogi prinsip** (ground-truth di prompt menggelembungkan hasil), bukan klaim bahwa mereka meneliti RAG;
   - keterbatasan: LLM kemungkinan sudah melihat data Stack Overflow saat pretraining, dan ini tidak bisa dihilangkan oleh filter retrieval.
4. Ringkasan akhir: apa yang berubah, angka self-retrieval sebelum/sesudah, dan keputusan terbuka (Linked, threshold near-dup, perlu tidaknya re-run stage-1 α).