# Tugas: tutup hasil leakage (Amendment 8) + perbaiki slot expansion kosong (Amendment 9)

Kerjakan **Bagian 1** sampai selesai dan di-commit. Setelah itu baru **Bagian 2**.
Ada beberapa titik BERHENTI. Di setiap titik itu, tunggu persetujuan saya.

Batasan umum:
- Jangan push dan jangan merge.
- Jangan jalankan n=384, jangan sentuh test sample.
- Jangan ubah bobot trustScore, edge KG, atau judge.
- Jangan sentuh `.env`.
- Proses berat dijalankan satu per satu. DuckDB `memory_limit='2GB'`, `threads=2`.
- Embedding query memakai `device="cpu"`.

---

## Bagian 1 — tutup hasil leakage (branch `fix/leakage-filter`)

### Keputusan saya
- Linked tetap mati (`LEAKAGE_EXCLUDE_LINKED=False`).
- Near-dup tetap 0,95.
- Metrik substring **tidak dipakai** sebagai metrik resmi. Di laporan, catat sebagai metrik yang dicoba lalu ditinggalkan, beserta alasannya (semua kecocokan berupa boilerplate). Metrik resmi: ID-level self-retrieval rate dan cosine ≥ 0,95 (konteks vs accepted answer, hanya alat ukur).
- Slot expansion kosong diperbaiki di Bagian 2, bukan di sini.

### Tugas
1. **Pool 1.471 vs 1.536.** Jelaskan kenapa pool exclusion berisi 1.471 pertanyaan, bukan `384*4 = 1.536`. Mana yang tidak ikut, dan kenapa? Jangan ubah sampling. Kalau penyebabnya bisa memengaruhi sampel uji (posisi 1–384), BERHENTI dan laporkan.
2. **Rujukan Wu dkk.** Lengkapi rujukan di draf Bab III (laporan §11) dan tambahkan entri biblatex:
   - Penulis: Wu, Xingyu; Yu, Kui; Wu, Jibin; Tan, Kay Chen
   - Judul: "Reassessing the role of LLMs in causal discovery: Critical analysis, benchmark evaluation, and exploratory framework"
   - Pattern Recognition, vol. 182, art. 114825, 2027 (online 31 Agustus 2026)
   - DOI 10.1016/j.patcog.2026.114825

   Sebut sebagai **analogi prinsip** (ground-truth di prompt menggelembungkan hasil), bukan klaim bahwa mereka meneliti RAG.
3. **Draf Amendment 8**, bahasa Inggris, gaya Amendment 1–7. Isinya:
   - temuan: self-retrieval tanpa filter (B 84%, C 100%, D 100%);
   - keputusan: definisi exclusion (thread q + Duplicate dua arah + near-dup pertanyaan ≥ 0,95; Linked off) dan alasan Linked off;
   - bukti tidak ada dampak ke stage-1 α (0/50 exclusion set berubah, retrieval identik 50/50 di α = 0,5 dan 0,25);
   - catatan metrik substring ditinggalkan;
   - catatan bahwa keputusan diambil sebelum stage 2 dan sebelum evaluasi resmi n=384.

   Simpan juga versi bahasa Indonesianya di `reports/`.

**BERHENTI:** tunjukkan jawaban nomor 1 dan diff Amendment 8 (beserta perubahan §11) sebelum menyisipkannya ke `docs/DECISION_C_SCORING.md`. Setelah saya setujui, commit lokal dengan `git add` hanya file yang relevan.

---

## Bagian 2 — perbaiki slot expansion kosong (branch baru `fix/expansion-answered` dari `fix/leakage-filter`)

### Masalah
Sekitar 64% Question hasil `semantic_expansion()` di C tidak menyumbang jawaban, karena Question itu memang tidak punya Answer di KG. Ini bukan akibat exclusion. Tahap 3 jadi mengisi slot dengan kandidat yang mustahil menyumbang konteks.

### Prinsip perbaikan
- Perbaikannya bersifat **struktural**: Question tanpa jawaban yang tersisa (setelah exclusion) tidak boleh dipilih sebagai hasil expansion. Keputusan ini tidak didasarkan pada hasil hallucination.
- Ubah **hanya** tahap Semantic Expansion. Entity Anchoring, Graph Traversal, skor trust, dan α tidak diubah.
- Kondisi D tidak diubah, karena high-level D sudah memakai `EXISTS`.

### Langkah 2.1 — rancangan (BERHENTI)
Sebelum mengubah kode, tulis rancangan singkat:
- **Over-fetch:** ambil kandidat Question via FAISS sebanyak `6 * top_n + |exclude_q|` per seed. Kalau menurutmu ada angka lain yang lebih tepat, usulkan beserta alasannya.
- **Verifikasi dalam satu round-trip Cypher berparameter** untuk seluruh kandidat: pertahankan hanya Question yang punya minimal satu Answer dengan `NOT a.id IN $exclude_a`. Polanya seperti `EXISTS` di `d_lightrag.py:461-464`.
- **Pemilihan:** pilih `top_n` teratas dari kandidat terverifikasi, dengan urutan similarity tetap dan tie-break deterministik (`ORDER BY ... , id ASC`).
- **Kalau kandidat terverifikasi kurang dari `top_n`:** pakai yang ada, tulis log, dan catat di record (`n_expansion_requested`, `n_expansion_answerable`).
- Jelaskan apakah fungsi lain yang memanggil `semantic_expansion()` (stage1_sweep, recheck scripts, `reputation_ablation_check.py`, dashboard) ikut berubah otomatis.
- **Cek stage-1:** apakah metrik pemilihan α di stage-1 butuh panggilan LLM/judge, atau cukup retrieval-only? Kalau butuh LLM, sertakan perkiraan jumlah panggilan dan biayanya.
- **Flag:** tambahkan `EXPANSION_REQUIRE_ANSWER` (default `True`) di Pydantic Settings, ikut config hash. Nilai `False` mereproduksi perilaku lama, hanya untuk audit.

Tunjukkan rancangan ini lalu BERHENTI.

### Langkah 2.2 — draf Amendment 9 (sebelum ada hasil apa pun)
Tulis draf Amendment 9 (bahasa Inggris, plus versi Indonesia di `reports/`) **sebelum** menjalankan apa pun. Isinya:
- temuan 64% dan penyebabnya;
- perubahan yang dibuat;
- pernyataan bahwa stage-1 α **akan dicek ulang dengan protokol yang sama persis** (dev set, kandidat α = 0,5 dan 0,25, metrik, dan aturan pemilihan tidak berubah), dan bahwa **hasilnya diterima apa pun itu**, termasuk bila α terpilih berubah;
- pernyataan bahwa keputusan diambil sebelum stage 2 dan sebelum evaluasi resmi n=384.

Bagian "Outcome" dibiarkan kosong dulu.

### Langkah 2.3 — implementasi dan test
- Implementasikan sesuai rancangan yang saya setujui.
- **Test baru:**
  - setiap Question hasil expansion punya ≥ 1 answer yang tidak dikecualikan;
  - hasilnya deterministik antar run;
  - mode `EXPANSION_REQUIRE_ANSWER=False` identik dengan snapshot lama;
  - irisan dengan exclusion set tetap ∅ (assert leakage tetap aktif).
- **Snapshot C:** dengan flag `True`, item konteks yang `source_stage`-nya anchor/traversal **harus identik** dengan snapshot lama. Hanya item `semantic_expansion` yang boleh berubah. Simpan snapshot baru sebagai file terpisah. Jangan menimpa snapshot lama.
- Semua test lama tetap lulus (offline dan `GRAPHRAG_INTEGRATION=1`).

### Langkah 2.4 — ukur dan cek ulang stage-1 (retrieval-only)
Pada dev set stage-1 (n=50), bandingkan flag `False` vs `True`:
- persentase slot expansion kosong (target jauh di bawah 64%);
- jumlah item konteks dari expansion per pertanyaan, dan total `final_k`;
- jumlah pertanyaan yang retrieval C-nya berubah, di α = 0,5 dan 0,25;
- latency C p95 (NF3 ≤ 15 detik).

Lalu jalankan ulang pemilihan stage-1 α dengan protokol yang sama persis:
- Kalau metrik stage-1 cukup retrieval-only, langsung jalankan.
- Kalau butuh panggilan LLM, **BERHENTI** dan tunggu persetujuan biaya dari saya.

Laporkan α terpilih lama vs baru beserta angkanya. **Jangan memilih α sendiri di luar aturan pemilihan yang ada.**

### Deliverable Bagian 2
1. Kode, setting, dan test.
2. `reports/expansion_answered.md` berisi:
   - rancangan;
   - angka sebelum/sesudah;
   - hasil cek ulang stage-1;
   - dampak latency.
3. Draf Amendment 9 dengan bagian "Outcome" terisi (hasil stage-1 apa adanya).

**BERHENTI:** tunjukkan diff Amendment 9 sebelum menyisipkannya ke `docs/DECISION_C_SCORING.md`. Setelah saya setujui, commit lokal dengan `git add` hanya file yang relevan.