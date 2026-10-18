# Amandemen 8 (2026-10-18): exclusion leakage per query untuk B, C, dan D

> Versi bahasa Indonesia dari Amendment 8. Versi bahasa Inggris untuk `docs/DECISION_C_SCORING.md` ada di `reports/leakage_amendment8_draft_en.md`.

**Status: dikonfirmasi penulis.** Ditulis setelah stage 1 dan sebelum stage 2 maupun evaluasi resmi pada test sample (n = 384). Keputusan diambil sebelum stage 2 dan sebelum run resmi n = 384 apa pun. Tidak ada retrieval maupun generation yang dijalankan pada test sample; untuk test sample hanya dihitung jumlah agregat pertanyaan terkait. Rincian teknis ada di `reports/leakage_audit.md`.

### Temuan
- Setiap korpus retrieval (B: seluruh jawaban KG; C dan D: KG) juga memuat pertanyaan evaluasi beserta jawabannya, karena KG dibangun dari seluruh populasi Stack Overflow.
- Tanpa exclusion apa pun, retrieval hampir selalu mengambil thread pertanyaan itu sendiri. Pada dev set (posisi 385–434, n = 50, retrieval saja, tanpa LLM), proporsi pertanyaan yang konteksnya memuat id dari thread-nya sendiri adalah **B 84% (42/50), C 100% (50/50), D 100% (50/50)**. C dan D mencapai 100% karena keduanya ber-anchor pada embedding pertanyaan, dan pertanyaan itu sendiri menjadi anchor teratas bila tidak dikecualikan.
- Di pool sampling (1.471 pertanyaan), 13 pertanyaan (0,9%) punya Duplicate `IS_RELATED_TO`, 3 (0,2%) punya pertanyaan near-duplicate (cosine ≥ 0,95), dan 209 (14,2%) punya pertanyaan Linked. Dev set (n = 50): 0, 0, dan 6. Test sample (hanya jumlah agregat): 2 Duplicate, 1 near-duplicate.

### Keputusan
1. **Exclusion set per pertanyaan evaluasi q** (satu fungsi bersama untuk B, C, dan D, `build_exclusion()` di `llm/common/leakage.py`): q dan seluruh jawaban di thread q (termasuk accepted answer); setiap Question yang terhubung ke q lewat `IS_RELATED_TO {link_type: 'Duplicate'}`, dua arah, beserta semua jawabannya; setiap Question yang embedding Title+Body-nya punya cosine ≥ 0,95 dengan q (embedding dan indeks pertanyaan milik C), beserta semua jawabannya. Exclusion hanya berdasarkan identitas dan kemiripan pertanyaan, tidak pernah berdasarkan isi accepted answer q.
2. **Linked tidak dikecualikan** (`LEAKAGE_EXCLUDE_LINKED=False`). Pertanyaan Linked adalah pengetahuan komunitas yang secara sah berkaitan dengan q, bukan ground truth q. Audit mendukung hal ini: pada dev set tidak ada item konteks yang berasal dari tetangga Linked, di kondisi mana pun; untuk 98 pertanyaan pool non-test yang punya tetangga Linked ber-answer, tidak ada jawaban tetangga yang mencapai cosine ≥ 0,95 dengan accepted answer q (maksimum 0,9468). Mengecualikan Linked hanya dipertahankan sebagai mode sensitivitas.
3. **Ambang near-duplicate 0,95.** Pada 0,95 ambang ini menangkap 3 pertanyaan pool (7 pertanyaan terkait). Kasus Linked 0,9468 menunjukkan bahwa ambang yang lebih rendah mulai menangkap jawaban setopik yang sah.
4. **Setting** `LEAKAGE_FILTER` (default True), `LEAKAGE_EXCLUDE_LINKED` (default False), dan `LEAKAGE_NEAR_DUP_COSINE` (default 0,95) menjadi bagian dari config dan config_hash setiap kondisi sebagai `"leakage": "dup+neardup0.95"`. Kunci ini tidak ada untuk exclusion lama yang hanya thread, sehingga config_hash semua run sebelumnya tidak berubah. `LEAKAGE_FILTER=False` hanya untuk audit.
5. **Implementasi.** Bagian pertanyaan terkait dihitung sekali secara offline untuk seluruh pool sampling ke `results/leakage/exclusion_sets.parquet` (byte-identik bila dibangun ulang); B membacanya dan tidak pernah mengakses Neo4j saat runtime. Setiap kondisi melakukan assert per query bahwa hasil retrieval ∩ exclusion set = ∅, dan mencatat mode, jumlah pertanyaan dan jawaban yang dikecualikan, jumlah hit yang tersaring, serta jumlah akhir item konteks.

### Dampak
- Dengan exclusion, ID-level self-retrieval menjadi **0% untuk B, C, dan D** di setiap mode, dan tidak ada item konteks dengan cosine ≥ 0,95 terhadap accepted answer.
- **Tidak ada dampak ke stage 1 (pemilihan α).** Pada mode default, 0 dari 50 exclusion set dev stage-1 berubah dibanding exclusion lama (hanya thread) yang dipakai di stage 1, dan retrieval C identik untuk 50/50 pertanyaan pada α = 0,5 dan pada α = 0,25 (id, urutan, skor, dan teks). Karena itu stage 1 tidak diulang.

### Metrik
Metrik leakage resmi adalah ID-level self-retrieval rate dan kemiripan kosinus (≥ 0,95) antara setiap item konteks dan accepted answer q. Kosinus ini hanya alat ukur, tidak pernah dipakai sebagai filter. Metrik ketiga, substring identik ≥ 50 karakter antara konteks dan accepted answer, dicoba lalu ditinggalkan: pada HTML mentah, setiap kecocokan yang ditemukan saat filter aktif (26 dari 26 yang diperiksa) berupa boilerplate (atribut XML Android, URL dokumentasi, baris Gradle, impor standar), bukan isi jawaban.

### Justifikasi
Ini koreksi validitas evaluasi (ground truth tidak boleh sampai ke prompt lewat retrieval), diputuskan berdasarkan struktur korpus dan hasil audit, bukan pada hasil hallucination. Keputusan diambil sebelum stage 2 dan sebelum evaluasi resmi pada test sample.
