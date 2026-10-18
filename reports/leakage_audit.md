# Audit kebocoran data (leakage) retrieval — perluasan exclusion per query

Tanggal: 2026-10-18 · Branch: `fix/leakage-filter` (dari `fix/b-corpus-parity`) · Prompt: `docs/agent_prompt_data_leakage.md`

## 1. Ringkasan

- **Exclusion per query diperluas dan dipakai seragam oleh B, C, dan D** lewat satu sumber kebenaran: `build_exclusion()` di `llm/common/leakage.py`.
  - Mode default `dup+neardup0.95` mengecualikan q, seluruh thread q, pertanyaan **Duplicate** (`IS_RELATED_TO`, dua arah) beserta jawabannya, dan pertanyaan **near-duplicate** (cosine embedding Title+Body ≥ 0,95) beserta jawabannya.
  - Linked tidak dikecualikan secara default (`LEAKAGE_EXCLUDE_LINKED=False`).
- **Tanpa filter, retrieval hampir selalu mengambil thread pertanyaan itu sendiri.** Pada 50 pertanyaan dev: B 84% (42/50; 31/35 pada 35 pertama, sama dengan angka tugas paritas), C 100%, D 100%.
- **Dengan filter aktif (semua mode), self-retrieval tingkat ID = 0%** di B, C, dan D. Kecocokan cosine ≥ 0,95 antara konteks dan accepted answer juga 0.
- **Mode default tidak mengubah apa pun pada dev set stage-1.**
  - 0 dari 50 exclusion set berubah.
  - Retrieval C identik 50/50 pada α = 0,5 dan α = 0,25.
  - Pada dev run, retrieval B, C, dan D identik 35/35 dengan exclusion lama.
  - Karena itu **stage-1 α tidak perlu diulang.**
- Dampak di pool sampel (1.471 pertanyaan): Duplicate 13 (0,9%), near-dup ≥ 0,95 sebanyak 3 (0,2%), Linked 209 (14,2%). Dari sampel uji 384, hanya angka agregat yang dihitung: 2 punya Duplicate dan 1 punya near-dup.
- Latency tidak terdampak: retrieval-only C p95 ≤ 2,5 detik, jauh di bawah NF3 15 detik.

## 2. Yang berubah

| File | Perubahan |
|---|---|
| `llm/common/leakage.py` | `build_exclusion(q, thread, accepted, related_q, related_a)`; `LeakageSettings` (Pydantic Settings: `LEAKAGE_FILTER` = True, `LEAKAGE_EXCLUDE_LINKED` = False, `LEAKAGE_NEAR_DUP_COSINE` = 0,95); string mode; `ExclusionTable` (membaca parquet); `LeakageContext` (`exclusion_for`, `query_log`); flag CLI bersama (`--leakage-filter`, `--leakage-exclude-linked`, `--leakage-near-dup-cosine`, `--leakage-mode`). Tetap tanpa neo4j. |
| `llm/leakage/build_exclusion_sets.py` (baru) | Precompute offline untuk seluruh pool (`MAX_PLANNED_N_SAMPLE*4` = 1.536, setelah filter token tersisa 1.471) menjadi `results/leakage/exclusion_sets.parquet` (kolom sesuai spesifikasi), per mode: `thread`, `dup`, `dup+neardup0.95`, `dup+linked+neardup0.95`. Tetangga diambil lewat Cypher berparameter (`IS_RELATED_TO`, dua arah, `link_type`). Near-dup memakai indeks IVF pertanyaan milik C (`anchor_via_vector_search`), tanpa indeks baru, nprobe 32, top-100, lalu skor dihitung ulang secara eksak dari memmap. Fallback encode memakai CPU. |
| C, D | `process_sample(leakage=…)`; exclusion lewat `LeakageContext`; `assert_no_leakage` per query; log dan record `leakage` = {mode, n_excluded_q, n_excluded_a, n_hits_filtered, final_k}; `anchor_via_vector_search`/`semantic_expansion` mendapat parameter penghitung `stats` (logika tidak berubah); C mencatat `expansion_qids` dan `expansion_qids_empty`. D: tambahan `--sample-split` dan `--retrieval-only`. |
| B | Exclusion dari tabel parquet (tanpa Neo4j); `search()` menghitung `n_hits_filtered`; bila `final_k < k`, B menulis warning dan mencatat `leakage.final_k_lt_k` di record. |
| `config` | Kunci `"leakage": <mode>` masuk ke `config_hash` B, C, dan D. Kunci ini **tidak ada** untuk mode `thread`, sehingga hash semua run lama tidak berubah; run baru tidak pernah tercampur dengan run lama. |
| Dashboard | `engine_service` memakai mode dari `LeakageSettings` (atau `params["leakage_mode"]`) dan tabel yang di-cache; `history_service` menampilkan `leakage_mode`. |
| `reputation_ablation_check.py` | Salinan exclusion dihapus dan diganti `build_exclusion()` (mode thread, sama seperti saat Amendment 5) serta `build_query_text()`. |
| `llm/leakage/audit.py` (baru) | Audit retrieval-only (`retrieve`, `analyze`) dan pembanding dev run (`devruns`). |

**Buffer over-fetch** ikut membesar mengikuti exclusion set:
- B: `k + |A| + 5`. A memuat semua jawaban dari semua pertanyaan yang dikecualikan, jadi batas ini tetap berlaku.
- Anchor dan expansion: `top_n + |exclude| + 5`.

Logika `semantic_expansion()` tidak diubah.

## 3. Jumlah yang dikecualikan per mode (pool 1.471 pertanyaan)

| Mode | Pertanyaan dengan exclusion tambahan | Duplicate (q / tetangga) | Linked (q / tetangga) | Near-dup (q / tetangga) | Rata-rata q / a dikecualikan | Maks q / a |
|---|---|---|---|---|---|---|
| `thread` (lama) | 0 | – | – | – | 1,00 / 1,28 | 1 / 8 |
| `dup` | 13 (0,9%) | 13 / 15 | – | – | 1,01 / 1,29 | 3 / 9 |
| **`dup+neardup0.95`** (default) | 16 (1,1%) | 13 / 15 | – | 3 / 7 | 1,01 / 1,29 | 6 / 9 |
| `dup+linked+neardup0.95` | 214 (14,5%) | 13 / 15 | 209 / 300 | 3 / 7 | 1,22 / 1,59 | 11 / 48 |

- **Dev stage-1 (n = 50):** 0 Duplicate, 0 near-dup, 6 Linked.
- **Sampel uji (n = 384, agregat):** 2 Duplicate, 1 near-dup.

`exclusion_sets.parquet`: sha256 `d2eeb0a0…7a3d`. Dibangun ulang dengan hasil byte-identik (test). Waktu build 315 detik, puncak RSS 1,44 GB.

**Kenapa pool berisi 1.471 pertanyaan, bukan 384 × 4 = 1.536.** Pool dibentuk oleh sampler bersama semua kondisi:

1. `get_candidate_questions()` mengambil **1.536** pertanyaan unik (reservoir sample, seed 42) dari pertanyaan yang punya accepted answer di parquet jawaban.
2. `filter_by_token_limit()` lalu membuang **65** pertanyaan yang Title + Body + Tags-nya melebihi 2.048 token cl100k_base (minimum 2.069, median 2.826, maksimum 8.685 token).

Ini kriteria eksklusi pertanyaan evaluasi yang sama sejak Kondisi A, dan sampel uji (posisi 1–384) maupun dev set selalu diambil dari 1.471 pertanyaan yang lolos filter. Tidak ada yang berubah karena tugas ini:

- Urutan 1.536 pertanyaan mentah, sampel uji 384, dan dev 50 identik pada DuckDB dengan thread default, `threads=2`, maupun `threads=1`.
- Hash pool dan dev sama persis dengan sidik jari yang tercatat saat batch kalibrasi (`llm/evaluation/calibration/calibration_runs.json`: `n_pool_after_token_filter` = 1.471, `pool_question_ids_sha256` dan `dev_question_ids_sha256` cocok).

## 4. Audit retrieval-only, filter off vs on (dev 385–434, n = 50, tanpa LLM)

C memakai v3 α = 0,5, sedangkan D dan B memakai konfigurasi default. Self-retrieval tingkat ID diukur sebagai persentase pertanyaan yang konteksnya memuat id dalam exclusion set. Hasilnya sama terhadap definisi `thread`, `default`, maupun `+linked`.

| Kondisi / mode retrieval | ID self-retrieval | (substring ≥ 50, ditinggalkan¹) | Text: cosine ≥ 0,95 | max-cos rata-rata | final_k < 5 / = 0 | latency rata-rata / p95 (detik) |
|---|---|---|---|---|---|---|
| B off | **84,0%** (42/50; 31/35) | 45 | 10 | 0,845 | 0 / 0 | 0,10 / 0,11 |
| B thread / dup / default / +linked | **0%** | 10 | 0 | 0,597 | 0 / 0 | 0,08–0,11 / 0,12–0,29 |
| C off | **100%** (50/50; 35/35) | 50 | 12 | 0,895 | 4 / 0 | 1,04 / 2,47 |
| C thread / dup / default | **0%** | 9 | 0 | 0,544 | 5 / 1 | 0,10–0,58 / 0,17–1,23 |
| C +linked | **0%** | 9 | 0 | 0,545 | 4 / 1 | 0,53 / 1,28 |
| D off | **100%** (50/50; 35/35) | 50 | 12 | 0,895 | 16 / 0 | 0,33 / 0,58 |
| D thread / dup / default / +linked | **0%** | 7 | 0 | 0,506–0,508 | 30 / 1 | 0,02–0,05 / 0,03–0,04 |

**Metrik resmi:** ID-level self-retrieval rate, dan cosine ≥ 0,95 antara konteks dan accepted answer. Cosine ini hanya alat ukur, tidak pernah dipakai sebagai filter.

¹ **Metrik substring ≥ 50 karakter dicoba lalu ditinggalkan** (keputusan penulis, 2026-10-18). Pada HTML mentah metrik ini didominasi boilerplate. Seluruh 26 kecocokan pada mode default (B 10, C 9, D 7) yang diperiksa satu per satu berupa atribut XML Android (`android:layout_width="match_parent"`), URL dokumentasi (`<a href="https://learn.microsoft.com/en-us/dotnet/`), baris dependensi Gradle, atau impor standar; tidak satu pun berupa isi jawaban q. Jadi metrik ini mengukur kemiripan bahasa kode, bukan kebocoran. Angkanya tetap ditampilkan dalam kurung hanya sebagai catatan.

**Pembanding (4b):** tanpa exclusion, pertanyaan yang mengambil thread-nya sendiri adalah 31/35 untuk B (sama dengan tugas paritas), **35/35 untuk C**, dan **35/35 untuk D**. Angka C dan D 100% karena keduanya ber-anchor pada embedding pertanyaan; pertanyaan itu sendiri selalu menjadi anchor teratas bila tidak dikecualikan.

**`final_k < k`:** D (30/50) dan C (5/50) sudah begitu pada exclusion lama; mode baru tidak menambah satu kasus pun. Sumbernya: anchor tanpa jawaban di KG (lihat §6). B tidak pernah kurang dari k.

## 5. Dampak ke stage-1 α (4c)

| | Hasil |
|---|---|
| Exclusion set dev-50 yang berubah, mode default vs thread | **0 / 50** (Duplicate 0, near-dup 0) |
| Retrieval C (id, urutan, skor, teks) default vs thread, α = 0,5 | identik **50 / 50** |
| Retrieval C default vs thread, α = 0,25 | identik **50 / 50** |
| Snapshot 12 q × 4 konfigurasi (v2/v3 × precomputed/embed_query), mode default | identik dengan snapshot pra-refactor (test) |
| Mode `+linked` (bukan default) | 6/50 exclusion set berubah, sehingga konteks C dapat berubah |

**Kesimpulan: stage-1 α tidak perlu diulang** selama Linked tetap `False`. Pemilihan α tidak dijalankan ulang.

## 6. Slot expansion kosong (4d, C, α = 0,5)

| Mode | Question expansion | Tidak menyumbang jawaban | …padahal punya jawaban di KG (artinya semua jawabannya dikecualikan) |
|---|---|---|---|
| off | 438 | 286 (65%) | 0 |
| thread / dup / default | 432 | 277 (64%) | **0** |
| +linked | 438 | 277 (63%) | 0 |

**Tidak ada slot kosong yang disebabkan oleh exclusion.** Seluruh slot kosong berasal dari Question **tanpa jawaban sama sekali** di KG: KG memuat 2,69 juta Question tetapi hanya 725 ribu Answer. Angkanya tidak naik dibanding exclusion lama (277 → 277).

Ini konsisten dengan keputusan tidak mengubah `semantic_expansion()`. Sekitar dua pertiga slot expansion terbuang karena Question tanpa jawaban; inilah masalah yang dulu hendak diatasi fix buffer 6× yang tidak pernah di-commit. Keputusannya tetap di tangan penulis.

## 7. Sensitivitas Linked (4e)

- **(i) Retrieval dev, mode default (Linked tidak dikecualikan):** di B, C (α 0,5 dan 0,25), dan D, **0** item konteks berasal dari tetangga Linked q. Tidak ada yang perlu dicek pada tingkat teks.
- **(ii) Cek langsung tanpa retrieval:** dicek 98 pertanyaan pool **non-test** (posisi > 384) yang punya tetangga Linked dengan jawaban. Setiap jawaban tetangga dibandingkan dengan accepted answer q.
  - **Cosine ≥ 0,95: 0/98.** Tertinggi 0,9468 (q 19621941; keduanya menyarankan instalasi "Android Support Repository"), lalu 0,885, 0,880, dan 0,874.
  - Substring ≥ 50: 32/98, juga boilerplate (blok Gradle `multiDexEnabled true`, layout Android, template Angular, URL GitHub). Substring terpanjang 179 karakter berupa kerangka `@Component({ selector…, templateUrl…, styleUrls…`.
- **Kesimpulan:** tidak ada bukti bahwa tetangga Linked membawa ground-truth q. Linked tetap `False`. Kasus 0,9468 menunjukkan bahwa ambang 0,95 cukup ketat; ambang yang lebih rendah akan mulai menangkap jawaban yang setopik tetapi sah.

## 8. Dev run dengan LLM (n = 35, posisi 385–419, gpt-4o-mini)

B memakai B-plain dengan korpus KG; C memakai v3 α = 0,5 grounded; D grounded. "Thread" berarti exclusion lama. Untuk B, pembandingnya adalah run tugas paritas.

| Run | ID self-retrieval | (substring ≥ 50, ditinggalkan¹) | cosine ≥ 0,95 | Cosine rata-rata / median | NF2 (% kutipan valid) | final_k < 5 / = 0 | latency rata-rata / p95 (detik) |
|---|---|---|---|---|---|---|---|
| B thread (tugas paritas) | 0/35 | 7 | 0 | 0,591 / 0,602 | 74,3 | 0 / 0 | 0,37 / 0,54 |
| **B default** | **0/35** | 7 | 0 | 0,587 / 0,611 | 71,4 | 0 / 0 | 1,11 / 3,39² |
| C thread | 0/35 | 7 | 0 | 0,576 / 0,582 | 74,3 | 3 / 0 | 0,63 / 1,21 |
| **C default** | **0/35** | 7 | 0 | 0,585 / 0,609 | 71,4 | 3 / 0 | 0,77 / 1,60 |
| D thread | 0/35 | 4 | 0 | 0,584 / 0,606 | 54,3 | 22 / 0 | 0,21 / 0,33 |
| **D default** | **0/35** | 4 | 0 | 0,568 / 0,588 | 54,3 | 22 / 0 | 0,28 / 0,54 |

- **Konteks yang diambil identik 35/35** antara thread dan default di B, C, dan D. Selisih cosine (≤ 0,017) dan NF2 (±1 pertanyaan) berasal dari variasi jawaban LLM antar-run, bukan dari retrieval.
- **NF3:** C p95 1,6 detik, jauh di bawah 15 detik.
- ² Kenaikan latency B (rata-rata 1,1 detik, p95 3,4 detik) tidak terkait exclusion. Pada audit retrieval-only, B default 0,086 detik vs thread 0,112 detik. Run ini dijalankan tepat setelah audit C/D memakai ±2 GB memori, sehingga indeks B (1,1 GB) dimuat dalam keadaan cache dingin.

Puncak RSS dev run: B 2,35 GB, C 1,71 GB, D 2,21 GB.

## 9. Test

- **Offline:** suite penuh 981 passed (15 skip = test integrasi). Test baru:
  - `llm/tests/test_leakage_exclusion.py`: mode, settings, perakitan exclusion set (Duplicate dua arah, Linked opsional, ambang near-dup), parquet byte-deterministik, `ExclusionTable`/`LeakageContext`, tidak ada parameter isi jawaban.
  - `llm/b_rag/tests/test_b_leakage.py`: duplikat dan jawabannya tidak pernah diambil; `final_k < k` memicu warning dan tercatat.
- **Integrasi (`GRAPHRAG_INTEGRATION=1`): 14 passed** (9 menit):
  - 20 pertanyaan dev × {thread, default, +linked} × {B, C, D}: irisan = ∅ dan `final_k` tercatat.
  - Kasus **Duplicate nyata** (pertanyaan pool non-test): duplikat tidak muncul sebagai anchor, q2 (traverse dari pertanyaan yang tertaut ke duplikat), Question expansion, maupun jawaban di C; juga tidak di D dan B.
  - Rebuild `exclusion_sets.parquet` menghasilkan sha256 identik.
  - Mode default ekuivalen dengan snapshot C.
  - Snapshot C pra-refactor tetap identik.
- Satu fixture test lama (`test_engine_service_factorial_e2e.py`) ternyata **bocor**: retrieval tiruan mengembalikan pertanyaan 101 untuk pertanyaan 101 itu sendiri. Assert baru menangkapnya, dan yang diperbaiki adalah data tiruannya (anchor 201), bukan assert-nya.

## 10. Keputusan (penulis, 2026-10-18)

1. **Linked tetap mati** (`LEAKAGE_EXCLUDE_LINKED=False`), didukung data di §7. Mode `+linked` tetap tersedia untuk analisis sensitivitas, dan akan mengubah 6/50 exclusion set dev.
2. **Ambang near-dup tetap 0,95.** Ambang ini hanya menangkap 3 pertanyaan pool (7 tetangga), tanpa satu pun di dev. Kasus Linked 0,9468 menunjukkan bahwa ambang di bawah ±0,95 mulai menangkap jawaban setopik yang sah.
3. **Stage-1 α tidak diulang** karena mode default tidak mengubah retrieval dev (§5).
4. **Slot expansion kosong (64%)** berasal dari Question tanpa jawaban, bukan dari exclusion. Diperbaiki terpisah (Amendment 9, branch `fix/expansion-answered`).
5. **Metrik substring ≥ 50 ditinggalkan.** Metrik resmi adalah ID-level self-retrieval rate dan cosine ≥ 0,95 (§4, catatan ¹).

## 11. Draf paragraf Bab III

> **Pencegahan kebocoran data pada retrieval.** Korpus retrieval ketiga kondisi berbasis retrieval (B, C, dan D) memuat juga pertanyaan evaluasi beserta jawabannya, karena knowledge graph dibangun dari seluruh populasi Stack Overflow. Oleh karena itu, pengecualian (exclusion) dilakukan per query: untuk setiap pertanyaan evaluasi *q*, retrieval tidak boleh mengembalikan *q*, seluruh jawaban dalam thread *q* (termasuk accepted answer yang menjadi ground truth), pertanyaan yang ditandai sebagai *Duplicate* dari *q* oleh komunitas Stack Overflow (relasi dua arah) beserta jawabannya, maupun pertanyaan yang hampir identik dengan *q* (kemiripan kosinus embedding judul dan isi pertanyaan ≥ 0,95) beserta jawabannya. Pengecualian ditentukan semata-mata berdasarkan identitas dan kemiripan pertanyaan, tidak berdasarkan isi accepted answer, agar prosedur ini tidak memakai informasi ground truth. Prinsip yang dijaga sejalan dengan yang ditekankan Wu dkk. (2027) dalam konteks penemuan kausal dengan LLM: informasi ground truth yang ikut masuk ke dalam prompt menggelembungkan hasil evaluasi. Penelitian tersebut dirujuk di sini sebagai analogi prinsip, bukan sebagai klaim bahwa Wu dkk. meneliti sistem RAG. Efektivitas prosedur diukur dengan *self-retrieval rate*, yaitu persentase pertanyaan evaluasi yang konteks hasil retrieval-nya memuat salah satu item dalam exclusion set *q*. Sebagai pelengkap, diukur pula kemiripan kosinus antara setiap item konteks dan accepted answer *q* (≥ 0,95 dianggap indikasi kebocoran teks); ukuran ini tidak pernah dipakai sebagai filter. Pada set pengembangan (n = 50), tanpa pengecualian *self-retrieval rate* mencapai 84% untuk Kondisi B dan 100% untuk Kondisi C dan D; dengan pengecualian, angkanya 0% pada ketiga kondisi. Satu keterbatasan tidak dapat diatasi oleh prosedur ini: model bahasa yang digunakan kemungkinan telah melihat data Stack Overflow, termasuk sebagian pertanyaan dan jawaban evaluasi, saat pretraining. Kebocoran semacam itu berada di dalam parameter model dan tidak dapat dihilangkan dengan menyaring konteks retrieval. Dampaknya berlaku sama untuk semua kondisi, termasuk Kondisi A tanpa retrieval, sehingga perbandingan antarkondisi tetap sah, tetapi angka absolut perlu ditafsirkan dengan mempertimbangkan kemungkinan tersebut.

Rujukan (biblatex, `reports/references.bib`, kunci `wu2027reassessing`): Wu, X., Yu, K., Wu, J., & Tan, K. C. (2027). Reassessing the role of LLMs in causal discovery: Critical analysis, benchmark evaluation, and exploratory framework. *Pattern Recognition*, 182, 114825. https://doi.org/10.1016/j.patcog.2026.114825 (online 31 Agustus 2026).
