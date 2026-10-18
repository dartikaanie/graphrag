# P1: judge-v2.2 dengan konteks dan tooling anotasi satu orang

Tanggal: 2026-10-19 · Branch: `fix/trust-ablation` (lokal, tidak di-push) · Keputusan: `docs/DECISION_C_SCORING.md`, Amendment 11 · Audit: `reports/evaluation_implementation_audit.md` §2 bagian 7–9, §3 P1-1 sampai P1-7

Tidak ada judge yang dijalankan pada data nyata, tidak ada panggilan API berbayar, dan tidak ada penulisan ke Neo4j. Retrieval, generator, prompt generasi, dan `PROMPT_VERSION` generator tidak diubah. Semua test memakai client tiruan.

## 1. Commit

| Langkah | Commit | Isi |
|---|---|---|
| 1 | `a8f6fb8` | Amendment 11 di decision record (sebelum kode) |
| 3 | `ecbd25e` | blind-v3 (dikerjakan sebelum Langkah 2, karena judge-v2.2 memakainya sejak awal) |
| 2 | `79b907c` | judge-v2.2: prompt, parser, runner |
| 4 | `ff86220` | Lembar anotasi baru + regenerasi lembar kalibrasi |
| 5 | `9df00db` | Penilaian ulang intra-annotator + referensi satu anotator |
| 6 | `ae10ea8` | κ dengan bootstrap CI untuk semua pasangan rater |
| 7 | `872e799` | Set konfirmasi 63 item |
| 8 | `4c7d221` | Probe bias judge |
| 9 | `089184e` | Penjaga pembekuan + log revisi + menu `run.py` |
| 10 | (commit ini) | Laporan, panduan singkat, pembaruan pedoman |

## 2. Perubahan per butir P1

**P1-1: judge menerima konteks** (`llm/evaluation/judge_prompt_v2_2.py`, `llm/evaluation/llm_judge_hallucination_v2_2.py`)

- **Prompt.** `PROMPT_VERSION = "judge-v2.2"` (:45), `REQUEST_TEMPERATURE = 0.0` (:50), `max_tokens` 2.000.
  - `SYSTEM_PROMPT` disusun dari prompt v2.1 apa adanya, ditambah Step 2b (konteks dan sitasi, :73) dan Step 2c (pendekatan), lalu skema keluaran baru.
  - Saat modul dimuat, kode memastikan bagian v2.1 yang disisipi tidak berubah.
- **Input.**
  - Blok konteks `[K1]`…`[K5]`, dengan batas 2.500 karakter per item (`CONTEXT_ITEM_MAX_CHARS`, :54). Di run dev, item terpanjang 2.366 karakter dan mediannya 981, jadi tidak ada yang terpotong.
  - Untuk A, blok konteks ditulis eksplisit "No retrieved context".
- **Peta sitasi.**
  - `citation_map` (:160) dibuat dari konteks record sebelum blinding. Satu `question_id` dipetakan ke semua `[Kn]` dari thread yang sama.
  - `cited_statements` (:171) mengirim kalimat bersitasi beserta `[Kn]`-nya. Sitasi ke ID yang tidak ada di konteks ditandai "not among the context items".
  - Judge tidak pernah melihat ID Stack Overflow.
- **Parser** `parse_judgment_v2_2` (:287).
  - Mulai dari parse v2.1, lalu menambahkan `supported_by_context` (YES/NO; N/A untuk A), `cited_items` dan `cited_item_supports`, serta `approach_match`.
  - Nilai yang tidak valid dianggap parse error.
  - Label `[Kn]` yang tidak ditawarkan di blok sitasi dibuang dan dicatat di `normalized_fields`.
- **Aturan di kode:**
  - HALU: `judge_binary.derive_hallucinated`, tidak berubah;
  - `faithfulness_of` (:266): None untuk A, ABSTAIN, dan jawaban tanpa klaim;
  - `citation_support_of` (:277): None bila tidak ada klaim bersitasi;
  - `approach_match` tidak memengaruhi HALU.
- **Runner** `judge_one` (:111): record mencatat versi judge, blinding, dan parse, temperature 0, `max_tokens`, `citation_map`, serta metrik turunan. Manifest dan riwayat run mencatat pengaturan yang sama.
- **v2.1 tidak diubah.** Hash `SYSTEM_PROMPT` dan `build_messages` v2.1 dikunci di test.

**P1-5: blind-v3** (`llm/evaluation/judge_blinding.py`)

- `BLINDING_VERSION = "blind-v3"` (:34), `NEUTRAL_MARKER = "[catatan: informasi tidak tersedia]"` (:35), `META_PATTERNS` (:51), `blind_candidate_v3` (:95).
- blind-v3 = blind-v2 (penanda sitasi dihapus; `judge_prompt_v1` tidak diubah) ditambah normalisasi frasa meta. Yang diubah hanya frasa meta, sehingga klaim di kalimat yang sama tetap terlihat.
- Judge dan lembar anotasi memakai fungsi yang sama.

**P1-2: lembar anotasi** (`llm/evaluation/annotation_tool.py`)

- Kolom `SHOWN_COLUMNS` (:93) dan `FILL_COLUMNS` (:94).
- `shown_texts` (:128) memanggil `prepare_judge_input`, jadi anotator melihat teks yang sama persis dengan input judge.
- `parse_wrong_claims` (:295) dan `read_annotation_xlsx` (:317): baris `hallucinated = Y` tanpa klaim salah yang berbukti (URL atau kutipan) ditolak, dan setiap masalah menyebut selnya. Lembar berformat lama ditolak dengan pesan yang jelas.
- `annotator_id` sudah terisi per lembar.
- **Regenerasi lembar kalibrasi.** Kedua lembar sebelumnya kosong (0/60, 0/30). Dibandingkan field demi field dengan kunci 2026-10-13:
  - **sama:** seed 42, 60 item, `item_id`, `question_id`, run, kondisi, `config_hash`, subset annotator 2;
  - **berubah, sesuai rencana:** teks kandidat di 22 item (20 B-grounded, 2 B-plain; 0 item A), kolom konteks baru (40 item B), dan sha256 kunci (`802e59c9…` → `1eb982f1…`);
  - kunci lama dicadangkan di luar repo sebagai `~/graphrag_annotation_keys/calibration_key_seed42.blind-v2_2026-10-13.json`.

**P1-4: intra-annotator dan κ** (`annotation_tool.py`, `judge_binary.py`)

- **Penanda selesai.** `validate` mencatat `<annotator>_completed.json` (`record_completion`, :710).
- **`export-reannotation`** (:1041) memilih 12 dari 60 item (seed 42) lewat `select_reannotation` (:1024), dengan ID dan urutan baru, tanpa label lama. Ekspor ditolak bila:
  - lembar belum pernah lolos validasi;
  - lembar berubah sesudah dicatat selesai;
  - belum lewat 14 hari. `--override-min-days "<alasan>"` dicatat di manifest.
  
  Peta ID disimpan di luar repo.
- **`intra-kappa`** (:1101): membandingkan penilaian pertama dan penilaian ulang. Teks dicek terhadap kunci, dan daftar label yang berubah ikut dilaporkan.
- **CI bootstrap di `judge_binary.py`.**
  - `bootstrap_kappa_ci` (:135): persentil 95%, 2.000 resample pasangan item, seed tetap 20261019.
  - `kappa_report` (:164) memberikan hal yang sama untuk skala label apa pun.
  - `agreement_report` (:178) kini menyertakan CI untuk κ 3 kategori dan κ biner.
  - `faithfulness_category` (:207) memetakan rasio: 1 → all, 0 → not, lainnya → partly.
- **`agreement`** (`agreement_pairs` :1162, `cmd_agreement` :1235). Pasangan yang dilaporkan:
  - primer vs peneliti (penentu penerimaan), sekunder vs peneliti, primer vs sekunder;
  - intra-annotator;
  - annotator 2 vs peneliti, bila lembarnya diisi.

  Laporan juga memuat kesepakatan faithfulness serta hasil terpisah untuk item familiar dan tidak familiar, dalam format Markdown dan JSON.
- **Satu anotator.** Lembar annotator 2 yang masih kosong sama sekali dianggap "tidak ada annotator kedua" (`sheet_started` :740, `finalize-reference` :893). Lembar yang terisi sebagian tetap error. Dengan begitu, annotator kedua bisa ditambahkan kelak tanpa mengubah kode.

**P1-3: set konfirmasi** (`annotation_tool.py`)

- `PURPOSES["confirmation"]` (:90): 7 per strata, total 63. Stratanya adalah sel run matrix resmi R1–R9.
- `check_confirmation_allowed` (:597) menolak ekspor bila:
  - Record Amendment 11 masih kosong (`frozen_judge_version`, :532);
  - yang diberikan bukan tepat sembilan run resmi (`confirmation_cells`, :563). Setiap run harus `sample_split = test` dan mencakup tepat sampel test beku.
- Peran run C diambil dari `official_role` di manifest, atau dari α lewat `official_params`.

**P1-6: probe bias** (`llm/evaluation/judge_bias_probes.py`)

- **Sampel.** `sample_probe_items` (:61) memilih 40 jawaban B/C/D dengan seed tetap, dibagi rata antar kondisi.
- **Tiga probe**, semuanya lewat `judge_one` milik runner v2.2:
  - authority: blind-v3 dibandingkan dengan label `[Kn]` yang dibiarkan → flip rate (`flip_stats`, :88);
  - order: urutan konteks diacak dengan seed per item → flip rate;
  - length: korelasi point-biserial (`point_biserial`, :107) per kondisi, dari penilaian blind-v3 yang sama (tanpa panggilan tambahan).
- Hanya untuk dev. Default-nya mode rencana tanpa panggilan API; panggilan judge hanya terjadi dengan `--run`.

**P1-7: penjaga pembekuan** (`llm/evaluation/judge_versions.py`, `docs/JUDGE_REVISION_LOG.md`)

- `CALIBRATION_ALLOWED` = v2.2 + 3 revisi (:31). `OFFICIAL_ALLOWED = ()` (:33) sampai judge dibekukan; sebuah test memastikan nilai ini dan Record Amendment 11 diisi bersamaan.
- `run_purpose` (:60) menentukan jenis run dari file-nya: run kalibrasi terdaftar, run test resmi, atau dev lain.
- `check_version` (:79):
  - **kalibrasi:** versi harus ada di daftar **dan** sudah dicatat di log; versi keempat setelah v2.2 ditolak;
  - **resmi:** hanya versi beku, dan dicek silang dengan Record.
- Runner v2.2 (:255) dan probe bias memanggil penjaga ini sebelum panggilan judge apa pun.
- Menu `run.py:212` sekarang menjalankan runner v2.2.

## 3. Skema keluaran judge-v2.2 dan contoh record dari dry-run

**Per klaim:** `claim`, `verdict` (SUPPORTED / CONTRADICTED / FABRICATED / UNVERIFIABLE), `severity` (CORE / MINOR / null), `reference_conflict`, `supported_by_context` (YES / NO / N/A), `cited_items` (`["K1", ...]`), `cited_item_supports` (YES / NO / null), `evidence`.

**Per jawaban:**
- dari judge: `answer_attempted`, `completeness`, `approach_match`, `reasoning`, dan `label` yang dilaporkan judge;
- diturunkan oleh kode: `hallucinated`, `severity_label`, `faithfulness`, `citation_support`, `n_cited_claims`.

**Dry-run.** Tiga record dev nyata (A, B, C; `question_id` 20952138) dinilai lewat `judge_one` dengan client tiruan. Tidak ada panggilan API.

| Kondisi | `has_context` | item konteks | pernyataan bersitasi | `faithfulness` | `citation_support` | `parse_error` |
|---|---|---|---|---|---|---|
| A | false | 0 | 0 | null | null | false |
| B | true | 5 | 2 | 0.5 | 1.0 | false |
| C | true | 5 | 3 | 0.5 | null* |  false |

\* Client tiruan menjawab `cited_items: ["K1"]`, sedangkan sitasi C ditawarkan sebagai `[K5]`. Parser membuang label yang tidak ditawarkan dan mencatatnya, jadi `citation_support` menjadi null. Ini perilaku yang benar; yang keliru adalah data tiruannya. Pemetaan C sendiri sudah dicek benar: ketiga sitasi `[SO-29044491]` → `[K5]`.

Record B (dipersingkat; `attempts`, `raw_response`, dan metadata respons dihilangkan):

```json
{
 "run_id": "B-9a7690c882", "condition": "B", "question_id": 20952138, "judge_id": "primary",
 "prompt_version": "judge-v2.2", "judge_version": "judge-v2.2", "blinding_version": "blind-v3",
 "parse_version": "pv2.2-1", "temperature": 0.0, "max_tokens": 2000,
 "has_context": true, "n_context_items": 5, "n_cited_statements": 2,
 "citation_map": {"29044491": ["K1"], "11928264": ["K2", "K3"], "20102464": ["K4"], "10919661": ["K5"]},
 "answer_attempted": true, "completeness": "FULL", "approach_match": "SAME",
 "claims": [
  {"claim": "the recommended API exists and solves the problem", "verdict": "SUPPORTED", "severity": null,
   "reference_conflict": false, "evidence": "reference states it",
   "supported_by_context": "YES", "cited_items": ["K1"], "cited_item_supports": "YES"},
  {"claim": "a secondary option is available", "verdict": "UNVERIFIABLE", "severity": null,
   "reference_conflict": false, "evidence": "not addressed",
   "supported_by_context": "NO", "cited_items": [], "cited_item_supports": null}
 ],
 "label": "FAKTUAL", "severity_label": null, "hallucinated": false,
 "faithfulness": 0.5, "citation_support": 1.0, "n_cited_claims": 1, "n_claims": 2,
 "normalized_fields": [], "parse_error": false, "call_failed": false, "truncated": false,
 "probe_citation_labels_kept": false
}
```

## 4. Frasa meta: 20 teratas dan pola blind-v3

**Sumber:** keluaran dev yang sudah ada, 395 jawaban dari 15 file (A 50, B 205, C 70, D 70; termasuk run kalibrasi).

**Hasil:**
- 462 frasa meta dinormalisasi, di 223 jawaban B/C/D (B 90, C 67, D 66);
- 0 dari 50 jawaban A berubah;
- setelah blind-v3, tidak ada lagi kalimat yang menyebut "retrieved sources/context", "provided context", atau "the context provided";
- ungkapan "in the context of …" (5 kemunculan) tidak tersentuh.

| # | Jumlah | Pola | Frasa |
|---|---|---|---|
| 1 | 149 | sources_do_not_cover | the retrieved sources do not cover |
| 2 | 18 | attribution_prefix | based on the context provided |
| 3 | 17 | attribution_prefix | based on the provided context |
| 4 | 15 | sources_do_not_cover | the retrieved sources do not provide |
| 5 | 15 | attribution_prefix | in the context (bukan "in the context of") |
| 6 | 6 | attribution_prefix | based on the given context |
| 7 | 6 | attribution_prefix | based on the available context |
| 8 | 6 | sources_do_not_cover | unfortunately, the retrieved sources do not cover |
| 9 | 5 | sources_do_not_cover | unfortunately, the retrieved sources do not provide |
| 10 | 5 | not_covered_in_sources | are not covered in the retrieved sources |
| 11 | 5 | attribution_prefix | from the context, |
| 12 | 5 | attribution_prefix | in the provided context |
| 13 | 4 | attribution_prefix | based on the retrieved sources |
| 14 | 4 | not_covered_in_sources | is not covered in the retrieved sources |
| 15 | 4 | attribution_prefix | from the provided context |
| 16 | 3 | sources_do_not_cover | however, the retrieved sources do not provide |
| 17 | 3 | attribution_prefix | in the sources |
| 18 | 3 | sources_do_not_cover | the sources retrieved do not cover |
| 19 | 3 | sources_do_not_cover | the context provided does not specifically address |
| 20 | 3 | sources_do_not_cover | the context does not provide |

**Pola.** Daftar lengkapnya ada di `META_PATTERNS`, `judge_blinding.py:51`.

| Pola | Contoh | Hasil |
|---|---|---|
| `sources_do_not_cover` | "(Unfortunately,) the retrieved sources / the context provided do(es) not (specifically) cover/provide/address/… X" | `[catatan: informasi tidak tersedia] X` |
| `no_information_in_sources` | "There is no information in/within the retrieved sources about X" | `[catatan: informasi tidak tersedia] X` |
| `not_covered_in_sources` | "this is not covered in the retrieved sources" | "this is not covered" |
| `attribution_prefix` | "According to / Based on / As mentioned in / From / In the provided context," | dihapus |
| `beyond_sources` | "… needed beyond the context provided" | dihapus |
| `sources_say` | "The retrieved sources (explicitly) confirm/suggest/provide …" | dihapus |

Semua pola mengecualikan "… of" (`_NOT_OF`). Rujukan ke materi milik penanya, seperti "the code snippet you provided", tidak termasuk pola.

## 5. Contoh baris lembar anotasi baru

Baris CAL-003 dari dry-run ekspor (teks dipotong untuk laporan ini). Kolom isian masih kosong, kecuali `annotator_id`.

| Kolom | Isi |
|---|---|
| `item_id` | CAL-003 |
| `question` | "Convert string to RenderFragment … I'm wondering if there's a recommended way of converting string to RenderFragment in code …" |
| `accepted_answer` | "There is no "recommended" method that I know of. A RenderFragment is a delegate defined as: …" |
| `candidate_answer` | "[catatan: informasi tidak tersedia] the specific method or best practice for converting a string to a RenderFragment in Blazor. They do not provide any information on the topic … Therefore, I cannot provide guidance." |
| `context` | "[K1] .toString()); … [K2] dynamically in the code. … [K3] …" |
| `annotator_id` | annotator1 |
| `abstain` … `notes` | (diisi anotator) |

Contoh isian yang valid untuk baris ini (buatan):
- `abstain` = `Y`;
- `domain_familiarity` = `familiar`;
- `uncertain` = `N`;
- kolom lain kosong.

Untuk baris lain, contoh isian `wrong_claims` yang lolos validasi:

    fabricated | RenderFragment.FromString(s) ada di Blazor | https://learn.microsoft.com/dotnet/api/microsoft.aspnetcore.components.renderfragment

Dry-run ekspor ke scratchpad (tanpa panggilan judge) menghasilkan 60 baris. Item dan hash teksnya identik dengan kunci yang di-commit.

## 6. Hasil pytest

Seluruh test offline, dijalankan setelah commit terakhir:

| Direktori | Hasil |
|---|---|
| `llm/tests` | 103 passed |
| `llm/c_graphrag/tests` | 232 passed, 20 skipped |
| `llm/b_rag/tests` | 23 passed, 1 skipped |
| `llm/evaluation/tests` | 485 passed |
| `backend/tests` | 307 passed |
| **Total** | **1.150 passed, 21 skipped** (semua yang di-skip adalah test integrasi `GRAPHRAG_INTEGRATION=1`) |

Test baru di tugas ini (`llm/evaluation/tests/`):
- `test_judge_blinding.py` (6);
- `test_judge_prompt_v2_2.py` (17);
- `test_annotation_sheet_v3.py` (14);
- `test_reannotation.py` (5);
- `test_agreement_ci.py` (10);
- `test_confirmation_set.py` (9);
- `test_judge_bias_probes.py` (6);
- `test_judge_versions.py` (6).

`test_amendment4_tooling.py` disesuaikan dengan format lembar dan set konfirmasi yang baru.

## 7. Yang belum selesai, dan alasannya

- **judge-v2.2 belum pernah dijalankan pada data nyata.** Ini batasan tugas. Bentuk respons model sungguhan, termasuk tingkat parse error pada field baru, baru terlihat saat run pertama.
- **Kode belum menangani kata ganti setelah frasa meta.** Contohnya CAL-003: "[catatan …] … They do not provide any information". Kata ganti seperti ini masih bisa menunjukkan bahwa jawaban berasal dari kondisi RAG. Mendeteksinya butuh analisis kalimat, jadi ini dicatat sebagai keterbatasan.
- **Blok konteks sendiri menunjukkan bahwa jawaban berasal dari kondisi B/C/D.** blind-v3 menyamarkan asal jawaban di dalam teks kandidat, tetapi judge dan anotator tetap bisa membedakan A dari B/C/D lewat ada atau tidaknya konteks. Ini konsekuensi keputusan memberi konteks (Amendment 11) dan perlu ditulis sebagai keterbatasan.
- **Konteks kalibrasi B berasal dari korpus `legacy_sample`.** Item konteksnya potongan HTML yang terputus di tengah, berbeda dari korpus KG yang dipakai run resmi. Run kalibrasi sudah dibuat sebelum Amendment 7, dan audit 8.1 sudah mencatatnya.
- **Isi `OFFICIAL_ALLOWED` dan Record Amendment 11 menunggu pembekuan judge.** Keduanya sengaja dibiarkan kosong.
- **Probe bias belum dijalankan.** Ini butuh panggilan API berbayar, yaitu 3 × 40 panggilan per judge.
- **Analisis deskriptif** (proporsi + Wilson CI) dikerjakan di tugas P2.
- **Alur lama `human-kappa` dan adjudikasi** untuk dua anotator tetap ada dan dipakai bila annotator 2 mengisi lembarnya.

## 8. Langkah berikutnya untuk peneliti

1. **Jawaban dev untuk kalibrasi sudah ada** (A, B-plain, B-grounded; `llm/evaluation/calibration/`), jadi tidak perlu membuat ulang. Buat ulang hanya bila Anda memutuskan kalibrasi memakai korpus B baru (Amendment 7). Itu keputusan terpisah yang mengubah item, dan harus diambil sebelum anotasi dimulai.
2. **Isi lembar kalibrasi** `llm/evaluation/annotation/calibration/annotator1.xlsx` mengikuti `docs/ANNOTATION_QUICKSTART.md`, lalu jalankan `annotation_tool.py validate` sampai bersih. Waktu selesai tercatat otomatis.
3. **Buat referensi:** `annotation_tool.py finalize-reference --purpose calibration`. Dengan satu anotator, referensinya adalah label Anda. Langkah ini membuka guard kalibrasi.
4. **Jalankan judge-v2.2** pada ketiga run kalibrasi: `llm_judge_hallucination_v2_2.py --run-files <3 run kalibrasi> --judge both --out results/jv2_2_calibration.jsonl`. Ini panggilan berbayar.
5. **Hitung κ:** `annotation_tool.py agreement --purpose calibration --judge-output results/jv2_2_calibration.jsonl --judge-version judge-v2.2`. Catat κ beserta CI-nya di `docs/JUDGE_REVISION_LOG.md`.
6. **Revisi bila perlu.** Bila κ primer vs peneliti < 0,6, maksimal 3 revisi (v2.3–v2.5). Untuk setiap revisi: commit prompt baru, catat di log revisi sebelum dijalankan, jalankan, lalu hitung κ lagi. Penjaga menolak revisi keempat. Bila tidak ada versi yang mencapai 0,6, bekukan versi dengan κ tertinggi dan laporkan keterbatasannya (Amendment 4).
7. **Penilaian ulang intra-annotator:** ≥ 14 hari setelah langkah 2, jalankan `export-reannotation`, isi lembarnya, lalu jalankan `intra-kappa`. Hasilnya otomatis ikut di laporan `agreement`.
8. **Bekukan**, dalam satu commit: isi Record Amendment 11, isi `OFFICIAL_ALLOWED = ("<versi>",)` di `judge_versions.py`, dan catat di log revisi. Setelah itu stage 2 dan run resmi baru boleh dijalankan. Set konfirmasi 63 item diekspor setelah kesembilan run resmi selesai.
