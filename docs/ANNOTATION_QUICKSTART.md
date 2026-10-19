# Panduan singkat anotasi (set kalibrasi)

Panduan kerja untuk peneliti sebagai anotator. Keputusan desainnya ada di `docs/DECISION_C_SCORING.md`, Amendment 4 dan 11.

**Pedoman mana yang berlaku:**
- **Lembar kalibrasi v2 (`annotator1_v2.xlsx`) dan set konfirmasi:** `docs/ANNOTATION_GUIDELINE_v2.md` (pedoman v2, dengan aturan waktu, istilah dari penanya, dan faithfulness lawan halusinasi).
- **Lembar kalibrasi v1 yang sudah dibekukan (`annotator1.xlsx`, `annotator1_freeze.json`):** `docs/ANNOTATION_GUIDELINE.md` (pedoman v1). Lembar ini tidak diubah lagi.

Bila panduan ini tampak bertentangan dengan pedoman yang berlaku, pedomanlah yang menang. Bagian 1–6 di bawah ditulis untuk lembar v1; untuk lembar v2 lihat bagian 8.

## 1. Sebelum mulai

1. Baca `docs/ANNOTATION_GUIDELINE.md` sampai selesai, lalu kerjakan latihan L1–L4 di bagian 6 dan cocokkan dengan kuncinya di bagian 8.
2. Buka `llm/evaluation/annotation/calibration/annotator1.xlsx` di Excel atau LibreOffice, sheet **`anotasi`**. Sheet **`petunjuk`** berisi ringkasan aturan per kolom.
3. Ada 60 baris. Kolom A–E (`item_id` sampai `context`) **jangan diubah**: teksnya dicek dengan hash, dan baris yang berubah akan ditolak.
4. Jangan mencari tahu kondisi atau sistem asal jawaban. Jangan membuka file hasil run, laporan judge, atau kunci di `~/graphrag_annotation_keys/` selama anotasi.

## 2. Urutan mengisi satu baris

| Urutan | Kolom | Kapan diisi | Isi |
|---|---|---|---|
| 1 | `abstain` | selalu | `Y` bila jawaban tidak mencoba menjawab, `N` bila mencoba |
| 2 | `hallucinated` | `abstain = N` | `Y` bila ada ≥ 1 klaim salah atau dikarang, `N` bila tidak |
| 3 | `severity` | `hallucinated = Y` | `PENUH` (solusi utama) atau `SEBAGIAN` (detail sekunder) |
| 4 | `wrong_claims` | `hallucinated = Y` | setiap klaim salah, satu per baris, **dengan bukti** (lihat bagian 3) |
| 5 | `faithfulness_level` | `abstain = N` **dan** `context` tidak kosong | `all_supported` / `partly_supported` / `not_supported` |
| 6 | `approach_match` | `abstain = N` | `same` / `different_valid` / `different_unassessable` |
| 7 | `domain_familiarity` | selalu | `familiar` / `not_familiar` |
| 8 | `uncertain` | selalu | `Y` / `N` (lihat bagian 4) |
| 9 | `notes` | bila perlu | catatan bebas; wajib `referensi usang` bila memakai aturan itu |

Kolom `annotator_id` sudah terisi `annotator1`; biarkan.

Saran urutan kerja: baca pertanyaan → baca referensi → baca jawaban kandidat dan tandai klaimnya → (bila ada) baca konteks `[K1]`…`[K5]` → isi kolom 1–9.

## 3. Mengisi `wrong_claims` dengan bukti

Format per baris (pisah baris dalam sel: Alt+Enter, di Mac Option+Enter):

    jenis | klaim | bukti

Contoh isi satu sel (buatan):

    fabricated | datetime.parse_iso(s) ada di Python 3.11 | https://docs.python.org/3/library/datetime.html
    contradicted | fromisoformat ditambahkan di Python 3.5 | "New in version 3.7" (dokumentasi datetime.fromisoformat)
    contradicted | flag --force menghapus cache | "[K2] ... --force only skips the confirmation prompt ..."

Aturan:
- `jenis` hanya `contradicted` atau `fabricated`.
- `bukti` harus berupa **URL** (dokumentasi resmi) atau **kutipan di antara tanda petik** dari referensi, thread, atau konteks. Kalimat seperti "saya rasa salah" ditolak.
- Dokumentasi resmi boleh dibuka hanya untuk memeriksa apakah API, fungsi, atau opsi yang disebut memang ada (pedoman bagian 2, butir 6).

## 4. Kapan memakai `uncertain = Y`

Isi `Y`, dan tulis alasannya di `notes`, bila:
- buktinya tidak tegas, misalnya dokumentasi tidak menyebut versi;
- Anda memakai aturan **referensi usang**;
- Anda ragu apakah jawaban `abstain` atau mencoba menjawab;
- `domain_familiarity = not_familiar` dan label `hallucinated` Anda bergantung pada pengetahuan yang tidak Anda kuasai.

`uncertain = Y` tidak membatalkan label. Labelnya tetap dipakai, dan `uncertain` hanya dilaporkan sebagai informasi tambahan.

## 5. Memeriksa lembar

Dari folder `llm/evaluation`:

    python3 annotation_tool.py validate --purpose calibration --xlsx annotation/calibration/annotator1.xlsx

Setiap masalah disebutkan beserta selnya (misalnya `annotator1.xlsx!J14 (CAL-013): ...`). Perbaiki semuanya sampai hasilnya `0 problem(s)`.

Saat lembar pertama kali lolos tanpa masalah, waktu selesainya dicatat di `annotation/calibration/annotator1_completed.json`. Jadwal penilaian ulang dihitung dari waktu ini. **Jangan mengubah lembar lagi setelah itu**: lembar yang berubah harus divalidasi ulang, dan jadwalnya mulai dari awal.

Lembar `annotator2.xlsx` biarkan kosong bila tidak ada anotator kedua. Lembar yang kosong sama sekali dibaca sebagai "anotasi satu orang".

## 6. Jadwal penilaian ulang (intra-annotator)

1. **Hari 0:** lembar `annotator1.xlsx` lolos `validate`.
2. **Hari ≥ 14:** ekspor 20% item (12 dari 60) dengan kode dan urutan baru:

       python3 annotation_tool.py export-reannotation --purpose calibration

   Sebelum 14 hari, ekspor ditolak. Pengecualian hanya dengan `--override-min-days "<alasan>"`, dan alasannya dicatat di manifest.
3. Isi `annotation/calibration/reannotation_annotator1.xlsx` **tanpa melihat** lembar pertama, dengan aturan yang sama.
4. Hitung kesepakatan dengan diri sendiri:

       python3 annotation_tool.py intra-kappa --purpose calibration

## 7. Setelah lembar selesai

Langkah berikutnya (judge-v2.2 pada set kalibrasi, κ, revisi, pembekuan) ada di ringkasan `reports/p1_judge_annotation.md`, bagian "Langkah berikutnya untuk peneliti".

## 8. Lembar kalibrasi v2 (`annotator1_v2.xlsx`)

1. Baca `docs/ANNOTATION_GUIDELINE_v2.md`, terutama aturan 7 (aturan waktu), aturan 8 (istilah dari penanya), dan bagian "Konteks yang salah terap" di Langkah 5.
2. Lembar v2 adalah salinan lembar v1 yang dibekukan, dengan label v1 sudah terisi, ditambah empat kolom di akhir:
   - `v2_change_reason`: isi **hanya** pada baris yang labelnya Anda ubah, dengan alasannya;
   - `question_date`, `reference_date`: tanggal dari data Stack Overflow, **jangan diubah**;
   - `time_rule_check`: `Y` pada item yang perlu diperiksa ulang dengan aturan waktu, **jangan diubah**.
3. Ubah hanya item yang memang perlu Anda nilai ulang. Kolom teks (`item_id` sampai `context`) tetap tidak boleh diubah.
4. Setelah selesai, simpan dan tutup Excel, lalu beri tahu. Lembar diperiksa dengan pemeriksaan yang sama seperti v1, dibandingkan sel per sel dengan v1, lalu dibekukan.
