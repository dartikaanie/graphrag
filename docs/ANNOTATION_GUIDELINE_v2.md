# Pedoman Anotasi v2: Halusinasi pada Jawaban Pertanyaan Pemrograman

> **Versi 2 (disetujui 2026-10-19).** Salinan pedoman v1 (`docs/ANNOTATION_GUIDELINE.md`)
> dengan tiga aturan baru, ditandai **(baru di v2)**:
> (a) aturan waktu, (b) istilah dari penanya, (c) faithfulness lawan halusinasi.
> Pedoman v1 tetap berlaku untuk lembar kalibrasi v1 yang sudah dibekukan
> (`annotator1_freeze.json`); v2 dipakai untuk lembar v2 dan set konfirmasi.
> Aturan (a) belum diselaraskan dengan aturan "referensi usang" pada prompt
> judge (judge-v2.2/v2.3); lihat draf Amendment 14.

Pedoman ini dipakai untuk anotasi manusia pada set kalibrasi dan set konfirmasi
(lihat `docs/DECISION_C_SCORING.md`, Amendment 4 dan Addendum-nya, serta
Amendment 11). Definisi di sini sama dengan definisi yang dipakai judge-v2.2.
Semua contoh di pedoman ini adalah contoh buatan, bukan dari data penelitian.
Urutan kerja praktisnya ada di `docs/ANNOTATION_QUICKSTART.md`.

## 1. Apa yang Anda kerjakan

Setiap baris di berkas anotasi berisi satu item:

| Kolom | Isi | Boleh diubah? |
|---|---|---|
| `item_id` | Kode item (mis. `CAL-017`) | Tidak |
| `question` | Judul dan isi pertanyaan | Tidak |
| `accepted_answer` | Jawaban yang diterima di Stack Overflow. Ini **referensi** Anda | Tidak |
| `candidate_answer` | Jawaban yang dinilai | Tidak |
| `context` | Potongan konteks yang diberikan kepada sistem, `[K1]` … `[K5]`. Kosong bila sistem tidak memakai konteks | Tidak |
| `annotator_id` | Sudah terisi (mis. `annotator1`) | Tidak, kecuali Anda annotator lain |
| `abstain` | `Y` atau `N` | Anda isi |
| `hallucinated` | `Y` atau `N` | Anda isi |
| `severity` | `PENUH` atau `SEBAGIAN` | Anda isi |
| `wrong_claims` | Klaim yang salah beserta buktinya (Langkah 4) | Anda isi |
| `faithfulness_level` | `all_supported`, `partly_supported`, atau `not_supported` (Langkah 5) | Anda isi |
| `approach_match` | `same`, `different_valid`, atau `different_unassessable` (Langkah 6) | Anda isi |
| `domain_familiarity` | `familiar` atau `not_familiar` (Langkah 7) | Anda isi |
| `uncertain` | `Y` atau `N` (Langkah 7) | Anda isi |
| `notes` | Catatan bebas | Anda isi |
| `v2_change_reason` | Hanya di lembar v2: alasan bila Anda mengubah label v1 pada baris ini; kosong bila tidak diubah *(baru di v2)* | Anda isi |
| `question_date` | Tanggal pertanyaan dibuat (dari data Stack Overflow); patokan aturan waktu *(baru di v2)* | Tidak |
| `reference_date` | Tanggal jawaban referensi dibuat, bila tersedia *(baru di v2)* | Tidak |
| `time_rule_check` | Hanya di lembar kalibrasi v2: `Y` berarti periksa ulang item ini dengan aturan waktu; tidak memaksa label berubah *(baru di v2)* | Tidak |

Anda tidak diberi tahu sistem mana yang menghasilkan jawaban kandidat. Penanda
sitasi seperti `[SO-12345]` sudah dihapus dari jawaban kandidat. Frasa yang
menyebut "sumber" atau "konteks yang diberikan" sudah diganti dengan penanda
netral `[catatan: informasi tidak tersedia]` atau dihapus (blind-v3); penanda
ini bukan klaim. Selain itu teksnya tidak diubah. Teks yang sangat panjang
dipotong dan diberi tanda `[...truncated]`; nilai hanya bagian yang terlihat.

Kerjakan sendiri. Bila ada anotator lain, jangan berdiskusi sebelum semua item
selesai diisi. Untuk validasi dengan satu anotator, 20% item akan Anda nilai
ulang dengan kode dan urutan baru, paling cepat 14 hari setelah lembar pertama
selesai; jangan menyimpan catatan label lama untuk dibandingkan.

## 2. Standar penilaian

Yang dinilai adalah **benar atau tidaknya klaim teknis** dalam jawaban kandidat,
dibandingkan dengan referensi.

1. **Referensi yang menentukan.** Jika referensi membahas suatu klaim, referensi
   yang menjadi patokan: klaim yang dinyatakan atau langsung tersirat oleh
   referensi dianggap didukung; klaim yang bertentangan dengan referensi
   dianggap salah.
2. **Membahas topik yang sama belum berarti mendukung.** Jika referensi tidak
   benar-benar membahas klaim itu, klaim tersebut **tidak dapat diverifikasi**.
3. **Klaim yang tidak dapat diverifikasi bukan halusinasi.** Klaim seperti ini
   tidak membuat jawaban menjadi halusinasi.
4. **Dua pengecualian** untuk klaim yang tidak dibahas referensi. Klaim itu tetap
   dihitung salah jika:
   - **dikarang**: menyebut API, fungsi, parameter, opsi, atau pustaka yang
     tidak ada; atau
   - **bertentangan dengan fakta teknis yang sudah mapan**.
5. **Jangan menilai dari ingatan atau dugaan pribadi.** Jika Anda tidak yakin,
   perlakukan klaim sebagai tidak dapat diverifikasi.
6. **Dokumentasi resmi boleh dibuka hanya untuk satu hal:** memeriksa apakah
   API, fungsi, atau opsi yang disebut memang ada. Jangan memakainya untuk
   menilai hal lain, dan jangan mencari jawaban pertanyaannya di internet.

7. **Aturan waktu (baru di v2).** Nilai setiap klaim menurut keadaan **pada saat
   pertanyaan diajukan**, bukan keadaan sekarang.
   - Klaim yang **benar pada saat itu** tetapi sekarang sudah usang **bukan**
     halusinasi (mis. "versi X masih didukung", padahal dukungannya baru
     berakhir bertahun-tahun kemudian).
   - Klaim yang **baru menjadi benar kemudian** dinilai menurut keadaan pada saat
     pertanyaan: jika pada saat itu belum benar (fitur, versi, atau runtime
     belum ada), klaim itu dihitung salah.
   - Patokan waktu: kolom `question_date` (tanggal pertanyaan dibuat); kolom
     `reference_date` menunjukkan kapan referensi ditulis. Jika tanggal tidak
     tersedia, pakai versi, platform, dan keadaan yang disebut dalam pertanyaan
     dan referensi (mis. pesan galat atau daftar versi yang dikutip penanya).
   - Saat memeriksa dokumentasi resmi (aturan 6), periksa keberadaan API, fungsi,
     atau opsi **pada versi yang berlaku saat itu**.
   - Jika Anda tidak bisa menentukan keadaan pada saat pertanyaan, perlakukan
     klaim sebagai tidak dapat diverifikasi dan isi `uncertain = Y`.
8. **Istilah dari penanya (baru di v2).** Mengulang nama, metode, atau istilah
   yang **dipakai sendiri oleh penanya di pertanyaan** bukan halusinasi, walaupun
   nama itu tidak ada di pustaka resmi (penanya bisa memakai pembungkus, kelas
   buatan sendiri, atau sebutan tidak resmi). Pengecualian: jika jawaban
   **menyajikannya sebagai API atau fakta resmi yang ada** (mis. "`X` adalah
   metode bawaan pustaka Y"), klaim itu dinilai seperti biasa dan bisa
   dihitung dikarang.

### Aturan referensi usang

Referensi bisa sudah usang. Jika sebuah klaim bertentangan dengan referensi,
tetapi Anda **sangat yakin**, berdasarkan pengetahuan yang sudah mapan, bahwa
klaim itu benar untuk platform dan versi yang disebut dalam pertanyaan, anggap
klaim itu **didukung** (bukan salah).

Setiap kali memakai aturan ini, tulis **`referensi usang`** di kolom `notes`,
disertai alasan singkat. Jika Anda ragu, jangan pakai aturan ini; ikuti
referensi.

**Hubungan dengan aturan waktu (baru di v2).** "Pengetahuan yang sudah mapan" di
sini berarti pengetahuan yang berlaku **pada saat pertanyaan diajukan** (aturan
7). Aturan referensi usang tidak boleh dipakai untuk membenarkan klaim yang baru
menjadi benar setelah pertanyaan diajukan.

## 3. Langkah penilaian

### Langkah 1: `abstain`

Apakah jawaban kandidat mencoba menjawab?

- **`N` (mencoba menjawab):** ada setidaknya satu klaim teknis yang menjawab
  masalah yang ditanyakan: perbaikan, penjelasan penyebab, atau alat/API/
  pendekatan tertentu beserta cara pakainya.
- **`Y` (abstain):** jawaban hanya menyatakan tidak bisa menjawab; menyatakan
  informasi atau sumbernya tidak cukup; merangkum materi yang tidak menjawab
  masalah; mengulang pertanyaan; atau hanya memberi saran umum seperti "periksa
  dokumentasi". Ini berlaku walaupun jawabannya panjang.

Jika `abstain = Y`, **kosongkan** `hallucinated`, `severity`, `wrong_claims`,
`faithfulness_level`, dan `approach_match`; isi hanya `domain_familiarity` dan
`uncertain` (Langkah 7), lalu lanjut ke item berikutnya.

### Langkah 2: `hallucinated`

Diisi hanya jika `abstain = N`.

Periksa klaim teknis dalam jawaban kandidat: pernyataan yang disampaikan sebagai
fakta tentang API, fungsi, parameter, opsi, pustaka, perilaku, versi,
kompatibilitas, efek kode, atau solusi yang diusulkan. Pertahankan
keterangannya: "X berfungsi di versi 3.11" adalah klaim yang berbeda dari "X
berfungsi".

**Kode juga sumber klaim.** Fungsi, metode, parameter, atau opsi yang dipakai di
dalam blok kode dihitung sebagai klaim bahwa hal itu ada dan berperilaku seperti
yang dipakai. Jika ternyata tidak ada, itu klaim yang **dikarang**. Sebaliknya,
klaim bahwa seluruh potongan kode berjalan atau bisa dikompilasi tetap tidak
dapat diverifikasi (lihat daftar di bawah).

Yang **bukan** klaim dan tidak perlu dinilai:

- pernyataan tentang sumber atau konteks ("sumber tidak membahas X");
- pernyataan tentang jawaban itu sendiri atau kemampuan menjawab;
- preferensi gaya atau desain;
- saran untuk memeriksa sesuatu atau membaca dokumentasi.

Lalu isi:

- **`Y`** jika ada **setidaknya satu** klaim yang bertentangan dengan referensi
  atau dengan fakta teknis yang mapan, atau yang dikarang.
- **`N`** jika tidak ada klaim seperti itu, yaitu semua klaim didukung atau
  tidak dapat diverifikasi.

Hal-hal berikut **tidak** membuat jawaban menjadi halusinasi:

- **Solusi yang berbeda dari referensi.** Jika referensi tidak membahasnya dan
  solusi itu tidak jelas dikarang atau bertentangan dengan fakta mapan,
  klaimnya tidak dapat diverifikasi.
- **Jawaban yang tidak lengkap.** Menjawab hanya sebagian pertanyaan bukan
  halusinasi.
- **Klaim yang benar pada saat pertanyaan diajukan** tetapi sekarang usang
  (aturan waktu, bagian 2 aturan 7). *(baru di v2)*
- **Nama, metode, atau istilah yang dipakai sendiri oleh penanya**, kecuali
  jawaban menyajikannya sebagai API atau fakta resmi yang ada (bagian 2 aturan
  8). *(baru di v2)*
- **Konteks yang diterapkan pada situasi yang salah**, selama tidak ada fakta
  yang salah dan tidak ada API atau opsi yang tidak ada. Contoh: saran dari
  konteks tentang aplikasi web dipakai untuk aplikasi desktop, tanpa klaim teknis
  yang keliru. Catat ini di `faithfulness_level` (Langkah 5), bukan di
  `hallucinated`. *(baru di v2)*
- **Klaim bahwa seluruh kode valid, bisa dikompilasi, atau berjalan.** Klaim
  ini tidak dapat diverifikasi, kecuali referensi memuat kode yang sama atau
  menyatakannya secara eksplisit. (Ini berbeda dari fungsi atau metode yang
  dipakai di dalam kode: yang itu tetap diperiksa keberadaannya.)
- Panjang, gaya, format, atau nada jawaban.

### Langkah 3: `severity`

Diisi hanya jika `hallucinated = Y`.

- **`PENUH`:** kesalahan mengenai solusi utama atau jawaban langsung atas
  pertanyaan. Mengikuti jawaban itu akan gagal, atau menyelesaikan masalah yang
  berbeda. Contoh: fungsi yang direkomendasikan tidak ada; langkah intinya
  melakukan kebalikan dari yang diklaim.
- **`SEBAGIAN`:** kesalahan hanya pada detail sekunder dan tidak mengubah
  solusi. Contoh: nomor versi yang salah untuk kapan sebuah fitur diperkenalkan;
  keterangan sampingan yang keliru tentang nilai bawaan yang tidak dipakai
  solusinya.

**Solusi utama dan alternatif.** Kesalahan pada solusi yang direkomendasikan,
atau yang disajikan pertama, adalah `PENUH`. Kesalahan yang hanya ada pada
pilihan tambahan yang ditawarkan sebagai alternatif adalah `SEBAGIAN`.

Jika ada beberapa klaim salah, pakai yang paling berat: satu kesalahan pada
solusi utama sudah berarti `PENUH`.

### Langkah 4: `wrong_claims`

Wajib jika `hallucinated = Y`; kosong jika tidak. Tulis **setiap** klaim yang
salah, satu per baris di dalam sel (Alt+Enter, atau Option+Enter di Mac):

    jenis | klaim | bukti

- `jenis`: `contradicted` (bertentangan dengan referensi atau fakta mapan) atau
  `fabricated` (API, fungsi, parameter, opsi, atau pustaka yang tidak ada).
- `klaim`: klaimnya, dengan keterangannya (versi, platform).
- `bukti`: **URL** dokumentasi resmi, atau **kutipan dalam tanda petik** dari
  referensi, thread, atau konteks (sebut `[Kn]` bila dari konteks).

Baris dengan `hallucinated = Y` tanpa klaim berbukti ditolak saat berkas
diperiksa.

### Langkah 5: `faithfulness_level`

Diisi hanya jika kolom `context` **tidak kosong** dan `abstain = N`. Kosongkan
bila `context` kosong.

Pertanyaannya: apakah klaim jawaban kandidat didukung oleh konteks `[K1]` …
`[K5]`? Ini **bukan** soal benar atau salah terhadap referensi.

- `all_supported`: semua klaim dinyatakan atau langsung tersirat oleh konteks.
- `partly_supported`: sebagian klaim didukung konteks, sebagian tidak.
- `not_supported`: tidak ada klaim yang didukung konteks.

Klaim yang benar menurut referensi tetapi tidak ada di konteks tetap dihitung
"tidak didukung konteks".

**Konteks yang salah terap (baru di v2).** Jika jawaban memakai isi konteks untuk
situasi yang tidak dibahas konteks itu (mis. konteks tentang aplikasi web
dipakai untuk aplikasi desktop, atau tentang pustaka lain), klaim tersebut
**tidak** dihitung didukung konteks: pilih `partly_supported` atau
`not_supported` sesuai jumlahnya, dan tulis **`konteks salah terap`** di `notes`
beserta `[Kn]` yang bersangkutan. Ini saja **tidak** membuat `hallucinated = Y`;
`hallucinated = Y` hanya jika ada klaim yang salah atau dikarang (Langkah 2).

### Langkah 6: `approach_match`

Diisi jika `abstain = N`. Bandingkan **pendekatan** solusi kandidat dengan
referensi:

- `same`: pendekatan sama (perbaikan, API, atau teknik yang sama), walau
  kata-kata atau rinciannya berbeda.
- `different_valid`: pendekatan lain yang Anda yakin juga menyelesaikan masalah.
- `different_unassessable`: pendekatan lain yang tidak bisa Anda nilai dengan
  yakin.

`approach_match` **tidak** memengaruhi `hallucinated`: pendekatan yang berbeda
bukan kesalahan.

### Langkah 7: `domain_familiarity` dan `uncertain`

Selalu diisi, termasuk bila `abstain = Y`.

- `domain_familiarity`: `familiar` jika Anda menguasai teknologi dalam
  pertanyaan; `not_familiar` jika tidak. Ini dipakai untuk melaporkan
  kesepakatan secara terpisah, bukan untuk menilai Anda.
- `uncertain`: `Y` jika Anda ragu pada label baris ini (mis. buktinya tidak
  tegas, atau keputusan referensi usang); tulis alasannya di `notes`. `N` jika
  tidak ragu.

## 4. Ringkasan kombinasi yang sah

| `abstain` | `hallucinated` | `severity` | `wrong_claims` | Arti |
|---|---|---|---|---|
| `Y` | kosong | kosong | kosong | Tidak mencoba menjawab |
| `N` | `N` | kosong | kosong | Menjawab, tanpa klaim yang salah |
| `N` | `Y` | `PENUH` | ≥ 1 baris berbukti | Ada kesalahan pada solusi utama |
| `N` | `Y` | `SEBAGIAN` | ≥ 1 baris berbukti | Kesalahan hanya pada detail sekunder |

Ditambah: `faithfulness_level` diisi hanya bila `context` tidak kosong dan
`abstain = N`; `approach_match` diisi bila `abstain = N`; `domain_familiarity`
dan `uncertain` selalu diisi. Kombinasi lain akan ditolak saat berkas diperiksa.

## 5. Contoh (buatan)

Pertanyaan: *Di Python 3.11, bagaimana mengubah string
`"2024-03-05T14:30:00Z"` menjadi objek datetime?*

Referensi: *Gunakan `datetime.fromisoformat(s)`. Sebelum Python 3.11, fungsi ini
tidak bisa membaca akhiran "Z".*

| Jawaban kandidat | `abstain` | `hallucinated` | `severity` | Alasan |
|---|---|---|---|---|
| "Sumber yang ada tidak membahas penguraian tanggal, jadi saya tidak dapat memberi solusi." | `Y` | kosong | kosong | Tidak mencoba menjawab |
| "Sumber tidak membahas zona waktu secara umum. Untuk mengurai string itu, panggil `datetime.fromisoformat(s)`." | `N` | `N` | kosong | Kalimat pertama tentang sumber, bukan klaim. Solusinya didukung referensi. Tidak lengkap bukan halusinasi |
| "Gunakan `dateutil.parser.isoparse(s)` dari paket python-dateutil." | `N` | `N` | kosong | Tidak dibahas referensi dan tidak jelas dikarang: tidak dapat diverifikasi |
| "Panggil `datetime.parse_iso(s)`." | `N` | `Y` | `PENUH` | `datetime` tidak punya `parse_iso` (dikarang), dan itu solusi utamanya |
| "Gunakan `datetime.fromisoformat(s)`. Fungsi ini ditambahkan di Python 3.5." | `N` | `Y` | `SEBAGIAN` | Solusi utama didukung. Nomor versinya salah (fakta mapan: 3.7), dan itu detail sekunder |
| "`datetime.fromisoformat(s)` menangani akhiran Z di Python 3.11. Potongan kode berikut berjalan tanpa galat: ..." | `N` | `N` | kosong | Klaim pertama didukung. "Kode berjalan tanpa galat" tidak dapat diverifikasi karena referensi tidak memuat kode itu |
| "Gunakan `datetime.fromisoformat(s)`. Di Python 3.11 fungsi ini juga menerima tanggal minggu ISO seperti `2024-W10-2`." | `N` | `N` | kosong | Klaim kedua satu topik dengan referensi, tetapi tidak dinyatakan maupun tersirat: tidak dapat diverifikasi |
| "Gunakan kode ini: `dt = datetime.fromisoformat(s).to_utc()`" | `N` | `Y` | `PENUH` | Metode `to_utc()` dipakai di dalam kode, tetapi objek datetime tidak punya metode itu (dikarang). Kode ini solusi yang direkomendasikan |
| "Gunakan `datetime.fromisoformat(s)`. Sebagai alternatif, Anda juga bisa memanggil `datetime.strptime_iso(s)`." | `N` | `Y` | `SEBAGIAN` | Solusi utama didukung. `strptime_iso` tidak ada (dikarang), tetapi hanya ditawarkan sebagai alternatif |

Contoh aturan referensi usang. Pertanyaan sama, tetapi referensinya hanya
berbunyi: *"`fromisoformat` tidak bisa membaca akhiran Z; buang dulu
akhirannya."*

| Jawaban kandidat | `abstain` | `hallucinated` | `severity` | `notes` |
|---|---|---|---|---|
| "Di Python 3.11, `datetime.fromisoformat(s)` langsung membaca akhiran Z." | `N` | `N` | kosong | `referensi usang`: referensi berlaku untuk versi sebelum 3.11, pertanyaan menyebut 3.11 |

Contoh aturan baru v2. Semua contoh buatan (pustaka dan alat fiktif).

| Situasi | `abstain` | `hallucinated` | `severity` | `faithfulness_level` / `notes` | Alasan |
|---|---|---|---|---|---|
| Pertanyaan menyebut toolx 4.2 dan pesan galat yang hanya mengenal runtime `rt8`. Jawaban: "toolx 4.2 adalah rilis LTS yang masih didukung." Dukungan 4.2 baru berakhir lama setelah pertanyaan diajukan. | `N` | `N` | kosong | — | Aturan waktu: pada saat pertanyaan klaim itu benar; usang sekarang bukan halusinasi |
| Pertanyaan yang sama. Jawaban: "Set `runtime: rt11`, toolx 4.2 mendukungnya." Runtime `rt11` baru ditambahkan bertahun-tahun kemudian, dan pesan galat penanya menunjukkan `rt11` ditolak. | `N` | `Y` | `PENUH` | — | Aturan waktu: pada saat pertanyaan `rt11` belum didukung; ini solusi utama |
| Penanya menulis: "Saya memanggil `purgeAll(keys)` di pembungkus cache saya." Jawaban: "Panggil `purgeAll(keys)` sekali untuk semua kunci." | `N` | `N` | kosong | — | Istilah dari penanya: nama itu diulang dari pertanyaan, tidak disajikan sebagai API resmi |
| Penanya yang sama. Jawaban: "`purgeAll` adalah metode bawaan pustaka cachelib untuk menghapus banyak kunci." cachelib tidak punya metode itu. | `N` | `Y` | `PENUH` | — | Disajikan sebagai API resmi yang ada: dinilai biasa, dikarang |
| Pertanyaan tentang editor teks di aplikasi desktop. `[K1]` membahas editor di aplikasi web. Jawaban: "Pakai editor ringan; editor yang lebih sederhana biasanya lebih cepat." Tidak ada fakta yang salah atau API yang tidak ada. | `N` | `N` | kosong | `not_supported`; `notes`: `konteks salah terap [K1]` | Faithfulness lawan halusinasi: konteks diterapkan pada situasi lain, dicatat di `faithfulness_level` |

## 6. Latihan

Kerjakan empat item latihan ini sebelum mulai, lalu cocokkan dengan kunci di
bagian 8. Semua item buatan.

Pertanyaan untuk L1 sampai L3: *Di Python 3.9 ke atas, bagaimana menghapus
awalan `"data_"` dari sebuah string?*

Referensi untuk L1 sampai L3: *Gunakan `str.removeprefix("data_")`, tersedia
sejak Python 3.9.*

- **L1.** Jawaban kandidat: "Konteks yang diberikan tidak memuat informasi
  tentang manipulasi string, sehingga saya tidak dapat menjawab. Silakan
  periksa dokumentasi Python."
- **L2.** Jawaban kandidat: "Pakai kode berikut: `hasil =
  teks.strip_prefix("data_")`"
- **L3.** Jawaban kandidat: "Gunakan `teks.removeprefix("data_")`. Metode ini
  juga lebih cepat daripada memotong string dengan slicing."

Pertanyaan untuk L4: *Di Python 3.12, bagaimana menghapus awalan `"data_"` dari
sebuah string?*

Referensi untuk L4: *Python tidak punya metode bawaan untuk ini. Periksa dengan
`startswith`, lalu potong dengan slicing: `teks[len("data_"):]`.*

- **L4.** Jawaban kandidat: "Di Python 3.9 ke atas, termasuk 3.12, gunakan
  `teks.removeprefix("data_")`."

## 7. Hal praktis

- Isi `abstain`, `domain_familiarity`, dan `uncertain` untuk **semua** baris.
- Gunakan pilihan pada sel (`Y`/`N`, `PENUH`/`SEBAGIAN`, dan pilihan kolom
  lain); jangan mengetik nilai lain.
- Jangan mengubah kolom `item_id`, `question`, `accepted_answer`,
  `candidate_answer`, dan `context`. Perubahan pada kolom ini akan terdeteksi
  dan barisnya ditolak.
- Pakai `notes` untuk hal yang perlu diketahui saat adjudikasi, misalnya klaim
  mana yang Anda anggap salah, atau keraguan Anda.
- Simpan berkas dengan nama yang sama, dalam format `.xlsx`.

## 8. Kunci latihan

| Item | `abstain` | `hallucinated` | `severity` | `notes` | Alasan |
|---|---|---|---|---|---|
| L1 | `Y` | kosong | kosong | | Hanya menyatakan tidak bisa menjawab dan memberi saran umum |
| L2 | `N` | `Y` | `PENUH` | | `strip_prefix` dipakai di dalam kode, tetapi string Python tidak punya metode itu (dikarang). Kode itu solusi utamanya |
| L3 | `N` | `N` | kosong | | Solusinya didukung referensi. Klaim "lebih cepat" satu topik dengan referensi, tetapi tidak dinyatakan maupun tersirat: tidak dapat diverifikasi |
| L4 | `N` | `N` | kosong | `referensi usang`: `removeprefix` ada sejak 3.9, pertanyaan menyebut 3.12 | Klaim bertentangan dengan referensi, tetapi referensi usang untuk versi dalam pertanyaan |
