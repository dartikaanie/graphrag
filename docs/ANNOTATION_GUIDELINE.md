# Pedoman Anotasi: Halusinasi pada Jawaban Pertanyaan Pemrograman

Pedoman ini dipakai untuk anotasi manusia pada set kalibrasi dan set konfirmasi
(lihat `docs/DECISION_C_SCORING.md`, Amendment 4 dan Addendum-nya). Definisi di
sini sama dengan definisi yang dipakai judge-v2.1. Semua contoh di pedoman ini
adalah contoh buatan, bukan dari data penelitian.

## 1. Apa yang Anda kerjakan

Setiap baris di berkas anotasi berisi satu item:

| Kolom | Isi | Boleh diubah? |
|---|---|---|
| `item_id` | Kode item (mis. `CAL-017`) | Tidak |
| `question` | Judul dan isi pertanyaan | Tidak |
| `accepted_answer` | Jawaban yang diterima di Stack Overflow. Ini **referensi** Anda | Tidak |
| `candidate_answer` | Jawaban yang dinilai | Tidak |
| `abstain` | `Y` atau `N` | Anda isi |
| `hallucinated` | `Y` atau `N` | Anda isi |
| `severity` | `PENUH` atau `SEBAGIAN` | Anda isi |
| `notes` | Catatan bebas | Anda isi |

Anda tidak diberi tahu sistem mana yang menghasilkan jawaban kandidat. Penanda
sitasi seperti `[SO-12345]` sudah dihapus dari jawaban kandidat; selain itu
teksnya tidak diubah. Teks yang sangat panjang dipotong dan diberi tanda
`[...truncated]`; nilai hanya bagian yang terlihat.

Kerjakan sendiri. Jangan berdiskusi dengan anotator lain sebelum semua item
selesai diisi.

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

### Aturan referensi usang

Referensi bisa sudah usang. Jika sebuah klaim bertentangan dengan referensi,
tetapi Anda **sangat yakin**, berdasarkan pengetahuan yang sudah mapan, bahwa
klaim itu benar untuk platform dan versi yang disebut dalam pertanyaan, anggap
klaim itu **didukung** (bukan salah).

Setiap kali memakai aturan ini, tulis **`referensi usang`** di kolom `notes`,
disertai alasan singkat. Jika Anda ragu, jangan pakai aturan ini; ikuti
referensi.

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

Jika `abstain = Y`, **kosongkan** `hallucinated` dan `severity`, lalu lanjut ke
item berikutnya.

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

## 4. Ringkasan kombinasi yang sah

| `abstain` | `hallucinated` | `severity` | Arti |
|---|---|---|---|
| `Y` | kosong | kosong | Tidak mencoba menjawab |
| `N` | `N` | kosong | Menjawab, tanpa klaim yang salah |
| `N` | `Y` | `PENUH` | Ada kesalahan pada solusi utama |
| `N` | `Y` | `SEBAGIAN` | Kesalahan hanya pada detail sekunder |

Kombinasi lain akan ditolak saat berkas diperiksa.

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

- Isi `abstain` untuk **semua** baris.
- Gunakan pilihan pada sel (`Y`/`N`, `PENUH`/`SEBAGIAN`); jangan mengetik nilai
  lain.
- Jangan mengubah kolom `item_id`, `question`, `accepted_answer`, dan
  `candidate_answer`. Perubahan pada kolom ini akan terdeteksi dan barisnya
  ditolak.
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
