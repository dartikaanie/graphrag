# Uji pengaruh reputasi kontributor pada top-5 Kondisi C

- Waktu: 2026-10-10T01:45:14+00:00  |  git commit: `a3e58a00fc1a1e287f0f0672548749f71fdf51b1`
- Dev set: 50 pertanyaan (posisi 385-434), seed 42, top_k 5
- Tanpa panggilan LLM. Neo4j hanya dibaca. Durasi 100.8 detik.

## Pemeriksaan asumsi graf

- Answer: 725,795; trustScore min 0.100, max 0.892
- Answer dengan authorReputation != 0: 0
- Edge AUTHOR_TRUST: 715,604
- Cakupan AUTHOR_TRUST pada kandidat: 99.2% (793 kandidat unik)
- Asumsi terpenuhi: **ya**

## Hasil (dibandingkan dengan `base` pada alpha yang sama)

| Varian | alpha | Top-5 berubah | Rank-1 berubah | Urutan berubah | Rata-rata Jaccard | Item tertukar | Accepted masuk/keluar | Rata-rata author_weight masuk/keluar |
|---|---|---|---|---|---|---|---|---|
| rep_in_trustscore | 0.0 | 0/50 (0.0%) | 0 | 0 | 1.000 | 0 | 0/0 | -/- |
| rep_in_trustscore | 0.25 | 2/50 (4.0%) | 0 | 7 | 0.987 | 2 | 1/1 | 0.755/0.232 |
| rep_in_trustscore | 0.5 | 3/50 (6.0%) | 2 | 14 | 0.980 | 3 | 2/3 | 1.000/0.706 |
| rep_in_trustscore | 0.75 | 5/50 (10.0%) | 1 | 21 | 0.967 | 5 | 3/4 | 0.903/0.583 |
| rep_in_trustscore | 1.0 | 20/50 (40.0%) | 14 | 37 | 0.830 | 28 | 21/21 | 0.925/0.700 |
| author_switch | 0.0 | 0/50 (0.0%) | 0 | 0 | 1.000 | 0 | 0/0 | -/- |
| author_switch | 0.25 | 9/50 (18.0%) | 6 | 24 | 0.940 | 9 | 4/5 | 0.890/0.586 |
| author_switch | 0.5 | 18/50 (36.0%) | 7 | 40 | 0.871 | 20 | 7/12 | 0.842/0.515 |
| author_switch | 0.75 | 27/50 (54.0%) | 10 | 42 | 0.783 | 35 | 13/27 | 0.919/0.577 |
| author_switch | 1.0 | 31/50 (62.0%) | 19 | 45 | 0.734 | 44 | 23/32 | 0.939/0.613 |

## Kriteria baca (ditetapkan sebelum run)

Varian `rep_in_trustscore`, alpha > 0: NETRAL jika top-5 berubah pada <= 5% pertanyaan di semua alpha; BERPENGARUH jika lebih.

**Hasil: BERPENGARUH** (perubahan terbesar 40.0% pada alpha=1.0).

Catatan: perubahan tidak sama dengan perbaikan. Skrip ini tidak menilai relevansi. Jika BERPENGARUH, nilai relevansi item yang masuk/keluar dengan ctxrel-v2.
