# Uji pengaruh reputasi kontributor pada top-5 Kondisi C

- Waktu: 2026-10-10T01:34:19+00:00  |  git commit: `a3e58a00fc1a1e287f0f0672548749f71fdf51b1`
- Dev set: 5 pertanyaan (posisi 385-389), seed 42, top_k 5
- Tanpa panggilan LLM. Neo4j hanya dibaca. Durasi 76.8 detik.

## Pemeriksaan asumsi graf

- Answer: 725,795; trustScore min 0.100, max 0.892
- Answer dengan authorReputation != 0: 0
- Edge AUTHOR_TRUST: 715,604
- Cakupan AUTHOR_TRUST pada kandidat: 100.0% (93 kandidat unik)
- Asumsi terpenuhi: **ya**

## Hasil (dibandingkan dengan `base` pada alpha yang sama)

| Varian | alpha | Top-5 berubah | Rank-1 berubah | Urutan berubah | Rata-rata Jaccard | Item tertukar | Accepted masuk/keluar | Rata-rata author_weight masuk/keluar |
|---|---|---|---|---|---|---|---|---|
| rep_in_trustscore | 0.0 | 0/5 (0.0%) | 0 | 0 | 1.000 | 0 | 0/0 | -/- |
| rep_in_trustscore | 0.25 | 0/5 (0.0%) | 0 | 1 | 1.000 | 0 | 0/0 | -/- |
| rep_in_trustscore | 0.5 | 1/5 (20.0%) | 0 | 1 | 0.933 | 1 | 1/1 | 1.000/0.790 |
| rep_in_trustscore | 0.75 | 0/5 (0.0%) | 0 | 1 | 1.000 | 0 | 0/0 | -/- |
| rep_in_trustscore | 1.0 | 2/5 (40.0%) | 3 | 4 | 0.819 | 3 | 3/3 | 0.837/0.751 |
| author_switch | 0.0 | 0/5 (0.0%) | 0 | 0 | 1.000 | 0 | 0/0 | -/- |
| author_switch | 0.25 | 0/5 (0.0%) | 1 | 4 | 1.000 | 0 | 0/0 | -/- |
| author_switch | 0.5 | 2/5 (40.0%) | 0 | 4 | 0.867 | 2 | 1/2 | 0.964/0.738 |
| author_switch | 0.75 | 3/5 (60.0%) | 1 | 5 | 0.752 | 4 | 3/4 | 0.946/0.725 |
| author_switch | 1.0 | 4/5 (80.0%) | 4 | 5 | 0.638 | 6 | 4/4 | 0.892/0.707 |

## Kriteria baca (ditetapkan sebelum run)

Varian `rep_in_trustscore`, alpha > 0: NETRAL jika top-5 berubah pada <= 5% pertanyaan di semua alpha; BERPENGARUH jika lebih.

**Hasil: BERPENGARUH** (perubahan terbesar 40.0% pada alpha=1.0).

Catatan: perubahan tidak sama dengan perbaikan. Skrip ini tidak menilai relevansi. Jika BERPENGARUH, nilai relevansi item yang masuk/keluar dengan ctxrel-v2.
