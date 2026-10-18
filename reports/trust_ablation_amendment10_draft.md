# Amandemen 10 (2026-10-18): ablation trust pada evaluasi resmi

> Versi bahasa Indonesia dari Amendment 10. Versi bahasa Inggris untuk `docs/DECISION_C_SCORING.md` ada di `reports/trust_ablation_amendment10_draft_en.md`.

**Status: dikonfirmasi penulis.** Ditulis setelah Amendment 9 dan **sebelum stage 2 maupun evaluasi resmi pada test sample (n = 384)**. Belum ada hasil stage 2; belum ada generasi atau penilaian judge untuk stage 2 maupun pada test sample.

### Motivasi
- Cek ulang stage-1 Amendment 9 menghasilkan top_two = [0,25; 0,0], dan keduanya nyaris seri (% pertanyaan dengan ≥1 item RELEVANT 48,0 vs 50,0; skor relevansi rata-rata 0,3504 vs 0,3525). α = 0,25 berada di urutan pertama karena tie-break "α lebih besar".
- Stage 2 hanya memilih α di dev set (n = 50). Ini langkah pemilihan, bukan pengukuran efek trustScore, dan hasilnya pun bisa kembali ditentukan oleh tie-break.
- Efek trustScore adalah inti kontribusi tesis, jadi efek itu diukur di data uji.

### Keputusan
1. **Kedua kandidat stage 2 (α = 0,25 dan α = 0) dijalankan pada evaluasi resmi di test sample (n = 384), apa pun hasil stage 2.**
   - **C-utama** = pemenang stage 2 menurut aturan pemilihan yang ada.
   - **C-ablation** = kandidat stage 2 yang lain.
2. **Prosedur identik; hanya α yang berbeda.** Sampel, prompt generasi, model generasi, grounding on dan off, judge, dan seluruh pipeline retrieval sama: tipe edge, max hops, exclusion leakage (Amendment 8), expansion answerable-only (Amendment 9), `use_author_trust`, dan `accepted_only`.
3. **Run terpisah.** Masing-masing run punya config_hash sendiri (α bagian dari config). Label run `c_primary` dan `c_ablation` ditentukan dari hasil stage 2, bukan diisi manual.
4. **Judge.** C-utama dan C-ablation, seperti semua kondisi pada evaluasi resmi, dinilai dengan judge beku yang sama dengan stage 2, memakai metrik hallucination biner Amendment 4.
5. **Analisis utama tidak berubah.** Perbandingan C-utama dengan A, B, dan D tetap analisis utama. C-ablation dilaporkan sebagai tambahan dan tidak menggantikan kondisi mana pun.

### Apa yang dihapus oleh α = 0 (penamaan)
Di C v3, `final_score = (1 − α) · sim_norm + α · trust`. Pada α = 0, ranking akhir hanya memakai similarity:
- trustScore (vote, status accepted, reputasi penulis) tidak masuk ke ranking;
- `use_author_trust` dan `accepted_only` mati pada parameter resmi;
- anchoring, traversal 1-hop, dan semantic expansion tidak memakai sinyal trust apa pun.

**C-ablation (α = 0) menghapus trustScore dari ranking akhir; sinyal accepted/vote masih ada secara tidak langsung hanya lewat urutan pemotongan kandidat 2-hop (LIMIT 50; aktif di 2/50 pertanyaan dev).**

Pemotongan itu mengurutkan kandidat 2-hop berdasarkan `r2.weight · r3.weight`, dengan r3 berupa `HAS_ACCEPTED_ANSWER` (1,0) atau `HAS_ANSWER` (skor vote yang dinormalisasi min-max per pertanyaan). Jalur 2-hop tidak diubah. Karena itu tesis menyebut C-ablation "tanpa trustScore di ranking akhir", bukan "tanpa sinyal trust sama sekali".

### Analisis C-utama vs C-ablation (deskriptif, tanpa uji inferensial)
Pada test sample n = 384, per grounding (on, off) dan gabungan:
- **Hallucination rate biner** tiap run, dengan ABSTAIN dikecualikan dan dengan ABSTAIN dihitung sebagai tidak faktual (Amendment 4).
- **Hitungan berpasangan per pertanyaan** (pertanyaan dan grounding yang sama): berhalusinasi di C-ablation tetapi tidak di C-utama; berhalusinasi di C-utama tetapi tidak di C-ablation; berhalusinasi di keduanya; tidak berhalusinasi di keduanya.
  - Tabel diberikan dengan ABSTAIN dihitung sebagai tidak faktual.
  - Tabel juga diberikan pada pasangan yang kedua jawabannya bukan ABSTAIN, beserta jumlah pasangan yang dikeluarkan.
- **Cosine similarity** terhadap accepted answer dan **NF2** (proporsi jawaban dengan sitasi valid), dilaporkan untuk kedua run dengan cara yang sama seperti kondisi lain.

### Batas klaim (ditetapkan sekarang)
- Bila stage 2 seri dan C-utama terpilih lewat tie-break "α lebih besar", tesis tidak menyebut stage 2 sebagai bukti efek trust.
- Klaim apa pun tentang efek trustScore hanya bersandar pada perbandingan C-utama dengan C-ablation di n = 384.
- Bab III menyebut aturan tie-break "α lebih besar" secara eksplisit, untuk stage 1 dan stage 2.

### Yang tidak berubah
Aturan pemilihan stage 2, kandidat stage 2 (α = 0,25 dan α = 0), judge, prompt, sampel (dev dan test), jalur 2-hop, serta kondisi A, B, dan D.

### Justifikasi
Keputusan ini menambah pengukuran, bukan mengubah pemilihan. Keputusan diambil sebelum stage 2 dan sebelum run resmi n = 384 apa pun, sehingga tidak mungkin bergantung pada hasil keduanya.
