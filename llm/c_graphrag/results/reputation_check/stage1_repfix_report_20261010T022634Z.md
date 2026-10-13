# Stage 1 ulang dengan perbaikan reputasi (dev set, ctxrel-v2 primary)

- Waktu: 20261010T022634Z  |  git commit: `6b0b40c7da280be362c7193e8be3daf9aebdc4a0`
- Reproduksi base vs file stage 1: identik (50 x 5)
- Item baru yang dinilai: 18  |  durasi 118.5 detik

| alpha | % q >=1 RELEVANT (base -> repfix) | mean score (base -> repfix) | % accepted (base -> repfix) |
|---|---|---|---|
| 0.0 | 48.0 -> 48.0 | 0.3263 -> 0.3263 | 37.29 -> 37.29 |
| 0.25 | 48.0 -> 48.0 | 0.3284 -> 0.3242 | 41.95 -> 41.95 |
| 0.5 | 48.0 -> 48.0 | 0.3114 -> 0.3178 | 52.54 -> 52.12 |
| 0.75 | 46.0 -> 46.0 | 0.286 -> 0.286 | 68.22 -> 67.8 |
| 1.0 | 34.0 -> 36.0 | 0.2288 -> 0.2225 | 74.58 -> 74.58 |

- Selection helper base: top_two [0.5, 0.25] (keep_current False)
- Selection helper repfix: top_two [0.5, 0.25] (keep_current False)

## Aturan R1 (alpha 0.25 dan 0.5; margin seri 2 pp / 0.02)

- alpha=0.25: delta % q = 0.0, delta skor = -0.0042 -> lolos
- alpha=0.5: delta % q = 0.0, delta skor = 0.0064 -> lolos

**Keputusan: SERTAKAN reputasi**

