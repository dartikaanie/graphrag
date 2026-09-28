"""
_stats_common.py
=====================================
Statistik kecil dipakai BERSAMA oleh analyze_trust_vs_relevance.py dan
compare_conditions_stats.py -- Cliff's delta (effect size non-parametrik)
dan koreksi Holm-Bonferroni. TIDAK ADA panggilan LLM di modul ini sama
sekali (murni fungsi statistik atas angka yang sudah ada).
"""


def cliffs_delta(x: list[float], y: list[float]) -> tuple[float | None, str | None]:
    """Cliff's delta membandingkan distribusi x vs y (delta positif berarti
    x cenderung LEBIH BESAR dari y). Kompleksitas O(n_x * n_y) -- aman utk
    n<=384 (skala thesis ini), TIDAK dioptimasi utk n besar.

    Pita interpretasi (Romano dkk. 2006, dipakai APA ADANYA sesuai
    permintaan): |delta| < 0.147 negligible, < 0.33 small, < 0.474 medium,
    selain itu large."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None, None
    gt = 0
    lt = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                gt += 1
            elif xi < yj:
                lt += 1
    delta = (gt - lt) / (nx * ny)
    ad = abs(delta)
    if ad < 0.147:
        band = "negligible"
    elif ad < 0.33:
        band = "small"
    elif ad < 0.474:
        band = "medium"
    else:
        band = "large"
    return delta, band


def holm_correction(p_values: list[float]) -> list[float]:
    """Koreksi Holm-Bonferroni (step-down), family-wise error rate control
    yang LEBIH KUAT (less conservative) daripada Bonferroni polos tapi
    tetap valid tanpa asumsi independensi antar tes. Return p-value
    teradjust dalam URUTAN YANG SAMA dengan input (bukan urutan terurut) --
    caller tidak perlu un-sort manual. p-value ke-i teradjust dijamin
    monoton tidak turun terhadap urutan rank (step-down enforcement) dan
    dibatasi maksimum 1.0."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        adj = min((n - rank) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted
