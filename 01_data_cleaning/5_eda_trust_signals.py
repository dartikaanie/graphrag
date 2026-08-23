"""
5_eda_trust_signals.py
========================
Fase 0.5 (opsional, jalankan setelah Fase 0.2 selesai / Parquet ter-merge) --
Exploratory Data Analysis atas dataset SORD, menghasilkan 19 chart yang
dikelompokkan 5 kategori:

  A. Kualitas & kelayakan data           (A1-A4)
  B. Karakteristik konten & topik        (B5-B7)
  C. Sinyal kepercayaan komunitas        (C8-C13)  <- paling relevan ke novelty tesis
  D. Dimensi waktu                       (D14-D15)
  E. Kelayakan sebagai basis Knowledge Graph (E16-E19)

PENDEKATAN TEKNIS
-------------------
Karena beberapa file berukuran puluhan juta baris (FilteredVotes ~50jt,
FilteredBadges ~35jt), SEMUA hitungan EXACT (jumlah, proporsi, agregasi per
kategori) dilakukan via DuckDB SQL langsung ke file Parquet/CSV -- TIDAK
di-load penuh ke pandas. Hanya chart DISTRIBUSI (histogram/boxplot skor,
reputasi, dst) yang memakai SAMPLING (default 5000 baris, via
`USING SAMPLE ... (reservoir, seed)` supaya representatif & reprodusibel)
karena histogram tidak butuh data penuh untuk terlihat bentuknya.

SUMBER DATA
------------
Wajib (dari Fase 0.2 -- merge Parquet):
    questions_raw_union.parquet, answers_raw_union.parquet
Opsional (dilewati dengan warning kalau tidak ada -- TIDAK menghentikan script):
    comments_raw_union.parquet
    00_datasource/raw/additional-metadata/FilteredUsers.csv
    00_datasource/raw/additional-metadata/FilteredVotes.csv
    00_datasource/raw/additional-metadata/FilteredBadges.csv
    00_datasource/raw/additional-metadata/FilteredTags.csv

LOKASI OUTPUT
-------------
    <repo_root>/report/5_eda_report.md
    <repo_root>/report/img/eda_*.png
Kalau report sudah ada: LANGSUNG TAMPILKAN laporan lama, tidak proses ulang
(sama seperti 1_preview_files.py & 2_check_data_integrity.py). Pakai --force
untuk regenerate.

CARA PAKAI
----------
    python 5_eda_trust_signals.py
    python 5_eda_trust_signals.py --groups C,D          # hanya grup tertentu
    python 5_eda_trust_signals.py --sample-size 10000    # sample lebih besar utk histogram
    python 5_eda_trust_signals.py --skip-heavy           # lewati FilteredVotes/FilteredBadges (file sangat besar)
    python 5_eda_trust_signals.py --force

INSTALL DEPENDENCY
-------------------
    pip install duckdb pandas matplotlib tiktoken
"""

import argparse
import sys
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

CHATGPT_RELEASE_DATE = "2022-11-30"


# ---------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------

def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, img_dir: Path, name: str) -> str:
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(img_dir / name, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


class ChartResult:
    """Hasil satu chart: sukses (fname + catatan) atau dilewati (alasan)."""
    def __init__(self, chart_id, title, fname=None, note=None, skipped_reason=None):
        self.chart_id = chart_id
        self.title = title
        self.fname = fname
        self.note = note
        self.skipped_reason = skipped_reason


def run_chart(chart_id, title, fn, results: list):
    """Bungkus tiap fungsi chart supaya kalau gagal (file tidak ada, kolom
    tidak ada, dll), script TETAP LANJUT ke chart berikutnya -- tidak crash
    total gara-gara satu file opsional hilang."""
    print(f"  [{chart_id}] {title} ...")
    try:
        fname, note = fn()
        if fname is None:
            print(f"      [skip] {note}")
            results.append(ChartResult(chart_id, title, skipped_reason=note))
        else:
            print(f"      [ok] -> {fname}")
            results.append(ChartResult(chart_id, title, fname=fname, note=note))
    except Exception as e:
        print(f"      [skip] error: {e}")
        results.append(ChartResult(chart_id, title, skipped_reason=f"error: {e}"))


# ---------------------------------------------------------------------
# GRUP A -- Kualitas & kelayakan data
# ---------------------------------------------------------------------

def chart_a1(con, paths, img_dir, plt):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    df = con.execute(f"""
        SELECT CASE WHEN AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
                    THEN 'Punya Accepted Answer' ELSE 'Tidak Punya' END AS grp,
               COUNT(*) AS n
        FROM read_parquet('{paths["questions"]}')
        GROUP BY grp
    """).df()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(df["grp"], df["n"], color=["#55A868", "#C44E52"])
    for i, v in enumerate(df["n"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("A1. Pertanyaan dengan vs tanpa Accepted Answer")
    fname = save_fig(fig, img_dir, "eda_a1_accepted_vs_not.png")
    total = int(df["n"].sum())
    pct = df.loc[df["grp"] == "Punya Accepted Answer", "n"].sum() / total * 100
    return fname, f"{pct:.1f}% dari {total:,} pertanyaan punya accepted answer."


def chart_a2(con, paths, img_dir, plt):
    if not paths["questions"].exists() or not paths["answers"].exists():
        return None, "questions/answers parquet tidak lengkap"
    df = con.execute(f"""
        WITH q AS (
            SELECT AcceptedAnswerId FROM read_parquet('{paths["questions"]}')
            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0
        )
        SELECT
            (SELECT COUNT(*) FROM q) AS total_has_accepted_id,
            (SELECT COUNT(*) FROM q WHERE AcceptedAnswerId IN
                (SELECT DISTINCT Id FROM read_parquet('{paths["answers"]}'))) AS matched
    """).df()
    total = int(df["total_has_accepted_id"][0])
    matched = int(df["matched"][0])
    not_matched = total - matched
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Matched ke Answers", "Tidak Matched"], [matched, not_matched],
           color=["#4C72B0", "#C44E52"])
    for i, v in enumerate([matched, not_matched]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("A2. Rasio Match AcceptedAnswerId -> Answers Parquet")
    fname = save_fig(fig, img_dir, "eda_a2_match_rate.png")
    pct = matched / total * 100 if total else 0
    return fname, f"Match rate: {pct:.1f}% ({matched:,} dari {total:,})."


def chart_a3(con, paths, img_dir, plt):
    available = {k: p for k, p in [("Questions", paths["questions"]),
                                     ("Answers", paths["answers"]),
                                     ("Comments", paths["comments"])] if p.exists()}
    if not available:
        return None, "tidak ada parquet dengan kolom match_source ditemukan"
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, available.items()):
        try:
            df = con.execute(f"""
                SELECT match_source, COUNT(*) AS n
                FROM read_parquet('{path}') GROUP BY match_source ORDER BY n DESC
            """).df()
        except Exception:
            ax.set_title(f"{label} (kolom match_source tidak ada)")
            continue
        ax.bar(df["match_source"].astype(str), df["n"], color="#55A868")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("A3. Distribusi match_source (Contain vs LikeMinusContain)")
    fname = save_fig(fig, img_dir, "eda_a3_match_source.png")
    return fname, "Proporsi baris asal Contain vs LikeMinusContain per kategori file."


def chart_a4(con, paths, img_dir, plt, sample_size):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    try:
        import tiktoken
    except ImportError:
        return None, "tiktoken tidak terinstall (pip install tiktoken)"
    df = con.execute(f"""
        SELECT Title, Body, Tags FROM read_parquet('{paths["questions"]}')
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()
    enc = tiktoken.get_encoding("cl100k_base")
    n_tokens = df.apply(lambda r: len(enc.encode(f"{r['Title']} {r['Body']} {r['Tags']}")), axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_tokens, bins=50, color="#4C72B0")
    ax.axvline(2048, color="#C44E52", linestyle="--", label="Batas 2048 token (Kondisi A/B)")
    ax.set_xlabel("Jumlah token (title+body+tags)")
    ax.set_ylabel("Jumlah pertanyaan (sample)")
    ax.set_title(f"A4. Distribusi Panjang Token Pertanyaan (n={sample_size})")
    ax.legend()
    fname = save_fig(fig, img_dir, "eda_a4_token_length.png")
    pct_excluded = (n_tokens > 2048).mean() * 100
    return fname, f"Estimasi dari sample: ~{pct_excluded:.1f}% pertanyaan akan ter-eksklusi filter 2048 token."


# ---------------------------------------------------------------------
# GRUP B -- Karakteristik konten & topik
# ---------------------------------------------------------------------

def chart_b5(con, paths, img_dir, plt):
    if not paths["tags"].exists():
        return None, "FilteredTags.csv tidak ditemukan"
    df = con.execute(f"""
        SELECT Tag, Count FROM read_csv_auto('{paths["tags"]}', SAMPLE_SIZE=-1)
        ORDER BY Count DESC LIMIT 20
    """).df()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["Tag"], df["Count"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Frekuensi kemunculan")
    ax.set_title("B5. Top-20 Tag Paling Sering Muncul")
    fname = save_fig(fig, img_dir, "eda_b5_top_tags.png")
    return fname, f"Tag terbanyak: {df.iloc[0]['Tag']} ({int(df.iloc[0]['Count']):,}x)."


def chart_b6(con, paths, img_dir, plt):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    df = con.execute(f"""
        SELECT (LENGTH(Tags) - LENGTH(REPLACE(Tags, '<', ''))) AS n_tags, COUNT(*) AS n
        FROM read_parquet('{paths["questions"]}')
        WHERE Tags IS NOT NULL
        GROUP BY n_tags ORDER BY n_tags
    """).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["n_tags"], df["n"], color="#55A868")
    ax.set_xlabel("Jumlah tag per pertanyaan")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("B6. Distribusi Jumlah Tag per Pertanyaan (populasi penuh)")
    fname = save_fig(fig, img_dir, "eda_b6_tags_per_question.png")
    return fname, None


def chart_b7(con, paths, img_dir, plt, sample_size):
    if not paths["questions"].exists() or not paths["answers"].exists():
        return None, "questions/answers parquet tidak lengkap"
    q_len = con.execute(f"""
        SELECT LENGTH(Body) AS n FROM read_parquet('{paths["questions"]}')
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()["n"]
    a_len = con.execute(f"""
        SELECT LENGTH(Body) AS n FROM read_parquet('{paths["answers"]}')
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()["n"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([q_len.dropna(), a_len.dropna()], labels=["Pertanyaan", "Jawaban"], showfliers=False)
    ax.set_ylabel("Panjang Body (karakter)")
    ax.set_title(f"B7. Panjang Body: Pertanyaan vs Jawaban (sample n={sample_size})")
    fname = save_fig(fig, img_dir, "eda_b7_body_length.png")
    return fname, "Outlier ekstrem disembunyikan (showfliers=False) supaya skala tetap terbaca."


# ---------------------------------------------------------------------
# GRUP C -- Sinyal kepercayaan komunitas
# ---------------------------------------------------------------------

def chart_c8(con, paths, img_dir, plt, sample_size):
    if not paths["questions"].exists() or not paths["answers"].exists():
        return None, "questions/answers parquet tidak lengkap"
    q_score = con.execute(f"""
        SELECT Score FROM read_parquet('{paths["questions"]}')
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()["Score"]
    a_score = con.execute(f"""
        SELECT Score FROM read_parquet('{paths["answers"]}')
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()["Score"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(q_score.dropna(), bins=40, alpha=0.6, label="Pertanyaan", color="#4C72B0")
    ax.hist(a_score.dropna(), bins=40, alpha=0.6, label="Jawaban", color="#55A868")
    ax.set_xlabel("Score (upvote - downvote)")
    ax.set_ylabel("Jumlah (sample)")
    ax.set_title(f"C8. Distribusi Score: Pertanyaan vs Jawaban (n={sample_size})")
    ax.legend()
    fname = save_fig(fig, img_dir, "eda_c8_score_dist.png")
    return fname, None


def chart_c9(con, paths, img_dir, plt, sample_size):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    df = con.execute(f"""
        SELECT ViewCount FROM read_parquet('{paths["questions"]}')
        WHERE ViewCount IS NOT NULL AND ViewCount > 0
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["ViewCount"], bins=50, color="#DD8452")
    ax.set_xscale("log")
    ax.set_xlabel("ViewCount (log scale)")
    ax.set_ylabel("Jumlah pertanyaan (sample)")
    ax.set_title(f"C9. Distribusi ViewCount Pertanyaan (n={sample_size})")
    fname = save_fig(fig, img_dir, "eda_c9_viewcount.png")
    return fname, None


def chart_c10(con, paths, img_dir, plt, sample_size):
    if not paths["questions"].exists() or not paths["answers"].exists():
        return None, "questions/answers parquet tidak lengkap"
    accepted = con.execute(f"""
        SELECT a.Score AS score
        FROM read_parquet('{paths["answers"]}') a
        JOIN read_parquet('{paths["questions"]}') q ON a.Id = q.AcceptedAnswerId
    """).df()["score"]
    non_accepted = con.execute(f"""
        SELECT a.Score AS score
        FROM read_parquet('{paths["answers"]}') a
        WHERE a.Id NOT IN (SELECT AcceptedAnswerId FROM read_parquet('{paths["questions"]}')
                            WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0)
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()["score"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([accepted.dropna(), non_accepted.dropna()],
               labels=[f"Accepted (n={len(accepted):,})", f"Non-accepted (sample n={len(non_accepted):,})"],
               showfliers=False)
    ax.set_ylabel("Score jawaban")
    ax.set_title("C10. Score: Accepted vs Non-Accepted Answer")
    fname = save_fig(fig, img_dir, "eda_c10_accepted_vs_nonaccepted_score.png")
    note = (f"Median accepted={accepted.median():.1f} vs non-accepted={non_accepted.median():.1f} -- "
            f"validasi awal apakah accepted answer memang cenderung Score lebih tinggi.")
    return fname, note


def chart_c11(con, paths, img_dir, plt, sample_size):
    if not paths["users"].exists():
        return None, "FilteredUsers.csv tidak ditemukan"
    df = con.execute(f"""
        SELECT Reputation FROM read_csv_auto('{paths["users"]}', SAMPLE_SIZE=-1, ignore_errors=true)
        WHERE Reputation IS NOT NULL AND Reputation > 0
        USING SAMPLE {sample_size} ROWS (reservoir, 42)
    """).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["Reputation"], bins=50, color="#8172B2")
    ax.set_xscale("log")
    ax.set_xlabel("Reputasi (log scale)")
    ax.set_ylabel("Jumlah user (sample)")
    ax.set_title(f"C11. Distribusi Reputasi Kontributor (n={sample_size})")
    fname = save_fig(fig, img_dir, "eda_c11_reputation.png")
    return fname, None


VOTE_TYPE_LABELS = {
    1: "AcceptedByOriginator", 2: "UpMod", 3: "DownMod", 4: "Offensive",
    5: "Favorite", 6: "Close", 7: "Reopen", 8: "BountyStart", 9: "BountyClose",
    10: "Deletion", 11: "Undeletion", 12: "Spam", 15: "ModeratorReview", 16: "Approve",
}


def chart_c12(con, paths, img_dir, plt):
    if not paths["votes"].exists():
        return None, "FilteredVotes.csv tidak ditemukan (atau dilewati via --skip-heavy)"
    df = con.execute(f"""
        SELECT VoteTypeId, COUNT(*) AS n
        FROM read_csv_auto('{paths["votes"]}', SAMPLE_SIZE=-1, ignore_errors=true)
        GROUP BY VoteTypeId ORDER BY n DESC LIMIT 10
    """).df()
    df["label"] = df["VoteTypeId"].map(lambda x: VOTE_TYPE_LABELS.get(int(x), f"Type {int(x)}"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["label"], df["n"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Jumlah vote (populasi penuh)")
    ax.set_title("C12. Distribusi Jenis Vote (Top 10)")
    fname = save_fig(fig, img_dir, "eda_c12_vote_types.png")
    return fname, "Dihitung dari populasi penuh FilteredVotes.csv (agregasi SQL, bukan sample)."


def chart_c13(con, paths, img_dir, plt):
    if not paths["badges"].exists():
        return None, "FilteredBadges.csv tidak ditemukan (atau dilewati via --skip-heavy)"
    df = con.execute(f"""
        SELECT n_badges, COUNT(*) AS n_users FROM (
            SELECT UserId, COUNT(*) AS n_badges
            FROM read_csv_auto('{paths["badges"]}', SAMPLE_SIZE=-1, ignore_errors=true)
            GROUP BY UserId
        )
        GROUP BY n_badges ORDER BY n_badges
    """).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["n_badges"], df["n_users"], color="#C44E52")
    ax.set_xlabel("Jumlah badge per user")
    ax.set_ylabel("Jumlah user")
    ax.set_title("C13. Distribusi Jumlah Badge per User (populasi penuh)")
    ax.set_xlim(0, min(50, df["n_badges"].max()))
    fname = save_fig(fig, img_dir, "eda_c13_badges_per_user.png")
    return fname, "Sumbu-x dibatasi sampai 50 badge/user supaya ekor panjang tidak mendominasi chart."


# ---------------------------------------------------------------------
# GRUP D -- Dimensi waktu
# ---------------------------------------------------------------------

def chart_d14(con, paths, img_dir, plt):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    df = con.execute(f"""
        SELECT date_trunc('month', CreationDate) AS month, COUNT(*) AS n
        FROM read_parquet('{paths["questions"]}')
        GROUP BY month ORDER BY month
    """).df()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["month"], df["n"], color="#4C72B0")
    ax.axvline(pd_to_datetime(CHATGPT_RELEASE_DATE), color="#C44E52", linestyle="--",
               label="Rilis ChatGPT (30 Nov 2022)")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("D14. Tren Jumlah Pertanyaan per Bulan (populasi penuh)")
    ax.legend()
    fname = save_fig(fig, img_dir, "eda_d14_questions_trend.png")
    return fname, None


def chart_d15(con, paths, img_dir, plt):
    available = {k: p for k, p in [("Pertanyaan", paths["questions"]),
                                     ("Jawaban", paths["answers"]),
                                     ("Comment", paths["comments"])] if p.exists()}
    if not available:
        return None, "tidak ada parquet ditemukan"
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"Pertanyaan": "#4C72B0", "Jawaban": "#55A868", "Comment": "#DD8452"}
    for label, path in available.items():
        df = con.execute(f"""
            SELECT date_trunc('month', CreationDate) AS month, COUNT(*) AS n
            FROM read_parquet('{path}') GROUP BY month ORDER BY month
        """).df()
        ax.plot(df["month"], df["n"], label=label, color=colors.get(label))
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Jumlah baris")
    ax.set_title("D15. Tren Waktu: Pertanyaan vs Jawaban vs Comment")
    ax.legend()
    fname = save_fig(fig, img_dir, "eda_d15_trend_comparison.png")
    return fname, "Cek apakah data historis representatif / tidak bias ke periode tertentu."


def pd_to_datetime(s):
    import pandas as pd
    return pd.to_datetime(s)


# ---------------------------------------------------------------------
# GRUP E -- Kelayakan sebagai basis Knowledge Graph
# ---------------------------------------------------------------------

def chart_e16(con, paths, img_dir, plt):
    available = {k: p for k, p in [("Pertanyaan", paths["questions"]), ("Jawaban", paths["answers"])]
                 if p.exists()}
    if not available:
        return None, "questions/answers parquet tidak ditemukan"
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, available.items()):
        try:
            df = con.execute(f"""
                SELECT CommentCount, COUNT(*) AS n FROM read_parquet('{path}')
                WHERE CommentCount IS NOT NULL GROUP BY CommentCount ORDER BY CommentCount
            """).df()
        except Exception:
            ax.set_title(f"{label} (kolom CommentCount tidak ada)")
            continue
        ax.bar(df["CommentCount"], df["n"], color="#8172B2")
        ax.set_xlim(0, min(20, df["CommentCount"].max()))
        ax.set_title(f"{label}")
        ax.set_xlabel("Jumlah comment")
    fig.suptitle("E16. Distribusi Jumlah Comment per Post (populasi penuh)")
    fname = save_fig(fig, img_dir, "eda_e16_comment_count.png")
    return fname, "Proxy edge density untuk relasi Comment di Knowledge Graph."


def chart_e17(con, paths, img_dir, plt):
    if not paths["questions"].exists():
        return None, "questions_raw_union.parquet tidak ditemukan"
    df = con.execute(f"""
        SELECT AnswerCount, COUNT(*) AS n FROM read_parquet('{paths["questions"]}')
        WHERE AnswerCount IS NOT NULL GROUP BY AnswerCount ORDER BY AnswerCount
    """).df()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["AnswerCount"], df["n"], color="#4C72B0")
    ax.set_xlim(0, min(20, df["AnswerCount"].max()))
    ax.set_xlabel("Jumlah jawaban per pertanyaan")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("E17. Distribusi Jumlah Jawaban per Pertanyaan (populasi penuh)")
    fname = save_fig(fig, img_dir, "eda_e17_answer_count.png")
    return fname, "Proxy derajat simpul (node degree) Question->Answer di KG."


def chart_e18(con, paths, img_dir, plt):
    if not paths["questions"].exists() or not paths["answers"].exists():
        return None, "questions/answers parquet tidak lengkap"
    df = con.execute(f"""
        WITH q AS (SELECT AcceptedAnswerId FROM read_parquet('{paths["questions"]}'))
        SELECT
            (SELECT COUNT(*) FROM q) AS total,
            (SELECT COUNT(*) FROM q WHERE AcceptedAnswerId IS NOT NULL AND AcceptedAnswerId != 0) AS has_id,
            (SELECT COUNT(*) FROM q WHERE AcceptedAnswerId IN
                (SELECT DISTINCT Id FROM read_parquet('{paths["answers"]}'))) AS matched
    """).df()
    total, has_id, matched = int(df["total"][0]), int(df["has_id"][0]), int(df["matched"][0])
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Total Populasi\nPertanyaan", "Punya\nAcceptedAnswerId", "Matched ke\nAnswers Parquet"]
    values = [total, has_id, matched]
    ax.bar(labels, values, color=["#94A3B8", "#4C72B0", "#55A868"])
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_title("E18. Ukuran Subset Matched vs Total Populasi")
    fname = save_fig(fig, img_dir, "eda_e18_matched_vs_total.png")
    return fname, f"Subset matched = {matched/total*100:.1f}% dari total populasi pertanyaan."


def chart_e19(con, paths, img_dir, plt):
    if not paths["tags"].exists():
        return None, "FilteredTags.csv tidak ditemukan"
    df = con.execute(f"""
        SELECT COUNT(DISTINCT Tag) AS n FROM read_csv_auto('{paths["tags"]}', SAMPLE_SIZE=-1)
    """).df()
    n = int(df["n"][0])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Tag Unik"], [n], color="#DD8452")
    ax.text(0, n, f"{n:,}", ha="center", va="bottom")
    ax.set_title("E19. Jumlah Entitas Tag Unik")
    fname = save_fig(fig, img_dir, "eda_e19_unique_tags.png")
    return fname, f"{n:,} tag unik -- estimasi kasar ukuran node kategori Tag di KG."


# ---------------------------------------------------------------------
# Report existing check
# ---------------------------------------------------------------------

def show_existing_report(output_path: Path):
    print(f"\n[i] Laporan EDA sudah ada -> {output_path.resolve()}")
    print("[i] Menampilkan laporan yang ada (tidak proses ulang). "
          "Pakai --force untuk regenerate.\n")
    print("-" * 78)
    print(output_path.read_text(encoding="utf-8"))
    print("-" * 78)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

GROUP_CHARTS = {
    "A": ["a1", "a2", "a3", "a4"],
    "B": ["b5", "b6", "b7"],
    "C": ["c8", "c9", "c10", "c11", "c12", "c13"],
    "D": ["d14", "d15"],
    "E": ["e16", "e17", "e18", "e19"],
}

GROUP_TITLES = {
    "A": "Kualitas & Kelayakan Data",
    "B": "Karakteristik Konten & Topik",
    "C": "Sinyal Kepercayaan Komunitas",
    "D": "Dimensi Waktu",
    "E": "Kelayakan sebagai Basis Knowledge Graph",
}

# ---------------------------------------------------------------------
# Metadata tiap chart: sumber data, tujuan analisis, kegunaan di tesis.
# Dipakai untuk membangun tabel ringkasan di awal laporan.
# ---------------------------------------------------------------------
CHART_META = {
    "a1": {
        "sumber": "questions_raw_union.parquet (AcceptedAnswerId)",
        "tujuan": "Mengukur proporsi pertanyaan yang punya ground truth (accepted answer).",
        "kegunaan": "Dasar kuantitatif kriteria inklusi Kondisi A/B/C (Bab III) & ukuran populasi awal sebelum sampling.",
    },
    "a2": {
        "sumber": "questions + answers parquet (join AcceptedAnswerId -> Answers.Id)",
        "tujuan": "Validasi rasio match setelah fix bug semi-join (filter sebelum sampling).",
        "kegunaan": "Bukti empiris justifikasi keputusan implementasi pipeline -- masuk lampiran/catatan metodologi.",
    },
    "a3": {
        "sumber": "kolom match_source di questions/answers/comments parquet",
        "tujuan": "Transparansi proporsi baris asal Contain vs LikeMinusContain.",
        "kegunaan": "Mendukung deskripsi karakteristik sumber data di Bab III.",
    },
    "a4": {
        "sumber": "sample Title+Body+Tags questions (token count via tiktoken)",
        "tujuan": "Estimasi persentase pertanyaan yang tereksklusi filter 2048 token.",
        "kegunaan": "Justifikasi kuantitatif kriteria eksklusi replikasi Da Silva dkk. (2025).",
    },
    "b5": {
        "sumber": "FilteredTags.csv (Tag, Count)",
        "tujuan": "Identifikasi topik/domain teknologi paling dominan.",
        "kegunaan": "Dasar diskusi keragaman domain KG & pemilihan contoh kasus (mis. subgraf FastAPI) di Bab III.",
    },
    "b6": {
        "sumber": "kolom Tags questions_raw_union.parquet",
        "tujuan": "Mengukur tag concordance (keragaman anotasi topik per pertanyaan).",
        "kegunaan": "Mendukung desain simpul Tag & relasi taggedWith di KG (Bab III.4.2).",
    },
    "b7": {
        "sumber": "sample panjang Body questions vs answers",
        "tujuan": "Membandingkan verbosity pertanyaan vs jawaban.",
        "kegunaan": "Pertimbangan desain chunking/token limit retrieval Kondisi B & C.",
    },
    "c8": {
        "sumber": "sample Score questions vs answers",
        "tujuan": "Melihat sebaran nilai vote komunitas pada pertanyaan vs jawaban.",
        "kegunaan": "Dasar empiris desain bobot berbasis Score di Tabel II.2/III.6.",
    },
    "c9": {
        "sumber": "sample ViewCount questions",
        "tujuan": "Melihat sebaran popularitas pertanyaan.",
        "kegunaan": "Dasar sinyal relevansi jangka panjang sebagai bobot tambahan KG.",
    },
    "c10": {
        "sumber": "Score jawaban accepted (populasi penuh) vs non-accepted (sample)",
        "tujuan": "Validasi asumsi 'accepted answer = kualitas lebih tinggi'.",
        "kegunaan": "Justifikasi langsung bobot w=1.0 pada edge hasAcceptedAnswer (Tabel III.6).",
    },
    "c11": {
        "sumber": "FilteredUsers.csv (Reputation)",
        "tujuan": "Melihat sebaran reputasi kontributor.",
        "kegunaan": "Dasar normalisasi bobot edge authorTrust (Tabel III.6).",
    },
    "c12": {
        "sumber": "FilteredVotes.csv (VoteTypeId, populasi penuh)",
        "tujuan": "Identifikasi jenis vote paling dominan (upvote/downvote/accepted/dst).",
        "kegunaan": "Validasi bahwa upvote/downvote representatif sebagai komponen pembentuk Score.",
    },
    "c13": {
        "sumber": "FilteredBadges.csv (agregasi per UserId, populasi penuh)",
        "tujuan": "Melihat sebaran jumlah badge per kontributor.",
        "kegunaan": "Sinyal trust tambahan potensial di luar 5 sinyal yang sudah ada di Tabel II.2.",
    },
    "d14": {
        "sumber": "CreationDate questions_raw_union.parquet (populasi penuh)",
        "tujuan": "Melihat tren volume pertanyaan dari waktu ke waktu.",
        "kegunaan": "Kontekstualisasi temuan Da Silva dkk. (2025) soal penurunan aktivitas SO pasca ChatGPT, pada subset SORD penelitian ini.",
    },
    "d15": {
        "sumber": "CreationDate questions/answers/comments (populasi penuh)",
        "tujuan": "Cek representativitas historis data gabungan antar kategori.",
        "kegunaan": "Mendukung argumen validitas temporal dataset di bagian limitasi/scope.",
    },
    "e16": {
        "sumber": "CommentCount questions & answers (populasi penuh)",
        "tujuan": "Estimasi densitas relasi Comment.",
        "kegunaan": "Estimasi kompleksitas graf (jumlah edge potensial) sebelum konstruksi KG riil di Neo4j.",
    },
    "e17": {
        "sumber": "AnswerCount questions_raw_union.parquet (populasi penuh)",
        "tujuan": "Estimasi derajat simpul (node degree) Question->Answer.",
        "kegunaan": "Input estimasi ukuran/densitas KG sebelum konstruksi riil.",
    },
    "e18": {
        "sumber": "gabungan total populasi vs subset matched (populasi penuh)",
        "tujuan": "Kuantifikasi ukuran subset layak dijadikan basis KG (~271K).",
        "kegunaan": "Dasar angka yang dipakai di bagian limitasi/scope (framing 'recommendation-oriented Q&A').",
    },
    "e19": {
        "sumber": "FilteredTags.csv (COUNT DISTINCT Tag)",
        "tujuan": "Estimasi jumlah entitas unik kategori Tag.",
        "kegunaan": "Estimasi awal ukuran KG (jumlah node Tag) untuk perencanaan kapasitas Neo4j.",
    },
}


def build_summary_table(all_results: dict) -> list[str]:
    """Tabel ringkasan di awal laporan: tiap chart -> sumber data, tujuan,
    kegunaan di tesis, dan status (berhasil/dilewati). Menjawab langsung
    pertanyaan 'tabel/chart ini berisi data apa, untuk apa, dipakai untuk
    apa' tanpa perlu buka tiap section satu-satu."""
    lines = [
        "## Ringkasan: Chart Ini Berisi Apa, Untuk Apa, Dipakai Untuk Apa",
        "",
        "| ID | Chart | Sumber Data | Tujuan Analisis | Kegunaan di Tesis | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for g, results in all_results.items():
        for r in results:
            cid = r.chart_id.lower()
            meta = CHART_META.get(cid, {})
            status = "OK" if r.fname else "Dilewati"
            lines.append(
                f"| {r.chart_id} | {r.title} | {meta.get('sumber','-')} | "
                f"{meta.get('tujuan','-')} | {meta.get('kegunaan','-')} | {status} |"
            )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged-dir", default="../00_datasource/merged",
                         help="Folder Parquet hasil merge Fase 0.2")
    parser.add_argument("--raw-dir", default="../00_datasource/raw",
                         help="Folder CSV mentah SORD (untuk FilteredUsers/Votes/Badges/Tags)")
    parser.add_argument("--groups", default="A,B,C,D,E",
                         help="Grup chart yang dijalankan, pisah koma (default semua: A,B,C,D,E)")
    parser.add_argument("--sample-size", type=int, default=5000,
                         help="Ukuran sample untuk chart distribusi (histogram/boxplot)")
    parser.add_argument("--skip-heavy", action="store_true",
                         help="Lewati FilteredVotes.csv & FilteredBadges.csv (file sangat besar)")
    parser.add_argument("--output", default=None,
                         help="Path file Markdown output. Default: <repo_root>/report/5_eda_report.md")
    parser.add_argument("--img-dir", default=None,
                         help="Folder chart .png. Default: <repo_root>/report/img")
    parser.add_argument("--force", action="store_true",
                         help="Regenerate laporan walau sudah ada")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (REPO_ROOT / "report" / "5_eda_report.md")
    img_dir = Path(args.img_dir) if args.img_dir else (REPO_ROOT / "report" / "img")

    if output_path.exists() and not args.force:
        show_existing_report(output_path)
        return

    merged_dir = Path(args.merged_dir)
    raw_dir = Path(args.raw_dir)

    paths = {
        "questions": merged_dir / "questions_raw_union.parquet",
        "answers": merged_dir / "answers_raw_union.parquet",
        "comments": merged_dir / "comments_raw_union.parquet",
        "users": raw_dir / "additional-metadata" / "FilteredUsers.csv",
        "votes": raw_dir / "additional-metadata" / "FilteredVotes.csv",
        "badges": raw_dir / "additional-metadata" / "FilteredBadges.csv",
        "tags": raw_dir / "additional-metadata" / "FilteredTags.csv",
    }
    if args.skip_heavy:
        paths["votes"] = Path("__skip_heavy__")
        paths["badges"] = Path("__skip_heavy__")

    plt = get_plt()
    con = duckdb.connect()
    sample_size = args.sample_size
    groups = [g.strip().upper() for g in args.groups.split(",")]

    fn_map = {
        "a1": lambda: chart_a1(con, paths, img_dir, plt),
        "a2": lambda: chart_a2(con, paths, img_dir, plt),
        "a3": lambda: chart_a3(con, paths, img_dir, plt),
        "a4": lambda: chart_a4(con, paths, img_dir, plt, sample_size),
        "b5": lambda: chart_b5(con, paths, img_dir, plt),
        "b6": lambda: chart_b6(con, paths, img_dir, plt),
        "b7": lambda: chart_b7(con, paths, img_dir, plt, sample_size),
        "c8": lambda: chart_c8(con, paths, img_dir, plt, sample_size),
        "c9": lambda: chart_c9(con, paths, img_dir, plt, sample_size),
        "c10": lambda: chart_c10(con, paths, img_dir, plt, sample_size),
        "c11": lambda: chart_c11(con, paths, img_dir, plt, sample_size),
        "c12": lambda: chart_c12(con, paths, img_dir, plt),
        "c13": lambda: chart_c13(con, paths, img_dir, plt),
        "d14": lambda: chart_d14(con, paths, img_dir, plt),
        "d15": lambda: chart_d15(con, paths, img_dir, plt),
        "e16": lambda: chart_e16(con, paths, img_dir, plt),
        "e17": lambda: chart_e17(con, paths, img_dir, plt),
        "e18": lambda: chart_e18(con, paths, img_dir, plt),
        "e19": lambda: chart_e19(con, paths, img_dir, plt),
    }
    title_map = {
        "a1": "Pertanyaan dgn vs tanpa Accepted Answer", "a2": "Rasio Match Question->Answer",
        "a3": "Distribusi match_source", "a4": "Distribusi Panjang Token Pertanyaan",
        "b5": "Top-20 Tag", "b6": "Jumlah Tag per Pertanyaan", "b7": "Panjang Body Q vs A",
        "c8": "Distribusi Score Q vs A", "c9": "Distribusi ViewCount",
        "c10": "Score Accepted vs Non-Accepted", "c11": "Distribusi Reputasi Kontributor",
        "c12": "Distribusi Jenis Vote", "c13": "Distribusi Badge per User",
        "d14": "Tren Pertanyaan per Bulan", "d15": "Tren Waktu Q vs A vs Comment",
        "e16": "Jumlah Comment per Post", "e17": "Jumlah Jawaban per Pertanyaan",
        "e18": "Subset Matched vs Total Populasi", "e19": "Jumlah Tag Unik",
    }

    print(f"Menjalankan EDA untuk grup: {', '.join(groups)}")
    all_results = {}
    for g in groups:
        if g not in GROUP_CHARTS:
            print(f"[warn] grup '{g}' tidak dikenal, dilewati")
            continue
        print(f"\n=== GRUP {g}: {GROUP_TITLES[g]} ===")
        results = []
        for cid in GROUP_CHARTS[g]:
            run_chart(cid.upper(), title_map[cid], fn_map[cid], results)
        all_results[g] = results

    # --- Bangun laporan markdown ---
    md_parts = ["# Laporan EDA Datasource SORD -- Sinyal Kepercayaan & Karakteristik Data", ""]
    n_ok = sum(1 for rs in all_results.values() for r in rs if r.fname)
    n_skip = sum(1 for rs in all_results.values() for r in rs if not r.fname)
    md_parts.append(f"Total chart berhasil: {n_ok} | dilewati (file/kolom tidak tersedia): {n_skip}")
    md_parts.append("")
    md_parts.extend(build_summary_table(all_results))

    for g, results in all_results.items():
        md_parts.append(f"## Grup {g}. {GROUP_TITLES[g]}")
        md_parts.append("")
        for r in results:
            md_parts.append(f"### {r.chart_id}. {r.title}")
            md_parts.append("")
            if r.fname:
                md_parts.append(f"![{r.chart_id}](img/{r.fname})")
                md_parts.append("")
                if r.note:
                    md_parts.append(f"*{r.note}*")
                    md_parts.append("")
            else:
                md_parts.append(f"_[Dilewati] {r.skipped_reason}_")
                md_parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"\n[OK] Laporan tersimpan -> {output_path.resolve()}")
    print(f"     {n_ok} chart berhasil, {n_skip} dilewati.")


if __name__ == "__main__":
    sys.exit(main())