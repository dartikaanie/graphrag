"""
6_postlinks_to_sord.py
=======================
Enrichment Question-Question edges dari StackExchange PostLinks -- tahap
persiapan Knowledge Graph untuk Kondisi C (GraphRAG). SORD (Fatima &
Maqbool, 2026) hanya menyediakan struktur Question-Answer; script ini
menambahkan TIPE EDGE BARU (Question-Question: Linked / Duplicate) ke
populasi entitas yang SUDAH ADA di SORD, bersumber dari tabel PostLinks
pada StackExchange Data Dump (archive.org).

TIDAK ada entitas baru yang ditambahkan -- PostId dan RelatedPostId
di-inner-join terhadap Question ID yang sudah ada di
questions_raw_union.parquet (kedua sisi harus match), sehingga populasi
tetap konsisten dengan SORD dan valid dipakai berdampingan dengan Kondisi
A/B untuk perbandingan apple-to-apple.

INSTALL DEPENDENCY
-------------------
    pip install duckdb pandas pyarrow python-dotenv

SETUP .env (di root project -- QUESTIONS_PARQUET sudah ada di sana,
dipakai bersama oleh Kondisi A/B, jadi TIDAK perlu didup di sini)
----------------------------------------------------------------------------
    QUESTIONS_PARQUET=../00_datasource/merged/questions_raw_union.parquet

CARA PAKAI
----------
    python 6_postlinks_to_sord.py
    python 6_postlinks_to_sord.py --postlinks-xml ../00_datasource/PostLinks.xml \\
        --output-dir ../00_datasource/merged

OUTPUT
------
    00_datasource/merged/question_links.parquet
        post_id          : int64  -- Id pertanyaan asal (ada di SORD Questions)
        related_post_id  : int64  -- Id pertanyaan terkait (ada di SORD Questions)
        link_type        : int64  -- 1 = Linked, 3 = Duplicate
        link_type_label  : str    -- "Linked" / "Duplicate"
        creation_date    : str    -- tanggal pembuatan link (dari PostLinks asli)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

VALID_LINK_TYPES = {1: "Linked", 3: "Duplicate"}
PROGRESS_EVERY = 500_000  # log progres tiap N baris <row> yang dibaca dari XML

LOG_DIR = Path("logs")


# ---------------------------------------------------------------------
# Logging (file + console), pola sama dengan setup_logging() di
# llm/a_pure_llm/a_baseline_replication.py
# ---------------------------------------------------------------------

def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{ts}_postlinks_to_sord.log"

    logger = logging.getLogger("postlinks_to_sord")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger, log_path


def step(logger: logging.Logger, n: int, title: str) -> None:
    logger.info("=" * 78)
    logger.info(f"STEP {n} - {title}")
    logger.info("=" * 78)


# ---------------------------------------------------------------------
# STEP 1 - Validasi input
# ---------------------------------------------------------------------

def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> dict:
    questions_parquet = Path(args.questions_parquet)
    if not questions_parquet.exists():
        raise FileNotFoundError(
            f"QUESTIONS_PARQUET tidak ditemukan: {questions_parquet}\n"
            f"  Cek QUESTIONS_PARQUET di .env (root project) atau pakai --questions-parquet."
        )

    postlinks_xml = Path(args.postlinks_xml)
    if not postlinks_xml.exists():
        raise FileNotFoundError(
            f"PostLinks.xml tidak ditemukan: {postlinks_xml}\n"
            f"  Pakai --postlinks-xml kalau lokasinya beda dari default."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "questions_parquet": questions_parquet,
        "postlinks_xml": postlinks_xml,
        "output_dir": output_dir,
        "output_path": output_dir / "question_links.parquet",
    }

    logger.info(f"QUESTIONS_PARQUET : {config['questions_parquet']}")
    logger.info(f"PostLinks.xml     : {config['postlinks_xml']} "
                f"({config['postlinks_xml'].stat().st_size / (1024*1024):,.1f} MB)")
    logger.info(f"Output            : {config['output_path']}")
    return config


# ---------------------------------------------------------------------
# STEP 2 - Parsing XML streaming + filter LinkTypeId
# ---------------------------------------------------------------------

def parse_postlinks_streaming(xml_path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Parsing streaming: {xml_path}")
    t0 = time.time()

    total_rows = 0
    kept_rows = 0
    skipped_malformed = 0
    records = []

    # Trik memory-safe untuk stdlib ElementTree: pakai events start+end,
    # simpan referensi root, lalu root.remove(elem) setelah tiap <row>
    # diproses supaya elemen yang sudah diproses tidak menumpuk di memori
    # (relevan untuk constraint RAM terbatas -- pola yang sama dengan
    # alasan streaming COPY di 3_merge_sord_sources.py).
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)

    for event, elem in context:
        if event != "end" or elem.tag != "row":
            continue

        total_rows += 1

        link_type_raw = elem.get("LinkTypeId")
        post_id_raw = elem.get("PostId")
        related_post_id_raw = elem.get("RelatedPostId")

        if link_type_raw is None or post_id_raw is None or related_post_id_raw is None:
            skipped_malformed += 1
            elem.clear()
            root.remove(elem)
            continue

        link_type = int(link_type_raw)
        if link_type in VALID_LINK_TYPES:
            records.append(
                {
                    "post_id": int(post_id_raw),
                    "related_post_id": int(related_post_id_raw),
                    "link_type": link_type,
                    "link_type_label": VALID_LINK_TYPES[link_type],
                    "creation_date": elem.get("CreationDate"),
                }
            )
            kept_rows += 1

        elem.clear()
        root.remove(elem)

        if total_rows % PROGRESS_EVERY == 0:
            logger.info(f"  ... {total_rows:,} baris PostLinks dibaca, {kept_rows:,} lolos filter LinkTypeId")

    elapsed = time.time() - t0
    logger.info(f"Parsing selesai dalam {elapsed:.1f}s")
    logger.info(f"Total baris PostLinks (semua LinkTypeId) : {total_rows:,}")
    if skipped_malformed:
        logger.info(f"Baris dilewati (atribut hilang)          : {skipped_malformed:,}")
    logger.info(f"Setelah filter LinkTypeId in (1, 3)      : {kept_rows:,}")

    return pd.DataFrame.from_records(
        records, columns=["post_id", "related_post_id", "link_type", "link_type_label", "creation_date"]
    )


# ---------------------------------------------------------------------
# STEP 3 - Inner-join terhadap Question IDs SORD
# ---------------------------------------------------------------------

def join_with_sord_questions(
    links_df: pd.DataFrame, questions_parquet: Path, logger: logging.Logger
) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("links_raw", links_df)

    q_parquet = questions_parquet.as_posix()

    # Diagnostik bertahap: berapa banyak sisi PostId saja, RelatedPostId
    # saja, dan kedua-duanya yang match ke populasi Question ID SORD.
    diag = con.execute(
        f"""
        WITH q AS (SELECT DISTINCT Id FROM read_parquet('{q_parquet}'))
        SELECT
            (SELECT COUNT(*) FROM links_raw) AS total_links,
            (SELECT COUNT(*) FROM links_raw l JOIN q ON l.post_id = q.Id) AS post_id_match,
            (SELECT COUNT(*) FROM links_raw l JOIN q ON l.related_post_id = q.Id) AS related_post_id_match,
            (SELECT COUNT(*) FROM links_raw l
                JOIN q q1 ON l.post_id = q1.Id
                JOIN q q2 ON l.related_post_id = q2.Id) AS both_match
        """
    ).fetchdf()

    logger.info(f"Kandidat edge (sebelum join)              : {int(diag['total_links'][0]):,}")
    logger.info(f"  - PostId cocok dengan Question ID SORD   : {int(diag['post_id_match'][0]):,}")
    logger.info(f"  - RelatedPostId cocok Question ID SORD   : {int(diag['related_post_id_match'][0]):,}")
    logger.info(f"  - KEDUANYA cocok (inner-join final)      : {int(diag['both_match'][0]):,}")

    result = con.execute(
        f"""
        WITH q AS (SELECT DISTINCT Id FROM read_parquet('{q_parquet}'))
        SELECT DISTINCT
            l.post_id,
            l.related_post_id,
            l.link_type,
            l.link_type_label,
            l.creation_date
        FROM links_raw l
        INNER JOIN q q1 ON l.post_id = q1.Id
        INNER JOIN q q2 ON l.related_post_id = q2.Id
        ORDER BY l.post_id, l.related_post_id
        """
    ).fetchdf()

    con.close()

    n_before_dedup = int(diag["both_match"][0])
    n_after_dedup = len(result)
    if n_after_dedup < n_before_dedup:
        logger.info(f"Duplikat edge dihapus (DISTINCT)          : {n_before_dedup - n_after_dedup:,}")
    logger.info(f"Jumlah edge Question-Question final       : {n_after_dedup:,}")

    if n_after_dedup > 0:
        breakdown = result["link_type_label"].value_counts()
        for label, count in breakdown.items():
            logger.info(f"  - {label:<10}: {count:,}")

    return result


# ---------------------------------------------------------------------
# STEP 4 - Simpan ke parquet
# ---------------------------------------------------------------------

def save_output(result: pd.DataFrame, output_path: Path, logger: logging.Logger) -> None:
    result = result.astype(
        {
            "post_id": "int64",
            "related_post_id": "int64",
            "link_type": "int64",
            "link_type_label": "string",
        }
    )

    result.to_parquet(output_path, index=False)

    # fsync eksplisit (pola sama dengan Kondisi B) supaya file tersinkron
    # penuh sebelum script selesai -- relevan untuk folder yang tersinkron
    # cloud (iCloud/OneDrive) di macOS.
    with open(output_path, "rb") as f:
        os.fsync(f.fileno())

    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Tersimpan: {output_path} ({size_kb:,.1f} KB, {len(result):,} baris)")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-parquet", default=os.getenv("QUESTIONS_PARQUET"),
                         help="Default dari QUESTIONS_PARQUET di .env (root project)")
    parser.add_argument("--postlinks-xml", default="../00_datasource/PostLinks.xml",
                         help="Path ke PostLinks.xml hasil ekstraksi (default: ../00_datasource/PostLinks.xml)")
    parser.add_argument("--output-dir", default="../00_datasource/merged",
                         help="Folder output Parquet (default: ../00_datasource/merged)")
    args = parser.parse_args()

    if not args.questions_parquet:
        print("[ERROR] QUESTIONS_PARQUET harus diset di .env (root project) atau lewat --questions-parquet")
        sys.exit(1)

    logger, log_path = setup_logging()
    logger.info("Question-Question Edge Enrichment dari PostLinks (persiapan Kondisi C / GraphRAG)")
    logger.info(f"Log file: {log_path}")

    t_start = time.time()

    step(logger, 1, "Validasi input")
    config = validate_inputs(args, logger)

    step(logger, 2, "Parsing XML streaming + filter LinkTypeId in (1=Linked, 3=Duplicate)")
    links_df = parse_postlinks_streaming(config["postlinks_xml"], logger)

    step(logger, 3, "Inner-join PostId & RelatedPostId terhadap Question IDs SORD")
    result = join_with_sord_questions(links_df, config["questions_parquet"], logger)

    if len(result) == 0:
        logger.warning(
            "Hasil join 0 baris. Cek: apakah QUESTIONS_PARQUET yang dipakai konsisten dengan "
            "populasi yang dipakai di Kondisi A/B (sama seperti kasus OVERSAMPLE_POOL mismatch "
            "sebelumnya)?"
        )

    step(logger, 4, "Simpan hasil ke parquet")
    save_output(result, config["output_path"], logger)

    elapsed = time.time() - t_start
    logger.info("=" * 78)
    logger.info(f"SELESAI dalam {elapsed:.1f}s")
    logger.info("=" * 78)


if __name__ == "__main__":
    sys.exit(main())