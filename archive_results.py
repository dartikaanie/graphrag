#!/usr/bin/env python3
"""
archive_results.py
=====================================
Step 5 (docs/Agent prompt grounding factor ui.md, data retention/audit
trail): copies ALL results, logs, manifests, and judge outputs to a
configurable archive directory -- preserves the original folder
structure, writes a SHA256SUMS file (one line per copied file, same
format `sha256sum` itself produces), verifies every checksum AFTER
copying, and NEVER deletes or modifies anything at the source (read-only
on the repo; only ever writes inside --dest).

CARA PAKAI
----------------------------------------------------------------------------
    python3 archive_results.py --dest /Volumes/T7\\ Shield/graphrag_archive
    python3 archive_results.py --dry-run   # list what WOULD be copied, copy nothing

Default --dest comes from GRAPHRAG_ARCHIVE_DIR in .env.
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent

SOURCE_DIRS = [
    "llm/a_pure_llm/results", "llm/a_pure_llm/logs",
    "llm/b_rag/results", "llm/b_rag/logs",
    "llm/c_graphrag/results", "llm/c_graphrag/logs",
    "llm/d_lightrag/results", "llm/d_lightrag/logs",
    "llm/evaluation/results", "llm/evaluation/logs",
]


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_source_files():
    for rel in SOURCE_DIRS:
        d = REPO_ROOT / rel
        if not d.exists():
            continue
        for path in sorted(d.rglob("*")):
            if path.is_file():
                yield path


def archive(dest_dir: Path, dry_run: bool = False) -> dict:
    """Copies every file under SOURCE_DIRS into dest_dir, preserving the
    path relative to REPO_ROOT. Returns a summary dict. Never touches the
    source -- every write happens strictly under dest_dir."""
    source_files = list(iter_source_files())

    if dry_run:
        for src_path in source_files:
            print(f"[dry-run] would copy {src_path.relative_to(REPO_ROOT)}")
        return {"copied": 0, "dry_run": True, "would_copy": len(source_files)}

    dest_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for src_path in source_files:
        rel_path = src_path.relative_to(REPO_ROOT)
        dst_path = dest_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)  # copy2 preserves mtime -- never alters/deletes the SOURCE file
        checksums[str(rel_path)] = sha256_of_file(src_path)

    sums_path = dest_dir / "SHA256SUMS"
    with open(sums_path, "w", encoding="utf-8") as f:
        for rel_path, digest in sorted(checksums.items()):
            f.write(f"{digest}  {rel_path}\n")

    mismatches = []
    for rel_path, expected_digest in checksums.items():
        actual_digest = sha256_of_file(dest_dir / rel_path)
        if actual_digest != expected_digest:
            mismatches.append(rel_path)

    return {
        "copied": len(checksums),
        "dry_run": False,
        "sums_path": str(sums_path),
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=os.getenv("GRAPHRAG_ARCHIVE_DIR"),
                         help="Destination directory. Default: GRAPHRAG_ARCHIVE_DIR in .env")
    parser.add_argument("--dry-run", action="store_true",
                         help="List what would be copied, copy nothing")
    args = parser.parse_args()

    if not args.dest:
        print("[ERROR] --dest (or GRAPHRAG_ARCHIVE_DIR in .env) is required")
        sys.exit(1)

    dest_dir = Path(args.dest)
    result = archive(dest_dir, dry_run=args.dry_run)

    if result["dry_run"]:
        print(f"\n[dry-run] {result['would_copy']} file(s) would be copied. Nothing was written.")
        return

    print(f"\nCopied {result['copied']} file(s) to {dest_dir}")
    print(f"Checksums written -> {result['sums_path']}")
    if result["mismatches"]:
        print(f"[ERROR] checksum MISMATCH after copy for: {result['mismatches']}")
        sys.exit(1)
    print("All checksums verified OK -- every copied file matches its source byte-for-byte.")


if __name__ == "__main__":
    main()
