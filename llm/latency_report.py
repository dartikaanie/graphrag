"""Latency of a run, read from its file log (no Neo4j, no LLM).

Every condition (A, B, C, D) writes one line per answered question to the run's file log
(`logs/<timestamp>_<provider>_<model>.log`, next to the script that made it):

    ... [latency] cond=C Id=123 retrieval_sec=1.234 total_sec=5.678

retrieval_sec = question in -> context ready (A: NA, no retrieval);
total_sec     = question in -> answer returned by the generator (== the record's
                end_to_end_latency_sec; evaluation steps such as citation extraction
                and cosine similarity are not included).
Retrieval-only runs log total_sec=NA.

    python -m llm.latency_report llm/c_graphrag/logs/20261024T101500Z_openai_gpt-4o-mini.log [more.log ...]

Prints, per condition and value: n, p50, p95, max (seconds; percentiles are linear
interpolation, the same as pandas' quantile used in the run summaries).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINE = re.compile(r"\[latency\] cond=(?P<cond>[ABCD]) Id=(?P<qid>\d+) retrieval_sec=(?P<retrieval>\S+) total_sec=(?P<total>\S+)")


def _num(text: str) -> float | None:
    return None if text == "NA" else float(text)


def read_latencies(paths) -> dict[str, dict[str, list[float]]]:
    """{condition: {"retrieval": [...], "total": [...]}} from the [latency] lines of the given logs."""
    out: dict[str, dict[str, list[float]]] = {}
    for path in paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if not m:
                continue
            bucket = out.setdefault(m["cond"], {"retrieval": [], "total": []})
            for key in ("retrieval", "total"):
                value = _num(m[key])
                if value is not None:
                    bucket[key].append(value)
    return out


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values)}


def format_report(latencies: dict[str, dict[str, list[float]]]) -> str:
    if not latencies:
        return "No [latency] lines found (logs written before this logging existed have none)."
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    rows = [f"{'cond':<5}{'value':<11}{'n':>5}{'p50':>9}{'p95':>9}{'max':>9}"]
    for cond in sorted(latencies):
        for key, label in (("retrieval", "retrieval"), ("total", "total")):
            s = summarize(latencies[cond][key])
            rows.append(f"{cond:<5}{label:<11}{s['n']:>5}{fmt(s['p50']):>9}{fmt(s['p95']):>9}{fmt(s['max']):>9}")
    return "\n".join(rows) + "\n(seconds; total = question in -> answer returned)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="file log(s) of a run")
    args = parser.parse_args(argv)
    missing = [p for p in args.logs if not Path(p).exists()]
    if missing:
        print(f"[ERROR] not found: {', '.join(missing)}")
        return 1
    print(format_report(read_latencies(args.logs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
