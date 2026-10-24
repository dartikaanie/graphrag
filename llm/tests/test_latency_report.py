"""llm/latency_report.py reads the [latency] lines every condition writes to its file log;
(the per-condition logging is tested in llm/b_rag/tests/test_latency_log.py and the C/D live tests)."""

import pytest

from llm import latency_report as lr

LOG = """12:00:01 | [1/3] Id=1 something else
12:00:02 |       [latency] cond=C Id=1 retrieval_sec=1.000 total_sec=4.000
12:00:03 |       [latency] cond=C Id=2 retrieval_sec=2.000 total_sec=6.000
12:00:04 |       [latency] cond=C Id=3 retrieval_sec=3.000 total_sec=8.000
12:00:05 |       [latency] cond=A Id=9 retrieval_sec=NA total_sec=2.500
12:00:06 |       [latency] cond=D Id=7 retrieval_sec=0.500 total_sec=NA
"""


def test_report_n_p50_p95_max_per_condition(tmp_path):
    f = tmp_path / "run.log"
    f.write_text(LOG)
    lat = lr.read_latencies([f])
    assert lr.summarize(lat["C"]["retrieval"]) == {"n": 3, "p50": 2.0, "p95": 2.9, "max": 3.0}
    assert lr.summarize(lat["C"]["total"])["p95"] == pytest.approx(7.8)
    assert lat["A"]["retrieval"] == [] and lat["A"]["total"] == [2.5]       # NA is skipped, not counted
    assert lat["D"]["total"] == [] and lat["D"]["retrieval"] == [0.5]
    text = lr.format_report(lat)
    assert "C    retrieval      3    2.000    2.900    3.000" in text and "A    retrieval      0        -" in text


def test_report_handles_logs_without_latency_lines_and_missing_files(tmp_path, capsys):
    old = tmp_path / "old.log"
    old.write_text("12:00:01 | [1/3] Id=1 retrieval=1.00s\n")
    assert "No [latency] lines" in lr.format_report(lr.read_latencies([old]))
    assert lr.main([str(old)]) == 0
    assert lr.main([str(tmp_path / "nope.log")]) == 1
