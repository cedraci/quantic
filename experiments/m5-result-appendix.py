"""Splice the generated tables into the M5 result document's appendix, idempotently.

    .venv/Scripts/python.exe experiments/m5-result-appendix.py <doc.md> \\
        --uniform <m5-p2-uniform.csv> --report <reps-2 report dir> \\
        --uniform-p1 <m5-p1-uniform.csv> --report-p1 <reps-1 report dir> \\
        --jobs <m5-hardware-jobs.csv>

Each block sits between ``<!-- name -->`` and ``<!-- /name -->`` markers in the document;
the end marker is added on the first run. The report tables are the Markdown files that
``quantic bench report`` writes, pasted as they are; the uniform tables come from
``experiments/m5-uniform-baseline.py`` and the job table from
``experiments/m5-hardware-jobs.py``.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import polars as pl

UNIFORM_HEAD = ["instance", "qubits", "bitstrings examined", "uniform feasible",
                "expected per 1,000 shots", "hardware feasible per job", "hardware within 1%",
                "jobs", "device / uniform", "noiseless / uniform", "k within 1%",
                "p(success) uniform"]
REPORT_TABLES = [
    ("Matched reads (`matched_reads.md`)", "matched_reads.md",
     "Every solver at 1,000 reads; `tts` is QPU seconds for `qaoa_ibm`, wall-clock seconds for "
     "the samplers, and \"simulated\" for the simulators."),
    ("The noise ladder per instance (`qaoa_noise.md`)", "qaoa_noise.md",
     "Routed depth and two-qubit count are medians over the runs; `qpu_s` sums the hardware jobs."),
    ("Training (`qaoa_training.md`)", "qaoa_training.md", ""),
    ("HiGHS at the 1 s budget (`crossover_1s.md`)", "crossover_1s.md",
     "median gap / feasibility / TTS99 at 1%."),
]


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "exact" if v else "100,000 draws"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 0.01:
            return f"{v:.2g}"
        return f"{v:.{digits}g}"
    return str(v)


def uniform_table(path: Path) -> str:
    rows = pl.read_csv(path).sort("n_vars", "dials", "instance_seed")
    lines = ["| " + " | ".join(UNIFORM_HEAD) + " |", "|" + "---|" * len(UNIFORM_HEAD)]
    for r in rows.iter_rows(named=True):
        cells = [f"`{r['dials']}-{r['instance_seed']}`", str(r["n_vars"]), fmt(bool(r["exact"])),
                 fmt(r["uniform_frac"]), fmt(r["exp_per_1000"]), str(r["hw_feasible"]),
                 str(r["hw_success"]), str(r["hw_jobs"]), fmt(r["ratio_hw"]),
                 fmt(r["ratio_ideal"]), str(r["k_within_1pct"]), fmt(r["p_success_uniform"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report_tables(report: Path, title: str, intro: str) -> str:
    parts = [f"## {title}", "", intro]
    for heading, name, note in REPORT_TABLES:
        parts += ["", f"### {heading}", ""]
        if note:
            parts += [note, ""]
        parts.append((report / name).read_text(encoding="utf-8").strip())
    return "\n".join(parts)


def jobs_table(path: Path) -> str:
    jobs = pl.read_csv(path).sort("submitted_at")
    head = ["#", "job id", "submitted (local)", "lane", "pass", "instance", "QPU s", "queue s",
            "routed CZ", "feasible / shots", "within 1%"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in jobs.iter_rows(named=True):
        when = datetime.fromtimestamp(r["submitted_at"]).strftime("%Y-%m-%d %H:%M:%S")
        cells = [str(r["job"]), f"`{r['job_id']}`", when, r["lane"], str(r["pass"]),
                 f"`{r['dials']}-{r['instance_seed']}`", fmt(r["usage_s"]),
                 f"{r['queue_s']:,.1f}", f"{r['routed_2q']:,}",
                 f"{r['n_feasible_samples']} / {r['n_samples']:,}",
                 "yes" if r["success_0.01"] else "no"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def splice(text: str, name: str, block: str) -> str:
    begin, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    replacement = f"{begin}\n{block}\n{end}"
    if pattern.search(text):
        return pattern.sub(lambda m: replacement, text)
    if begin not in text:
        raise ValueError(f"the document has no {begin} marker")
    return text.replace(begin, replacement, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", type=Path)
    ap.add_argument("--uniform", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--uniform-p1", type=Path, required=True)
    ap.add_argument("--report-p1", type=Path, required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    args = ap.parse_args()

    text = args.doc.read_text(encoding="utf-8")
    blocks = {
        "uniform-table": uniform_table(args.uniform),
        "uniform-table-p1": uniform_table(args.uniform_p1),
        "jobs": jobs_table(args.jobs),
        "tables": report_tables(
            args.report, "Appendix: the report's tables, reps = 2",
            "Regenerated by `quantic bench report <store> --out <dir>` from the main study's "
            "store."),
        "tables-p1": report_tables(
            args.report_p1, "Appendix: the report's tables, reps = 1",
            "Regenerated by `quantic bench report <store> --out <dir>` from the reps = 1 "
            "store."),
    }
    for name, block in blocks.items():
        text = splice(text, name, block)
    args.doc.write_text(text, encoding="utf-8")
    print("spliced", args.doc, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
