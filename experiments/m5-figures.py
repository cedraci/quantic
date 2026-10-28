"""Regenerate the M5 figures from the committed CSVs -- the M4/M5 comparison document.

    uv run --no-sync --with matplotlib experiments/m5-figures.py [--only NAME ...]

Reads every ``docs/figures/m5/data/*.csv`` into ``{stem: DataFrame}`` and
calls ``make(data, out)`` of every ``experiments/m5_figures/fig_*.py``, which
writes its PNGs under ``docs/figures/m5/`` and returns their paths. The CSVs are
``quantic bench report`` output copied from the study stores, plus the uniform
baselines from ``experiments/m5-uniform-baseline.py``.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figures" / "m5" / "data"
OUT = ROOT / "docs" / "figures" / "m5"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from m5_figures import style  # noqa: E402


def load() -> dict[str, pl.DataFrame]:
    return {path.stem: pl.read_csv(path, infer_schema_length=10000)
            for path in sorted(DATA.glob("*.csv"))}


def figure_modules() -> list[str]:
    names = [m.name for m in pkgutil.iter_modules([str(HERE / "m5_figures")])]
    return sorted(n for n in names if n.startswith("fig_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="figure module names, without fig_")
    args = ap.parse_args()
    style.use_style()
    data = load()
    wanted = set(f"fig_{n.removeprefix('fig_')}" for n in args.only) if args.only else None
    for name in figure_modules():
        if wanted is not None and name not in wanted:
            continue
        module = importlib.import_module(f"m5_figures.{name}")
        for path in module.make(data, OUT):
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
