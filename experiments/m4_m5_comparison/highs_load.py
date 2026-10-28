import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
D = str(Path(__file__).resolve().parents[2] / "docs" / "figures" / "m5" / "data") + "/"
DIALS = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4"]
m4 = pl.read_csv(D + "m4-cells.csv", infer_schema_length=10000).filter(
    (pl.col("tier") == "T0") & (pl.col("solver") == "exact_pwl_milp") & pl.col("dials").is_in(DIALS)
)
m5 = pl.read_csv(D + "m5-p2-cells.csv", infer_schema_length=10000).filter(
    pl.col("solver") == "exact_pwl_milp"
)
for b in (0.1, 1.0, 10.0, 60.0):
    x = m4.filter(pl.col("budget_s") == b).select(
        "dials", pl.col("t_run_s").alias("m4"), pl.col("tts99_s_0.01").alias("m4tts")
    )
    print(b, x["m4"].min(), x["m4"].max(), "tts", x["m4tts"].min(), x["m4tts"].max())
j = (
    m4.filter(pl.col("budget_s") == 1.0)
    .select("dials", pl.col("t_run_s").alias("m4"))
    .join(m5.select("dials", pl.col("t_run_s").alias("m5")), on="dials")
    .with_columns((pl.col("m5") / pl.col("m4")).alias("ratio"))
)
print(j.sort("dials"))
print("ratio range", j["ratio"].min(), j["ratio"].max())
