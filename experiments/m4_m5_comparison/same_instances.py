"""Do M4 and M5 score the same T0 instances against the same brackets?"""

import sys

import duckdb

sys.stdout.reconfigure(encoding="utf-8")

DIALS = ["none", "D1", "D2", "D4", "D1+D2", "D1+D4", "D2+D4", "D1+D2+D4"]
Q = """select tier, dials, instance_seed, instance_hash, lower, upper
       from runs where solver = 'ref:exact_pwl_milp' and tier = 'T0'
       and store_status = 'ok' order by dials, instance_seed"""


def load(path):
    con = duckdb.connect(path, read_only=True)
    try:
        rows = con.execute(Q).fetchall()
    finally:
        con.close()
    return {(r[1], r[2]): r for r in rows if r[1] in DIALS}


a = load(sys.argv[1])
b = load(sys.argv[2])
print("M4 refs", len(a), "M5 refs", len(b))
same_hash = sum(1 for k in b if k in a and a[k][3] == b[k][3])
same_br = sum(1 for k in b if k in a and a[k][4] == b[k][4] and a[k][5] == b[k][5])
print("same instance hash:", same_hash, "of", len(b))
print("bit-identical bracket:", same_br, "of", len(b))
worst = max(abs(a[k][4] - b[k][4]) / abs(b[k][4]) for k in b if k in a)
print("worst relative difference in lower:", worst)
