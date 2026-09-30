"""The pieces of the exact program -- spec section 7.3.

One binary ``y[i,t,k]`` per admissible lot value ``k`` of each cell. Any cost
that depends on one cell alone is then exactly linear in the binaries, however
concave it is in the lots: the piecewise-linear under- and over-estimators of
section 7.3 coincide on the lot grid.

Each piece here is a claim about the objective, and each is tested against
``problem.objective.evaluate`` or ``ScenarioSet.losses`` on its own, before any
solver sees it. Where a piece can be read off the evaluator it is, so that no
second implementation exists to drift; where it cannot -- the loss model, the
gradient -- the test is what ties it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from quantic.problem.instance import Instance
from quantic.problem.objective import evaluate
from quantic.problem.risk import CVaRRisk
from quantic.problem.schedule import Schedule
from quantic.solvers.classical.lot_grid import LotGrid
from quantic.solvers.classical.scoring import CellCost


@dataclass(frozen=True)
class Columns:
    """Which cell and which lot value each binary stands for."""

    asset: np.ndarray     # (n_y,)
    bucket: np.ndarray    # (n_y,)
    lots: np.ndarray      # (n_y,)
    first: np.ndarray     # (N, T) index of the cell's first column
    count: np.ndarray     # (N, T) how many columns the cell has

    @property
    def n(self) -> int:
        return int(self.lots.size)

    def schedule(self, y: np.ndarray) -> Schedule:
        """The schedule a solution's binaries select."""
        n_assets, n_buckets = self.first.shape
        grid = [[0] * n_buckets for _ in range(n_assets)]
        for i in range(n_assets):
            for t in range(n_buckets):
                lo = int(self.first[i, t])
                hi = lo + int(self.count[i, t])
                grid[i][t] = int(self.lots[lo + int(np.argmax(y[lo:hi]))])
        return Schedule(lots=tuple(tuple(row) for row in grid))


def columns_of(grid: LotGrid) -> Columns:
    asset, bucket, lots = [], [], []
    first = np.zeros((grid.n_assets, grid.n_buckets), dtype=np.int64)
    count = np.zeros((grid.n_assets, grid.n_buckets), dtype=np.int64)
    for i in range(grid.n_assets):
        for t in range(grid.n_buckets):
            first[i, t] = len(lots)
            count[i, t] = len(grid.values[i][t])
            for k in grid.values[i][t]:
                asset.append(i)
                bucket.append(t)
                lots.append(k)
    return Columns(
        asset=np.array(asset, dtype=np.int64),
        bucket=np.array(bucket, dtype=np.int64),
        lots=np.array(lots, dtype=np.int64),
        first=first,
        count=count,
    )


def cell_costs(
    instance: Instance, columns: Columns, *, cell_cost: CellCost | None = None
) -> np.ndarray:
    """Spread plus temporary impact of each column, read off the evaluator.

    Both terms are sums over cells, so what one cell costs is what the
    evaluator charges a schedule that trades in that cell alone. No second
    implementation of either formula exists to drift away from the first.
    """
    zeta = 0.0 if isinstance(instance.risk, CVaRRisk) else None
    out = np.zeros(columns.n)
    for c in range(columns.n):
        i, t, k = int(columns.asset[c]), int(columns.bucket[c]), int(columns.lots[c])
        if k == 0:
            continue
        lots = [[0] * instance.n_buckets for _ in range(instance.n_assets)]
        lots[i][t] = k
        alone = evaluate(
            instance, Schedule(lots=tuple(tuple(r) for r in lots)), zeta=zeta
        )
        temporary = (
            alone.temporary_impact_cost if cell_cost is None else cell_cost(i, t, k)
        )
        out[c] = alone.spread_cost + temporary
    return out


def permanent_constant(instance: Instance) -> float:
    """Permanent impact under full liquidation, which no schedule can change.

    Per asset the term is ``sum_t (sum_{s<t} q_s + q_t / 2) * q_t``, and that
    sum is ``(sum_t q_t)**2 / 2``. Selling everything in the first bucket is
    one schedule that liquidates in full, so the evaluator's figure there is
    the figure everywhere.
    """
    lots = tuple(
        (int(x), *([0] * (instance.n_buckets - 1))) for x in instance.initial_lots
    )
    zeta = 0.0 if isinstance(instance.risk, CVaRRisk) else None
    return float(
        evaluate(instance, Schedule(lots=lots), zeta=zeta).permanent_impact_cost
    )


def loss_model(instance: Instance) -> tuple[np.ndarray, np.ndarray]:
    """``L_s = base_s - sum over cells of x[i,t] * tail[s,i,t]``.

    A loss is linear in the holdings, and a lot sold in bucket ``t`` leaves
    every holding from ``t`` on, so its coefficient is the sum of the
    per-holding coefficients from ``t`` to the horizon.
    """
    risk = instance.risk
    returns = np.asarray(risk.scenarios.returns, dtype=float)          # (S, T, N)
    per_lot = instance.lot_sizes().astype(float) * instance.params.prices()
    per_holding = np.einsum("stn,n->snt", -returns, per_lot)           # (S, N, T)
    position = np.asarray(instance.initial_lots, dtype=float)
    base = np.einsum("snt,n->s", per_holding, position)
    tail = np.cumsum(per_holding[:, :, ::-1], axis=2)[:, :, ::-1]
    return base, tail


def variance_gradient(instance: Instance, schedule: Schedule) -> np.ndarray:
    """``d(lam * sum_t h_t' Sigma h_t) / d x[i,t]``, in currency per lot."""
    lam = float(instance.risk.lam)
    lot = instance.lot_sizes().astype(float)
    sigma = np.asarray(instance.params.price_covariance(), dtype=float)
    holdings = instance.remaining_lots(schedule).astype(float) * lot[:, None]
    pull = sigma @ holdings                                            # (N, T)
    tail = np.cumsum(pull[:, ::-1], axis=1)[:, ::-1]
    return -2.0 * lam * lot[:, None] * tail


def coupling_rows(columns: Columns, grid: LotGrid, n_vars: int):
    """The constraints that hold whatever the risk term is."""
    n_assets, n_buckets = columns.first.shape
    a_rows, a_cols, a_vals, lower, upper = [], [], [], [], []
    row = 0

    def add(cols, vals, lo, hi):
        nonlocal row
        a_rows.extend([row] * len(cols))
        a_cols.extend(int(c) for c in cols)
        a_vals.extend(float(v) for v in vals)
        lower.append(lo)
        upper.append(hi)
        row += 1

    every = np.arange(columns.n)
    for i in range(n_assets):
        for t in range(n_buckets):
            mine = every[(columns.asset == i) & (columns.bucket == t)]
            add(mine, np.ones(mine.size), 1.0, 1.0)

    for i in range(n_assets):
        mine = every[columns.asset == i]
        position = float(grid.initial_lots[i])
        add(mine, columns.lots[mine], position, position)

    if grid.cardinality is not None:
        for t in range(n_buckets):
            mine = every[(columns.bucket == t) & (columns.lots > 0)]
            add(mine, np.ones(mine.size), -np.inf, float(grid.cardinality))

    matrix = sparse.csr_array(
        (a_vals, (a_rows, a_cols)), shape=(row, n_vars), dtype=float
    )
    return matrix, np.array(lower), np.array(upper)
