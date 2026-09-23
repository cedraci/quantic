"""Emit a precise data-request spec for the external market data machine.

The database is not reachable from this project (spec section 3.3), so requests
are produced as files and fulfilled out of band.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GranularityRequest:
    granularity: str
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataRequest:
    request_id: str
    created_utc: str
    market: str
    notes: str
    items: tuple[GranularityRequest, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["items"] = [
            {**asdict(i), "symbols": list(i.symbols)} for i in self.items
        ]
        return json.dumps(payload, indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


def default_request(
    *,
    symbols: tuple[str, ...],
    l3_symbols: tuple[str, ...],
    market: str,
    l2_start: str,
    l2_end: str,
    daily_start: str,
    daily_end: str,
    l3_start: str,
    l3_end: str,
    depth_levels: int = 10,
    request_id: str | None = None,
) -> DataRequest:
    """Build the first-request shape described in spec section 8.4."""
    unknown = set(l3_symbols) - set(symbols)
    if unknown:
        raise ValueError(f"l3_symbols must be a subset of symbols; extra: {sorted(unknown)}")

    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    return DataRequest(
        request_id=request_id or f"req-{now.strftime('%Y%m%dT%H%M%SZ')}",
        created_utc=now.isoformat(),
        market=market,
        notes=(
            "Quantic vertical B. L1+L2 for impact calibration and liquidity metrics; "
            "daily bars for covariance and ADV; L3 on a subset for book reconstruction "
            "cross-validation."
        ),
        items=(
            GranularityRequest("l1_taq", symbols, l2_start, l2_end),
            GranularityRequest(
                "l2_depth", symbols, l2_start, l2_end, {"depth_levels": depth_levels}
            ),
            GranularityRequest("l3_messages", l3_symbols, l3_start, l3_end),
            GranularityRequest("daily_bars", symbols, daily_start, daily_end),
        ),
    )
