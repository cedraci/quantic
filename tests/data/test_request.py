import json

from quantic.data.request import default_request


def test_default_request_matches_spec_section_8_4(tmp_path):
    req = default_request(
        symbols=tuple(f"SYM{i:02d}" for i in range(25)),
        l3_symbols=("SYM00", "SYM01", "SYM02"),
        market="XNAS",
        l2_start="2026-02-02",
        l2_end="2026-02-27",
        daily_start="2025-03-01",
        daily_end="2026-02-27",
        l3_start="2026-02-23",
        l3_end="2026-02-27",
    )
    by_granularity = {item.granularity: item for item in req.items}
    assert set(by_granularity) == {"l1_taq", "l2_depth", "l3_messages", "daily_bars"}
    assert len(by_granularity["l1_taq"].symbols) == 25
    assert by_granularity["l2_depth"].options["depth_levels"] == 10
    assert by_granularity["l3_messages"].symbols == ("SYM00", "SYM01", "SYM02")
    assert by_granularity["daily_bars"].start_date == "2025-03-01"


def test_request_writes_readable_json(tmp_path):
    req = default_request(
        symbols=("AAA",),
        l3_symbols=("AAA",),
        market="XNAS",
        l2_start="2026-02-02",
        l2_end="2026-02-27",
        daily_start="2025-03-01",
        daily_end="2026-02-27",
        l3_start="2026-02-23",
        l3_end="2026-02-27",
    )
    out = tmp_path / "request.json"
    req.write(out)
    payload = json.loads(out.read_text())
    assert payload["market"] == "XNAS"
    assert len(payload["items"]) == 4
