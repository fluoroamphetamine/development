# Market Data Contract

This file defines the canonical market-data schemas that generated backtests in this repository should use. Development/backtest agents must read this before generating strategy-specific code.

## Runtime data access

Backtests run in `fluoroamphetamine/development` and should load market data from Cloudflare R2 through the shared helper:

```python
from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns
```

Backtests must not hard-code R2 credentials. Credentials are supplied by GitHub Actions secrets through environment variables.

Required runtime env vars:

- `R2_ENDPOINT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`

Known dataset keys are indexed in:

```text
data/r2_manifest.yaml
```

## 1-minute OHLCV schema

The 1-minute futures OHLCV data uses the same schema as the current ES 1-minute dataset.

Known ES 1m object:

```text
ES/ES-20100606-20260315.ohlcv-1m.parquet
```

Canonical config mapping:

```yaml
data:
  source: r2
  market_data:
    r2_key: ES/ES-20100606-20260315.ohlcv-1m.parquet
  local_cache_dir: .cache/r2

columns:
  timestamp: datetime_utc
  open: Open
  high: High
  low: Low
  close: Close
  volume: Volume
```

Notes:

- `datetime_utc` is the event/bar timestamp and should be treated as UTC unless a dataset-specific contract says otherwise.
- Backtests may normalize these columns internally to lowercase canonical names: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- Generated code must not assume lowercase OHLCV columns exist in the raw Parquet file.

## Sanitized tick schema

Tick data is sanitized trade data with deterministic intra-timestamp ordering.

Columns:

| Column | Type | Meaning |
|---|---|---|
| `ts_event` | `datetime[ns, UTC]` | Exchange event timestamp. |
| `intra_ts_rank` | `UInt8` | Zero-based order of trades sharing the same `ts_event` after sorting by sequence. |
| `side` | `UInt8` | Encoded aggressor side: `0 = N` none/unspecified, `1 = A` sell aggressor, `2 = B` buy aggressor. |
| `price_ticks` | `float64` | Trade price represented in ticks. |
| `size` | `UInt16` | Positive trade quantity. |

Ordering rule:

```text
Trades are ordered by (ts_event, sequence) before intra_ts_rank is assigned.
```

Backtest requirements for tick data:

- Preserve ordering by `(ts_event, intra_ts_rank)` when replaying trades.
- Do not sort only by `ts_event`, because multiple trades can share the same timestamp.
- Treat `side` as encoded aggressor side, not position direction.
- Convert `price_ticks` to price only when the strategy or reporting layer requires price units.
- Validate that `size > 0` before computing volume or trade-flow metrics.

Recommended config mapping for tick data:

```yaml
data:
  source: r2
  market_data:
    r2_key: <instrument>/<tick-file>.parquet
  local_cache_dir: .cache/r2
  schema: sanitized_ticks

columns:
  timestamp: ts_event
  intra_ts_rank: intra_ts_rank
  side: side
  price_ticks: price_ticks
  size: size
```

## Event-calendar rules

Cash-open calendars may be generated deterministically from trading dates or from the market data itself. They represent the 09:30 America/New_York regular U.S. equity cash open.

Macro calendars must not be invented. If a strategy requires macro-release dates and no macro calendar is provided, generated backtests must either:

1. skip the macro event family and report it as unavailable, or
2. fail clearly with a missing-input error.

Do not silently create fake macro-release dates.

## Agent generation rules

Development/backtest agents must:

1. Read this data contract before generating backtest code.
2. Read `data/r2_manifest.yaml` before choosing R2 keys.
3. Use `backtests.lib.r2_data` for R2 access.
4. Emit `config.yaml` with explicit column mappings.
5. Avoid requiring local input files unless they are committed with the generated package or explicitly declared as missing required inputs.
6. Keep tick replay deterministic by sorting ticks by `(ts_event, intra_ts_rank)`.
