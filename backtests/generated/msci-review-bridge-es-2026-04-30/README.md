# MSCI Review Bridge ES Backtest

This package covers the uncovered strategy contract `msci-review-bridge-es-2026-04-30` from `fluoroamphetamine/strategy`.

## What It Tests

- Baseline: continuation from the first five minutes after the 16:00 close checkpoint.
- Treatment: continuation from the broader 16:05-16:30 bridge with sensitivity checks for shorter bridge endpoints.
- Diagnostics: month-end overlap handling and next-open exit timing.

## Required Input

Populate `inputs/msci_review_schedule.csv` with real MSCI implementation dates:

```csv
event_date
2024-02-29
2024-05-31
```

The placeholder file included in this package is intentionally empty so the backtest never invents event dates.

## Data Source

- Market data loader: `backtests.lib.r2_data`
- Confirmed ES 1-minute R2 key: `ES/ES-20100606-20260315.ohlcv-1m.parquet`

## Run

```bash
pip install -r backtests/generated/msci-review-bridge-es-2026-04-30/requirements.txt
python backtests/generated/msci-review-bridge-es-2026-04-30/backtest.py \
  --config backtests/generated/msci-review-bridge-es-2026-04-30/config.yaml
```

## Outputs

- `trades.parquet`
- `equity_curve.parquet`
- `metrics.json`
- `diagnostics.parquet`
- `report.md`
