# ES Replenishment Fragility Backtest

This package implements the canonical contract at `contracts/es-replenishment-fragility/strategy_contract.yaml` and keeps the data-loading path compatible with `backtests.lib.r2_data`.

## What Changed

- The parameter grid now matches the contract more closely:
  - observation windows: `1, 3, 5` minutes
  - continuation horizons: `5, 15, 30` minutes
  - volatility and proxy thresholds fit from training quantiles
  - window-family modes: `08:30_only`, `09:30_only`, `separate_models`
- The backtest now makes the missing validation experiments explicit instead of treating them as hidden implementation choices.

## Required Inputs

Create these files under `backtests/generated/es_replenishment_fragility_2026_05_02/inputs/`:

- `macro_release_calendar.csv`
- `cash_open_calendar.csv`

Each file must contain at least:

```csv
event_date
2024-01-03
2024-02-01
```

## Data Source

- Market data is loaded from Cloudflare R2 through `backtests.lib.r2_data`.
- The confirmed ES dataset key from `data/r2_manifest.yaml` is:
  - `ES/ES-20100606-20260315.ohlcv-1m.parquet`

## Run

From the repository root:

```bash
python backtests/generated/es_replenishment_fragility_2026_05_02/backtest.py \
  --config backtests/generated/es_replenishment_fragility_2026_05_02/config.yaml
```

## Outputs

The run writes:

- `trades.parquet`
- `equity_curve.parquet`
- `metrics.json`
- `diagnostics.parquet`
- `report.md`

## Validation Experiments Added

- Observation and exit stability
- Proxy incremental value versus the volatility-only baseline
- Window-family stability across 08:30 and 09:30