# ES Replenishment Fragility Backtest

This directory contains a runnable first-pass backtest for the strategy contract:

- `strategy-contracts/es-replenishment-fragility/2026-05-02-strategy_contract.yaml`

## What It Does

- Loads ES 1-minute bars from Cloudflare R2 through `backtests.lib.r2_data`
- Loads two local public calendar files:
  - `macro_release_calendar.csv`
  - `cash_open_calendar.csv`
- Builds event-level features for 08:30 macro windows and 09:30 cash-open windows
- Runs expanding walk-forward parameter selection and out-of-sample evaluation
- Writes:
  - `trades.parquet`
  - `equity_curve.parquet`
  - `metrics.json`
  - `diagnostics.parquet`
  - `report.md`

## Required Inputs

Create these files under `backtests/generated/es_replenishment_fragility_2026_05_02/inputs/`:

- `macro_release_calendar.csv`
- `cash_open_calendar.csv`

Each file should contain at least:

- `event_date`

Example:

```csv
event_date
2024-01-03
2024-02-01
```

## Run

From the repository root:

```bash
python backtests/generated/es_replenishment_fragility_2026_05_02/backtest.py \
  --config backtests/generated/es_replenishment_fragility_2026_05_02/config.yaml
```

## Notes

- The ES R2 key is confirmed from `data/r2_manifest.yaml`.
- The public event calendars were not available in the current run, so their paths remain explicit placeholders in `config.yaml`.
- The contract already includes a solid first-pass validation path, so no extra experiments were added before implementation.
