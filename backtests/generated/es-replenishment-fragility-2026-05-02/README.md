# ES Replenishment Fragility Backtest

This exact package path matches the contract strategy ID `es-replenishment-fragility-2026-05-02`, which was still missing the required runnable files.

## What It Tests

- Baseline: volatility-only continuation filter using the initial benchmark-window move.
- Treatment: replenishment proxy stack using dispersion stress, relative volume context, and impact containment.
- Family splits: 08:30 macro windows and 09:30 cash-open windows are evaluated separately before any merged interpretation.

## Required Inputs

- `inputs/macro_release_calendar.csv`
- `inputs/cash_open_calendar.csv`

Both files should contain at least:

```csv
event_date
2024-01-03
2024-02-01
```

The macro calendar may stay empty if you only want to run the 09:30 family. The backtest will skip 08:30 windows rather than fabricate them.

## Run

```bash
pip install -r backtests/generated/es-replenishment-fragility-2026-05-02/requirements.txt
python backtests/generated/es-replenishment-fragility-2026-05-02/backtest.py \
  --config backtests/generated/es-replenishment-fragility-2026-05-02/config.yaml
```
