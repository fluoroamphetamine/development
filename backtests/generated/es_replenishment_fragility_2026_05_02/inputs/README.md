# Input Calendars

Place the required public event calendar files in this folder before running the backtest.

## Required Files

- `macro_release_calendar.csv` or `.parquet`
- `cash_open_calendar.csv` or `.parquet`

## Minimum Schema

Each file must contain at least one date column. The default config expects:

- `event_date`

Example:

```csv
event_date
2024-01-03
2024-02-01
```

## Notes

- `macro_release_calendar` should include only relevant 08:30 ET U.S. macro-release dates.
- `cash_open_calendar` should include valid regular 09:30 ET cash-session opens and exclude holidays or truncated sessions.
- If you use a different date-column name, update `config.yaml` accordingly.
