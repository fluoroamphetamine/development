# Input Calendars

Provide two local event calendars:

- `macro_release_calendar.csv`
- `cash_open_calendar.csv`

Minimum schema:

```csv
event_date
2024-01-03
2024-02-01
```

Notes:

- Dates should be trading dates in `YYYY-MM-DD` form.
- The backtest maps `macro_release_calendar.csv` to the 08:30 ET family.
- The backtest maps `cash_open_calendar.csv` to the 09:30 ET family.
- Events with missing required bars around the observation or continuation windows are dropped and logged.
