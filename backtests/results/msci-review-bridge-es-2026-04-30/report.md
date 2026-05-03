# msci-review-bridge-es-2026-04-30

## Contract interpretation
- Baseline: trade continuation of the 16:00-16:05 move and hold to the chosen next-session exit.
- Treatment: trade continuation of the broader post-close bridge and require retained absorption before entry.
- Explicit fallback assumption: month-end overlap is approximated as the last 2 business days of the month.

## Added validation experiments
### Month-end overlap sensitivity
- Purpose: Test whether the apparent bridge edge survives different contamination handling around month-end flows.
- Assumption: The signal is MSCI-transfer specific, not mostly ordinary month-end rebalance noise.
- Success metric: Treatment-minus-baseline alignment remains positive on non-overlap events and under exclusion.
- Failure metric: The positive effect disappears once overlap dates are excluded.
- Decision rule: Keep the bridge thesis only if non-overlap results stay directionally positive after costs.
- Limitation: The overlap tag is still a public-calendar heuristic until the repo freezes an exact tagging rule.

### Execution timing sensitivity
- Purpose: Measure whether the edge belongs to the bridge checkpoint itself or only to a fragile overnight/open print.
- Assumption: The bridge contains transfer information before the next official cash open, not just at one exit print.
- Success metric: Results stay stable across 09:30, 09:35, and 10:00 exit variants.
- Failure metric: Performance is concentrated in one print and collapses on nearby exits.
- Decision rule: Reject the bridge-first trade framing if the apparent edge vanishes outside a single exit timestamp.
- Limitation: This still does not resolve the separate question of entering at 16:30 versus deferring entry to the next open.

## Walk-forward summary
```json
{
  "reason": "The packaged MSCI review schedule template is empty. Replace it with real review dates to run the backtest against R2 market data.",
  "required_inputs": [
    "MSCI review schedule CSV or Parquet with a review_date column",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET"
  ],
  "status": "missing_schedule_data"
}
```

## Selected parameters

## Output notes
- No trades were generated. This usually means the MSCI schedule file is still empty or no events survived the filters.
