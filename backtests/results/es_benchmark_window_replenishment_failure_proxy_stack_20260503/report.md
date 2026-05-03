# Strategy Contract Review & Backtesting Brief

## 1. Contract Review Summary

| Source | Key Insight | Relevance |
| --- | --- | --- |
| Canonical contract | Defines a trust-classification problem across tagged 08:30 and 09:30 benchmark windows. | Primary source of scope and metrics. |
| Backtest ledger | Shows completed equivalents only for MSCI review bridge and ES replenishment fragility. | Confirms this strategy is materially new. |
| Data contract and R2 manifest | Confirm the shared R2 loader pattern and the ES 1-minute parquet key. | Makes the package runnable against the existing data path. |

## 2. Strategy Interpretation

- Contract file: `strategies/pending/es_benchmark_window_replenishment_failure_proxy_stack_20260503.yaml`
- Strategy ID: `es_benchmark_window_replenishment_failure_proxy_stack_20260503`
- Core thesis: A public proxy stack built from dispersion stress, relative volume, and impulse containment should identify more trustworthy continuation windows than a volatility-only gate.
- Assumptions: The first executable pass uses continuation direction, 3- or 5-minute observation slices, 5/10/15-minute horizons, and separate or family-specific evaluation of 08:30 and 09:30 windows.
- Required inputs and outputs: ES 1-minute R2 bars, an optional real 08:30 macro calendar, deterministic 09:30 cash-open dates from the bar set, and the standard trades/equity/metrics/diagnostics/report outputs.
- Key constraints: The source contract leaves direction, exact exit, and production thresholds unresolved, so this package freezes only the minimum necessary assumptions and reports them explicitly.

## 3. Build / Skip / Repair Decision

- Chosen action: `build_new`
- Decision evidence: No equivalent generated package exists for `es_benchmark_window_replenishment_failure_proxy_stack_20260503`, while the existing completed packages map to different strategies. The contract is new and materially distinct from the completed ES replenishment-fragility implementation because it frames a benchmark-window trust classifier with 08:30/09:30 family diagnostics instead of the prior replenishment fragility trade package.

## 4. Validation Assessment

The contract is sufficient for a first runnable backtest only if the implementation makes its unresolved execution choices explicit. The added experiments are therefore the minimum useful set needed to test whether the proxy stack is real, stable across nearby timing choices, and consistent across event families.

## 5. Added Experiments

- Name: Observation And Horizon Stability
  Purpose: Check whether the proxy edge is still present when timing moves slightly.
  Assumption being tested: The edge is not just a single lucky slice-length and exit-pair artifact.
  Required data or inputs: ES 1-minute bars plus the event-family calendars already required by the contract.
  Success metric: Treatment uplift stays directionally positive across neighboring observation and horizon settings.
  Failure metric: Lift disappears once timing moves away from one narrow setting.
  Decision rule: Reject any timing choice that cannot survive adjacent timing variants.
  Key limitation: Minute bars still hide sub-minute execution differences.
- Name: Incremental Value Versus Volatility
  Purpose: Test whether the proxy stack adds information beyond a simple impulse-size gate.
  Assumption being tested: Dispersion stress, relative volume, and containment are not just restating volatility.
  Required data or inputs: The same event set and market bars used for the baseline.
  Success metric: Treatment minus baseline classification accuracy is positive out of sample.
  Failure metric: Treatment matches or trails baseline despite greater complexity.
  Decision rule: Drop the proxy layer if walk-forward uplift is non-positive.
  Key limitation: A positive result is still tied to the first-pass continuation framing.
- Name: Window Family Stability
  Purpose: Verify whether 08:30 and 09:30 belong in the same strategy lane.
  Assumption being tested: The mechanism behaves consistently across macro-release and cash-open families.
  Required data or inputs: A real 08:30 macro calendar plus the deterministic 09:30 family.
  Success metric: Family-level uplift has the same sign and similar magnitude.
  Failure metric: One family works and the other does not.
  Decision rule: Split the strategy by family if the signs conflict.
  Key limitation: Family imbalance can distort confidence if one side has far fewer usable events.

## 6. Backtesting Approach

The backtest loads the confirmed ES 1-minute R2 dataset, derives deterministic 09:30 cash-open events from the market-data dates, optionally loads a real 08:30 macro-release calendar, computes observation-slice features for 3- and 5-minute windows, labels continuation over 5/10/15-minute horizons, and fits a volatility-only baseline against a replenishment-proxy treatment in rolling month-based walk-forward splits.

## 7. Backtesting Code or Repair Output

Run `python backtests/generated/es_benchmark_window_replenishment_failure_proxy_stack_20260503/backtest.py --config backtests/generated/es_benchmark_window_replenishment_failure_proxy_stack_20260503/config.yaml` from the repo root. If the macro calendar file is absent, the package skips the 08:30 family and records that gap in `metrics.json` and `report.md` rather than inventing release dates.

## 8. Immediate Next Actions

- Add a real public 08:30 macro-release calendar at the configured package path to enable the macro family.
- Run the package and inspect `metrics.json` before treating any threshold choice as meaningful.
- Compare treatment uplift separately for `08:30` and `09:30` before pooling conclusions.
- Promote only the timing and threshold settings that remain stable across neighboring variants.
- Reject the proxy stack quickly if it cannot beat the volatility-only baseline out of sample.
