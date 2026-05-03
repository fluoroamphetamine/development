# Strategy Contract Review & Backtesting Brief

## 1. Contract Review Summary

| Source | Key Insight | Relevance |
| --- | --- | --- |
| Canonical contract | Defines the ES benchmark-window trust-filter question and named proxy stack. | Primary implementation source. |
| Executable-gate contract | Confirms compiler-facing gaps around decision time, exit horizon, thresholds, and evaluation protocol. | Explains which experiments need to be made explicit. |
| R2 manifest | Confirms the ES 1-minute parquet key in Cloudflare R2. | Lets the config stay runnable against the shared data path. |

## 2. Strategy Interpretation

- Contract: `contracts/es-replenishment-fragility/strategy_contract.yaml`
- Strategy ID: `es_replenishment_fragility_2026_05_02`
- Core thesis: A public proxy stack built from short-window dispersion stress, volume-rate context, and impact containment should separate trustworthy benchmark-window continuation from fragile failure better than a volatility-only gate.
- Key assumptions: The first-pass test stays ES-only, uses 1-minute bars, treats the signal as a continuation filter, and evaluates 08:30 and 09:30 windows separately before any merge.
- Inputs and outputs: Inputs are ES 1-minute OHLCV bars from R2 plus public macro-release and cash-open calendars. Outputs are event diagnostics, trades, equity, metrics, and a written report.
- Constraints: The contract remains incomplete, so entry timing, exit timing, thresholds, and walk-forward protocol must be frozen as explicit implementation assumptions.

## 3. Validation Assessment

The contract is strong enough for a first runnable backtest, but it still needs a small set of validation experiments because its main risks are not just parameter tuning risks. They are structural: whether the proxy adds information beyond volatility, whether the best observation and holding windows are stable, and whether 08:30 and 09:30 behave like the same family.

## 4. Added Experiments

- Observation And Exit Stability: Test whether the strategy only works for one narrow timing choice.
  Assumption tested: Decision-time and exit-horizon choices are not hiding a brittle implementation.
  Success metric: Treatment accuracy and average trade return remain competitive across multiple observation and continuation windows.
  Failure metric: The edge collapses outside one single timing pair.
  Decision rule: Reject any timing choice whose treatment uplift is isolated and not reproducible in neighboring windows.
  Limitation: This is still based on 1-minute bars, so sub-minute microstructure remains unobserved.
- Proxy Incremental Value Vs Volatility: Check whether the proxy stack adds real information beyond the volatility gate.
  Assumption tested: Dispersion, volume-rate context, and containment are not just restating impulse size.
  Success metric: Treatment minus baseline classification accuracy is positive overall and by window family.
  Failure metric: Treatment matches or trails the baseline despite extra complexity.
  Decision rule: Demote the proxy layer if uplift is non-positive after walk-forward testing.
  Limitation: A positive result is still conditional on the chosen first-pass continuation mapping.
- Window Family Stability: Confirm whether 08:30 and 09:30 can share a single concept.
  Assumption tested: The mechanism behaves similarly across macro-release and cash-open benchmark windows.
  Success metric: Treatment uplift is directionally consistent in both families.
  Failure metric: One family works while the other is flat or negative.
  Decision rule: Split the strategy by family or reject the merged framing if instability persists.
  Limitation: Family imbalance can still reduce confidence if one side has much fewer valid events.

## 5. Backtesting Approach

The backtest loads the confirmed ES R2 dataset, tags 08:30 and 09:30 event dates from local calendar files, computes feature rows for observation windows of 1, 3, and 5 minutes, labels continuation outcomes over 5, 15, and 30 minute horizons, fits quantile thresholds in an anchored 5-year train / 1-year test walk-forward, and compares a volatility-only baseline with a proxy-stack treatment.

## 6. Backtesting Code

- Run from repo root with the package config in this directory.
- Outputs written: `trades.parquet`, `equity_curve.parquet`, `metrics.json`, `diagnostics.parquet`, `report.md`.

## 7. Immediate Next Actions

- Populate the macro-release and cash-open input calendars in the package inputs directory.
- Run the package against the confirmed ES R2 key and inspect `metrics.json` first.
- Check treatment uplift separately for `macro_release_0830` and `cash_open_0930` before pooling anything.
- Promote the winning observation window, horizon, and family mode back into the canonical contract if the results are stable.
- Reject the replenishment layer quickly if it does not beat the volatility-only baseline after costs.

## Result Snapshot

- Baseline accuracy: 0.5327
- Treatment accuracy: 0.5334
- Treatment minus baseline accuracy: 0.0007