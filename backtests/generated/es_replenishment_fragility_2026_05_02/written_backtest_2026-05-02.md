# ES Replenishment Fragility Written Backtest

## Report Context

- Prepared on: 2026-05-02
- Latest triage brief reviewed: `fluoroamphetamine/triage/briefs/2026-05-02-triage-assumption-brief-scheduled-relevant-materials-review-strongest-cme-candidate-no-handoff-1915z.md`
- Primary contract reviewed: `fluoroamphetamine/strategy/contracts/es-replenishment-fragility/strategy_contract.yaml`
- Implemented backtest reviewed: `fluoroamphetamine/development/backtests/generated/es_replenishment_fragility_2026_05_02/backtest.py`
- Config reviewed: `fluoroamphetamine/development/backtests/generated/es_replenishment_fragility_2026_05_02/config.yaml`

## Executive Summary

The latest available triage brief, committed on 2026-05-02 at 19:21Z, still identifies the ES benchmark-window replenishment-failure proxy stack as the strongest current intraday CME candidate. The strategy is attractive because it stays inside data that appears to be available now: ES 1-minute bars in Cloudflare R2 plus public 08:30 macro-release dates and 09:30 cash-open dates.

The important limitation is not data readiness. It is evidence readiness. The accessible repositories contain a strategy contract, a candidate card, and a runnable first-pass backtest implementation, but they do not contain any committed run outputs such as `metrics.json`, `trades.parquet`, `equity_curve.parquet`, or a completed report with observed performance. The same materials also show that the public event calendars required to run the backtest were left as placeholders. Because of that, the strategy has an implemented method but no reviewable empirical result in the currently accessible materials.

The written backtest therefore ends with a hard conclusion: the strategy remains promising and well-formed enough for a first-pass test, but it is not yet validated. No defensible claim can be made yet about uplift versus the volatility-only baseline, net expectancy after costs, or stability across the 08:30 and 09:30 families.

## Source Review Summary

| Source | Date | Use In This Backtest | What It Contributes |
| --- | --- | --- | --- |
| Latest triage brief | 2026-05-02 19:21Z | Primary status reference | Confirms this is still the top candidate and that the blocker is executable specificity, not candidate selection. |
| Candidate card | 2026-05-02 | Mechanism and gating reference | Defines the hypothesis, required data, kill criteria, and unresolved strategy fields. |
| Main strategy contract | 2026-05-02 | Baseline contract reference | Frames the question as a trust-classification and trade-filter test with separate 08:30 and 09:30 evaluation. |
| Executable-gate contract | 2026-05-02 19:12Z | Implementation bridge | Shows the same candidate in a more explicit compiler-readiness frame and preserves the unresolved fields. |
| Public proxy-stack research note | 2026-04-29 | Feature selection rationale | Prioritizes dispersion stress, volume-rate context, and impact containment over static depth. |
| Execution-state build-order note | 2026-04-29 | Panel-assembly rationale | Explains why the same panel should be evaluated around both 08:30 and 09:30 benchmark windows. |
| Development backtest code and config | 2026-05-02 | Implemented methodology | Freezes concrete first-pass rules that go beyond the still-incomplete contract. |

## Strategy Definition

The strategy is not a pure directional alpha idea. It is a trust filter for benchmark-window moves in ES. The underlying thesis is that public bar-derived proxies can distinguish a trustworthy continuation from a fragile move better than a volatility-only gate can.

The target windows are:

- 08:30 ET macro-release windows
- 09:30 ET cash-open windows

The proxy stack named in the research and candidate materials is:

1. benchmark-window tag
2. short-window dispersion stress
3. volume-rate context
4. impact containment
5. trust or downgrade verdict

The core assumptions are:

- 1-minute OHLCV bars are coarse but still informative enough for a first-pass replenishment proxy.
- Dispersion, participation support, and containment add information beyond simple realized volatility.
- 08:30 and 09:30 should be evaluated separately before any merged threshold family is trusted.
- ES is the correct first laboratory before widening the idea to NQ or other instruments.

The main failure condition remains unchanged from the latest brief: if the proxy stack does not beat a volatility-only baseline, the replenishment layer should be demoted.

## Implemented Method

The reviewed implementation in `backtest.py` makes several concrete choices that are stronger than the incomplete contract and should be treated as implementation assumptions, not repo-validated facts.

### Data and Event Tagging

- Market data source: Cloudflare R2 key `ES/ES-20100606-20260315.ohlcv-1m.parquet`
- Bar frequency: 1 minute
- Timezone: America/New_York
- Event inputs required locally:
  - `macro_release_calendar.csv`
  - `cash_open_calendar.csv`

The code creates two event families:

- `macro_release_0830`
- `cash_open_0930`

### Observation and Entry Logic

The implementation fixes the observation slice at two minutes after the event anchor:

- 08:30 family decision time: 08:32
- 09:30 family decision time: 09:32

It then enters on the next minute's open:

- 08:33 for the 08:30 family
- 09:33 for the 09:30 family

This resolves the contract's missing entry trigger, but it does so inside the implementation rather than from an approved final contract.

### Feature Construction

The code derives:

- `atr20_pre_event`: 20-minute average true range before the event
- `impulse_return_0_2`: signed move from event-anchor open to the decision close
- `dispersion_stress_0_2`: event window range divided by pre-event ATR
- `event_window_volume`: total volume during the observation slice
- `volume_rate_ratio_0_2`: event volume relative to the median of the prior 20 same-family events
- `containment_score_0_2`: absolute retained move divided by event window range

### Baseline Rule

The baseline trades only when the first two-minute impulse exceeds a threshold expressed as a multiple of pre-event ATR. It is therefore a volatility-conditioned continuation rule.

### Treatment Rule

The treatment keeps the same volatility gate, then adds three trust conditions:

- dispersion stress must stay below a threshold
- volume-rate ratio must stay above a threshold
- containment score must stay above a threshold

If those conditions fail, the move is treated as fragile and the trade is blocked.

### Direction, Exit, and Costs

The implementation chooses continuation, not fade. It trades in the same direction as the initial two-minute impulse.

The parameter grid explores:

- impulse ATR threshold: 0.50, 0.75, 1.00, 1.25
- maximum dispersion: 1.25, 1.50, 1.75, 2.00
- minimum volume ratio: 0.80, 1.00, 1.20, 1.50
- minimum containment score: 0.45, 0.55, 0.65, 0.75
- stop multiple of event range: 0.75, 1.00, 1.25
- exit horizon in minutes: 10, 15, 20

Trading costs and execution assumptions are:

- 1 tick of slippage per side, encoded by worse entry and exit prices
- $2.50 commission per side
- ES tick size: 0.25
- ES point value: $50

### Validation Design

The contract language points toward anchored walk-forward testing. The implementation uses a shorter rolling design:

- initial training span: 3 calendar years
- next test span: 1 calendar year
- then annual expansion
- plus a `2024 onward` holdout when available

This is a real methodological divergence from the main contract, which described a first 5 years / next 1 year anchored pattern.

## Results

### Observable Results From Accessible Materials

No empirical performance results were available in the materials accessible during this run.

Specifically, the accessible repositories did not provide any committed or attached copies of:

- `metrics.json`
- `trades.parquet`
- `equity_curve.parquet`
- `diagnostics.parquet`
- a completed run report with observed metrics

The development repository does contain a runnable implementation and a config file, but its input calendar README still asks the operator to create the needed public calendar files manually. In the current run environment, no such files were attached, no R2 credentials were exposed, and no saved run artifacts were available for inspection.

### What Can Be Concluded

The current evidence supports only these result statements:

- the strategy has moved past idea-only status and into runnable first-pass implementation
- the latest triage still ranks it as the strongest current CME candidate
- the strategy remains unvalidated because no accessible run output proves that the treatment beats the baseline

### What Cannot Be Claimed

The current evidence does not support any claim about:

- classification uplift over the volatility-only baseline
- positive expectancy after slippage and commissions
- robustness across both benchmark-window families
- acceptable drawdown
- parameter stability over time

## Interpretation

This is a promising but still unproven strategy.

The good news is that the conceptual story is coherent. The latest triage brief, the candidate card, and the research notes all point in the same direction: benchmark-window liquidity quality should be read through recovery and containment, not through static depth alone. The development repo also shows that the strategy is now concrete enough to run once the public event calendars are supplied.

The less comfortable truth is that the implementation had to freeze several unresolved fields on its own. The observation length, entry timestamp, continuation direction, stop model, exit horizon grid, and walk-forward design all reflect downstream choices rather than fully approved source decisions. That means even a future positive run would need to be interpreted as "positive under this implementation choice set," not yet "source-validated across the canonical contract."

The biggest strategic risk is still the same one named in the latest triage brief: the replenishment proxy may end up being only a dressed-up volatility filter. Until treatment-minus-baseline results are visible separately for 08:30 and 09:30 windows, the strategy should be treated as a credible test candidate rather than as an evidence-backed edge.

## Bottom Line

As of 2026-05-02, the ES replenishment-fragility idea is the strongest currently triaged intraday CME candidate, and the backtest design is specific enough to run. But the latest available materials do not contain observed results, so the strategy remains data-feasible rather than validated.

## Next Actions

1. Supply the two missing public event calendars so the existing implementation can actually run: one file for relevant 08:30 macro-release dates and one file for valid 09:30 cash-session opens.
2. Run the current backtest implementation against the configured R2 ES dataset and persist the full output set, especially `metrics.json`, `trades.parquet`, and the generated report.
3. Evaluate treatment versus baseline separately for 08:30 and 09:30 before looking at any pooled conclusion.
4. Decide whether the current implementation assumptions should be promoted into the canonical contract, especially the two-minute observation slice, continuation-only direction rule, 10 to 20 minute exits, and 3-year rolling train window.
5. Reject the replenishment layer quickly if treatment-minus-baseline uplift is not positive net of costs in both window families.
6. If one family works and the other does not, split the strategy rather than forcing a merged rule set.
