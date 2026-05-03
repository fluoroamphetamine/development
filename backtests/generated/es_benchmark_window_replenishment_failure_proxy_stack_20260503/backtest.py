#!/usr/bin/env python3
"""Backtest for ES benchmark-window replenishment-failure proxy stack.

This implementation keeps the contract honest in two ways:
1. It evaluates the 09:30 cash-open family deterministically from market-data dates.
2. It never invents the 08:30 macro calendar. If that file is absent, the macro family
   is skipped and reported as unavailable.

The first executable assumptions are deliberately minimal:
- continuation-only downstream trade direction
- decision after a 3- or 5-minute observation slice
- fixed 5, 10, or 15 minute evaluation horizon
- 1 contract, 1 tick of slippage per side, and 2.50 USD commission per side
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml


TIMEZONE = "America/New_York"
NY_TZ = ZoneInfo(TIMEZONE)
EPSILON = 1e-9


@dataclass(frozen=True)
class ParameterChoice:
    observation_window_minutes: int
    evaluation_horizon_minutes: int
    baseline_impulse_threshold_bps: float
    dispersion_stress_threshold_ratio: float
    volume_rate_context_threshold_ratio: float
    impulse_containment_threshold_ratio: float
    event_family_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_market_bars(config: dict[str, Any]) -> pl.DataFrame:
    from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns

    bars = load_market_data_from_config(config)
    bars = normalize_ohlcv_columns(bars, config)
    if "timestamp" not in bars.columns:
        raise ValueError("Normalized bars are missing the canonical 'timestamp' column.")

    dtype = bars.schema.get("timestamp")
    expr = pl.col("timestamp")
    if dtype == pl.Utf8:
        expr = expr.str.to_datetime(strict=False)
    bars = bars.with_columns(expr.alias("timestamp"))
    if getattr(bars.schema["timestamp"], "time_zone", None):
        bars = bars.with_columns(pl.col("timestamp").dt.convert_time_zone(TIMEZONE))
    else:
        bars = bars.with_columns(pl.col("timestamp").dt.replace_time_zone(TIMEZONE))

    return (
        bars.sort("timestamp")
        .with_columns(
            pl.col("timestamp").dt.date().alias("session_date"),
            pl.col("timestamp").dt.year().alias("calendar_year"),
        )
    )


def build_cash_open_calendar(bars: pl.DataFrame) -> list[date]:
    cash_open_rows = bars.filter(
        (pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() == 30)
    )
    return cash_open_rows.get_column("session_date").unique().sort().to_list()


def read_optional_macro_calendar(config: dict[str, Any]) -> tuple[list[date], str | None]:
    macro_cfg = config["event_calendars"]["macro_release_calendar"]
    path = Path(macro_cfg["path"])
    if not path.exists():
        return [], f"Macro calendar not found at {path.as_posix()}."

    if macro_cfg.get("format", "csv") == "csv":
        table = pl.read_csv(path, try_parse_dates=True)
    else:
        table = pl.read_parquet(path)

    column = macro_cfg.get("date_column", "event_date")
    if column not in table.columns:
        raise ValueError(
            f"Macro calendar is missing '{column}'. Available columns: {table.columns}"
        )

    series = table.get_column(column)
    if series.dtype != pl.Date:
        if series.dtype == pl.Utf8:
            series = series.str.strptime(pl.Date, strict=False)
        else:
            series = series.cast(pl.Date)
    return series.drop_nulls().unique().sort().to_list(), None


def observation_slice(records: list[dict[str, Any]], anchor_idx: int, obs_minutes: int) -> list[dict[str, Any]]:
    decision_idx = anchor_idx + obs_minutes
    return records[anchor_idx : decision_idx + 1]


def expected_range(anchor_ts: datetime, horizon_minutes: int) -> list[datetime]:
    return [anchor_ts + timedelta(minutes=offset) for offset in range(horizon_minutes + 1)]


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        return 0.0
    return numerator / denominator


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(q, 0.0), 1.0)
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compute_true_range(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    prev_close: float | None = None
    values: list[float] = []
    for row in rows:
        high = float(row["high"])
        low = float(row["low"])
        if prev_close is None:
            values.append(high - low)
        else:
            values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        prev_close = float(row["close"])
    return sum(values) / len(values)


def compute_event_rows(
    bars: pl.DataFrame,
    config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    parameter_grid = config["strategy"]["parameter_grid"]
    observation_windows = sorted(parameter_grid["observation_window_minutes"])
    horizons = sorted(parameter_grid["evaluation_horizon_minutes"])
    trailing_context_events = int(config["strategy"]["trailing_context_events"])
    max_obs = max(observation_windows)
    max_horizon = max(horizons)

    records = bars.select(
        ["timestamp", "session_date", "calendar_year", "open", "high", "low", "close", "volume"]
    ).to_dicts()
    idx_by_timestamp = {row["timestamp"]: idx for idx, row in enumerate(records)}

    cash_dates = set(build_cash_open_calendar(bars))
    macro_dates, macro_note = read_optional_macro_calendar(config)
    event_dates_by_family: dict[str, list[date]] = {
        "09:30": sorted(cash_dates),
        "08:30": sorted(macro_dates),
    }

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    trailing_context: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)

    for family, dates in [("08:30", event_dates_by_family["08:30"]), ("09:30", event_dates_by_family["09:30"])]:
        anchor_clock = time(8, 30) if family == "08:30" else time(9, 30)
        for event_date in dates:
            anchor_ts = datetime.combine(event_date, anchor_clock, tzinfo=NY_TZ)
            required = expected_range(anchor_ts, max_obs + max_horizon + 1)
            if any(ts not in idx_by_timestamp for ts in required):
                skipped.append(
                    {
                        "event_date": event_date.isoformat(),
                        "event_family": family,
                        "reason": "missing_required_minutes",
                    }
                )
                continue

            anchor_idx = idx_by_timestamp[anchor_ts]
            pre_rows = records[max(0, anchor_idx - 20) : anchor_idx]
            atr20 = compute_true_range(pre_rows)
            anchor_open = float(records[anchor_idx]["open"])

            for obs_minutes in observation_windows:
                slice_rows = observation_slice(records, anchor_idx, obs_minutes)
                decision_idx = anchor_idx + obs_minutes
                decision_row = records[decision_idx]
                decision_close = float(decision_row["close"])
                entry_idx = decision_idx + 1
                entry_row = records[entry_idx]
                impulse_move = decision_close - anchor_open
                impulse_sign = 1 if impulse_move > 0 else (-1 if impulse_move < 0 else 0)
                impulse_size_bp = 10000.0 * safe_div(abs(impulse_move), anchor_open)
                slice_high = max(float(row["high"]) for row in slice_rows)
                slice_low = min(float(row["low"]) for row in slice_rows)
                slice_range = max(slice_high - slice_low, EPSILON)
                event_volume = sum(float(row["volume"]) for row in slice_rows)
                path_length = abs(decision_close - anchor_open)
                directional_containment = safe_div(path_length, slice_range)
                stress_raw = safe_div(slice_range, max(atr20, config["execution"]["tick_size"]))

                trail_key = (family, obs_minutes)
                prior_context = trailing_context[trail_key][-trailing_context_events:]
                prior_ranges = sorted(context["slice_range"] for context in prior_context)
                prior_volumes = sorted(context["event_volume"] for context in prior_context)
                if prior_ranges:
                    dispersion_stress = safe_div(slice_range, median(prior_ranges))
                    volume_context = safe_div(event_volume, median(prior_volumes))
                else:
                    dispersion_stress = stress_raw
                    volume_context = 1.0

                row: dict[str, Any] = {
                    "event_date": event_date,
                    "calendar_year": int(records[anchor_idx]["calendar_year"]),
                    "event_family": family,
                    "anchor_ts": anchor_ts,
                    "decision_ts": decision_row["timestamp"],
                    "entry_ts": entry_row["timestamp"],
                    "anchor_idx": anchor_idx,
                    "decision_idx": decision_idx,
                    "entry_idx": entry_idx,
                    "observation_window_minutes": obs_minutes,
                    "atr20_pre_event": atr20,
                    "anchor_open": anchor_open,
                    "decision_close": decision_close,
                    "entry_open": float(entry_row["open"]),
                    "impulse_sign": impulse_sign,
                    "impulse_size_bp": impulse_size_bp,
                    "slice_range": slice_range,
                    "dispersion_stress_ratio": dispersion_stress,
                    "volume_rate_context_ratio": volume_context,
                    "impulse_containment_ratio": directional_containment,
                }
                for horizon in horizons:
                    exit_row = records[decision_idx + horizon]
                    continuation_bp = (
                        impulse_sign
                        * 10000.0
                        * safe_div(float(exit_row["close"]) - float(entry_row["open"]), float(entry_row["open"]))
                    )
                    row[f"continuation_return_bps_{horizon}"] = continuation_bp
                    row[f"trust_label_{horizon}"] = int(continuation_bp > 0.0)
                rows.append(row)
                trailing_context[trail_key].append(
                    {"slice_range": slice_range, "event_volume": event_volume}
                )

    metadata = {
        "macro_calendar_note": macro_note,
        "cash_open_events": len(event_dates_by_family["09:30"]),
        "macro_events": len(event_dates_by_family["08:30"]),
        "skipped_events": skipped,
    }
    if not rows:
        return pl.DataFrame(schema={"event_date": pl.Date}), metadata
    return pl.DataFrame(rows).sort(["event_family", "event_date", "observation_window_minutes"]), metadata


def make_walk_forward_windows(events: pl.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    unique_months = (
        events.select(pl.col("event_date").dt.strftime("%Y-%m").alias("year_month"))
        .unique()
        .sort("year_month")
        .get_column("year_month")
        .to_list()
    )
    train_months = int(config["strategy"]["train_months"])
    test_months = int(config["strategy"]["test_months"])
    windows: list[dict[str, Any]] = []
    start = 0
    split_id = 1
    while start + train_months + test_months <= len(unique_months):
        train_set = set(unique_months[start : start + train_months])
        test_set = set(unique_months[start + train_months : start + train_months + test_months])
        windows.append({"split_id": split_id, "train": train_set, "test": test_set})
        start += test_months
        split_id += 1
    return windows


def make_choices(config: dict[str, Any]) -> list[ParameterChoice]:
    grid = config["strategy"]["parameter_grid"]
    choices: list[ParameterChoice] = []
    for obs in grid["observation_window_minutes"]:
        for horizon in grid["evaluation_horizon_minutes"]:
            for baseline in grid["baseline_impulse_threshold_bps"]:
                for dispersion in grid["dispersion_stress_threshold_ratio"]:
                    for volume in grid["volume_rate_context_threshold_ratio"]:
                        for containment in grid["impulse_containment_threshold_ratio"]:
                            for mode in grid["event_family_mode"]:
                                choices.append(
                                    ParameterChoice(
                                        observation_window_minutes=int(obs),
                                        evaluation_horizon_minutes=int(horizon),
                                        baseline_impulse_threshold_bps=float(baseline),
                                        dispersion_stress_threshold_ratio=float(dispersion),
                                        volume_rate_context_threshold_ratio=float(volume),
                                        impulse_containment_threshold_ratio=float(containment),
                                        event_family_mode=str(mode),
                                    )
                                )
    return choices


def filter_for_choice(events: pl.DataFrame, choice: ParameterChoice) -> pl.DataFrame:
    scoped = events.filter(pl.col("observation_window_minutes") == choice.observation_window_minutes)
    if choice.event_family_mode == "separate":
        return scoped
    return scoped.filter(pl.col("event_family") == choice.event_family_mode)


def family_groups(choice: ParameterChoice, scoped: pl.DataFrame) -> list[str]:
    if choice.event_family_mode == "separate":
        return sorted(scoped.get_column("event_family").unique().to_list())
    return [choice.event_family_mode]


def rule_predictions(events: pl.DataFrame, choice: ParameterChoice, rule_name: str) -> pl.DataFrame:
    if events.is_empty():
        return events

    parts: list[pl.DataFrame] = []
    for family in family_groups(choice, events):
        subset = events.filter(pl.col("event_family") == family)
        if subset.is_empty():
            continue
        baseline_pass = pl.col("impulse_size_bp") >= choice.baseline_impulse_threshold_bps
        if rule_name == "baseline":
            scored = subset.with_columns(
                baseline_pass.cast(pl.Int64).alias("selected"),
                pl.lit(rule_name).alias("rule_name"),
                pl.lit(choice.event_family_mode).alias("event_family_mode"),
            )
        else:
            scored = subset.with_columns(
                (
                    baseline_pass
                    & (pl.col("dispersion_stress_ratio") <= choice.dispersion_stress_threshold_ratio)
                    & (pl.col("volume_rate_context_ratio") >= choice.volume_rate_context_threshold_ratio)
                    & (pl.col("impulse_containment_ratio") >= choice.impulse_containment_threshold_ratio)
                )
                .cast(pl.Int64)
                .alias("selected"),
                pl.lit(rule_name).alias("rule_name"),
                pl.lit(choice.event_family_mode).alias("event_family_mode"),
            )
        parts.append(scored)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="vertical_relaxed")


def classification_score(events: pl.DataFrame, choice: ParameterChoice, rule_name: str) -> tuple[float, float, int]:
    scored = rule_predictions(events, choice, rule_name)
    if scored.is_empty():
        return (-1.0, float("-inf"), 0)
    label_col = f"trust_label_{choice.evaluation_horizon_minutes}"
    selected = scored.filter(pl.col("selected") == 1)
    if selected.is_empty():
        return (-1.0, float("-inf"), 0)
    accuracy = float((selected.get_column(label_col).mean()))
    average_return = float(selected.get_column(f"continuation_return_bps_{choice.evaluation_horizon_minutes}").mean())
    return (accuracy, average_return, selected.height)


def choose_best(events: pl.DataFrame, choices: list[ParameterChoice], rule_name: str) -> ParameterChoice:
    scoped_choices = [choice for choice in choices if not filter_for_choice(events, choice).is_empty()]
    best = scoped_choices[0]
    best_score = classification_score(filter_for_choice(events, best), best, rule_name)
    for choice in scoped_choices[1:]:
        score = classification_score(filter_for_choice(events, choice), choice, rule_name)
        if score > best_score:
            best = choice
            best_score = score
    return best


def simulate_trades(
    scored: pl.DataFrame,
    choice: ParameterChoice,
    bars: pl.DataFrame,
    config: dict[str, Any],
    split_id: int,
) -> pl.DataFrame:
    selected = scored.filter(pl.col("selected") == 1)
    if selected.is_empty():
        return pl.DataFrame()

    records = bars.select(["timestamp", "open", "high", "low", "close"]).to_dicts()
    tick_size = float(config["execution"]["tick_size"])
    point_value = float(config["execution"]["point_value"])
    slippage_ticks = float(config["execution"]["slippage_ticks_per_side"])
    commissions = float(config["execution"]["commissions_per_side_usd"])
    label_col = f"continuation_return_bps_{choice.evaluation_horizon_minutes}"

    rows: list[dict[str, Any]] = []
    for event in selected.iter_rows(named=True):
        entry_idx = int(event["entry_idx"])
        decision_idx = int(event["decision_idx"])
        exit_idx = decision_idx + choice.evaluation_horizon_minutes
        direction = int(event["impulse_sign"])
        if direction == 0:
            continue
        entry_raw = float(records[entry_idx]["open"])
        exit_raw = float(records[exit_idx]["close"])
        slippage_points = slippage_ticks * tick_size
        entry_price = entry_raw + slippage_points if direction > 0 else entry_raw - slippage_points
        exit_price = exit_raw - slippage_points if direction > 0 else exit_raw + slippage_points
        gross_points = direction * (exit_price - entry_price)
        gross_return_bp = 10000.0 * safe_div(gross_points, entry_price)
        net_pnl_usd = gross_points * point_value - (2.0 * commissions)
        rows.append(
            {
                "split_id": split_id,
                "event_date": event["event_date"],
                "event_family": event["event_family"],
                "rule_name": event["rule_name"],
                "event_family_mode": choice.event_family_mode,
                "observation_window_minutes": choice.observation_window_minutes,
                "evaluation_horizon_minutes": choice.evaluation_horizon_minutes,
                "entry_ts": records[entry_idx]["timestamp"],
                "exit_ts": records[exit_idx]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return_bp": gross_return_bp,
                "continuation_return_bps": float(event[label_col]),
                "net_pnl_usd": net_pnl_usd,
            }
        )
    return pl.DataFrame(rows).sort(["event_date", "rule_name"])


def summarize_rule(trades: pl.DataFrame, scored: pl.DataFrame, choice: ParameterChoice) -> dict[str, Any]:
    label_col = f"trust_label_{choice.evaluation_horizon_minutes}"
    selected = scored.filter(pl.col("selected") == 1)
    if selected.is_empty():
        return {
            "selected_events": 0,
            "classification_accuracy": None,
            "continuation_return_bps_mean": None,
            "average_trade_pnl_usd": None,
        }
    return {
        "selected_events": selected.height,
        "classification_accuracy": float(selected.get_column(label_col).mean()),
        "continuation_return_bps_mean": float(
            selected.get_column(f"continuation_return_bps_{choice.evaluation_horizon_minutes}").mean()
        ),
        "average_trade_pnl_usd": float(trades.get_column("net_pnl_usd").mean()) if not trades.is_empty() else None,
    }


def fold_month_key(events: pl.DataFrame) -> pl.DataFrame:
    return events.with_columns(pl.col("event_date").dt.strftime("%Y-%m").alias("year_month"))


def build_equity_curve(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame()
    return (
        trades.sort(["rule_name", "exit_ts"])
        .with_columns(pl.col("net_pnl_usd").cum_sum().over("rule_name").alias("equity_usd"))
        .select(["rule_name", "exit_ts", "net_pnl_usd", "equity_usd"])
    )


def build_report_lines(
    strategy_id: str,
    contract_path: str,
    chosen_action: str,
    review_summary: list[dict[str, str]],
    decision_text: str,
    validation_text: str,
    experiments: list[dict[str, str]],
    approach: str,
    run_text: str,
    next_actions: list[str],
) -> str:
    lines = [
        "# Strategy Contract Review & Backtesting Brief",
        "",
        "## 1. Contract Review Summary",
        "",
        "| Source | Key Insight | Relevance |",
        "| --- | --- | --- |",
    ]
    for row in review_summary:
        lines.append(f"| {row['source']} | {row['insight']} | {row['relevance']} |")
    lines.extend(
        [
            "",
            "## 2. Strategy Interpretation",
            "",
            f"- Contract file: `{contract_path}`",
            f"- Strategy ID: `{strategy_id}`",
            "- Core thesis: A public proxy stack built from dispersion stress, relative volume, and impulse containment should identify more trustworthy continuation windows than a volatility-only gate.",
            "- Assumptions: The first executable pass uses continuation direction, 3- or 5-minute observation slices, 5/10/15-minute horizons, and separate or family-specific evaluation of 08:30 and 09:30 windows.",
            "- Required inputs and outputs: ES 1-minute R2 bars, an optional real 08:30 macro calendar, deterministic 09:30 cash-open dates from the bar set, and the standard trades/equity/metrics/diagnostics/report outputs.",
            "- Key constraints: The source contract leaves direction, exact exit, and production thresholds unresolved, so this package freezes only the minimum necessary assumptions and reports them explicitly.",
            "",
            "## 3. Build / Skip / Repair Decision",
            "",
            f"- Chosen action: `{chosen_action}`",
            f"- Decision evidence: {decision_text}",
            "",
            "## 4. Validation Assessment",
            "",
            validation_text,
            "",
            "## 5. Added Experiments",
            "",
        ]
    )
    for experiment in experiments:
        lines.extend(
            [
                f"- Name: {experiment['name']}",
                f"  Purpose: {experiment['purpose']}",
                f"  Assumption being tested: {experiment['assumption']}",
                f"  Required data or inputs: {experiment['required_data']}",
                f"  Success metric: {experiment['success_metric']}",
                f"  Failure metric: {experiment['failure_metric']}",
                f"  Decision rule: {experiment['decision_rule']}",
                f"  Key limitation: {experiment['limitation']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 6. Backtesting Approach",
            "",
            approach,
            "",
            "## 7. Backtesting Code or Repair Output",
            "",
            run_text,
            "",
            "## 8. Immediate Next Actions",
            "",
        ]
    )
    for step in next_actions:
        lines.append(f"- {step}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_market_bars(config)
    events, metadata = compute_event_rows(bars, config)
    if events.is_empty():
        raise RuntimeError("No benchmark-window events were available after feature extraction.")

    month_keyed = fold_month_key(events)
    windows = make_walk_forward_windows(month_keyed, config)
    choices = make_choices(config)

    all_scored_parts: list[pl.DataFrame] = []
    all_trades_parts: list[pl.DataFrame] = []
    chosen_parameters: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []

    for window in windows:
        train = month_keyed.filter(pl.col("year_month").is_in(window["train"]))
        test = month_keyed.filter(pl.col("year_month").is_in(window["test"]))
        if train.is_empty() or test.is_empty():
            continue
        for rule_name in ["baseline", "treatment"]:
            best_choice = choose_best(train, choices, rule_name)
            train_scored = rule_predictions(filter_for_choice(train, best_choice), best_choice, rule_name)
            test_scored = rule_predictions(filter_for_choice(test, best_choice), best_choice, rule_name)
            if test_scored.is_empty():
                continue
            trades = simulate_trades(test_scored, best_choice, bars, config, window["split_id"])
            summary = summarize_rule(trades, test_scored, best_choice)
            all_scored_parts.append(
                test_scored.with_columns(
                    pl.lit(window["split_id"]).alias("split_id"),
                    pl.lit(best_choice.evaluation_horizon_minutes).alias("evaluation_horizon_minutes_used"),
                )
            )
            if not trades.is_empty():
                all_trades_parts.append(trades)
            chosen_parameters.append(
                {
                    "split_id": window["split_id"],
                    "rule_name": rule_name,
                    "observation_window_minutes": best_choice.observation_window_minutes,
                    "evaluation_horizon_minutes": best_choice.evaluation_horizon_minutes,
                    "baseline_impulse_threshold_bps": best_choice.baseline_impulse_threshold_bps,
                    "dispersion_stress_threshold_ratio": best_choice.dispersion_stress_threshold_ratio,
                    "volume_rate_context_threshold_ratio": best_choice.volume_rate_context_threshold_ratio,
                    "impulse_containment_threshold_ratio": best_choice.impulse_containment_threshold_ratio,
                    "event_family_mode": best_choice.event_family_mode,
                }
            )
            fold_metrics.append({"split_id": window["split_id"], "rule_name": rule_name, **summary})

    diagnostics = (
        pl.concat(all_scored_parts, how="vertical_relaxed").sort(["event_date", "rule_name", "split_id"])
        if all_scored_parts
        else pl.DataFrame()
    )
    trades = pl.concat(all_trades_parts, how="vertical_relaxed") if all_trades_parts else pl.DataFrame()
    equity_curve = build_equity_curve(trades)

    overall: dict[str, Any] = {}
    for rule_name in ["baseline", "treatment"]:
        rule_scored = diagnostics.filter(pl.col("rule_name") == rule_name) if not diagnostics.is_empty() else pl.DataFrame()
        rule_trades = trades.filter(pl.col("rule_name") == rule_name) if not trades.is_empty() else pl.DataFrame()
        if rule_scored.is_empty():
            overall[rule_name] = {
                "selected_events": 0,
                "classification_accuracy": None,
                "continuation_return_bps_mean": None,
                "average_trade_pnl_usd": None,
            }
            continue
        used_horizon = int(rule_scored.get_column("evaluation_horizon_minutes_used")[0])
        label_col = f"trust_label_{used_horizon}"
        selected = rule_scored.filter(pl.col("selected") == 1)
        overall[rule_name] = {
            "selected_events": selected.height,
            "classification_accuracy": float(selected.get_column(label_col).mean()) if not selected.is_empty() else None,
            "continuation_return_bps_mean": float(
                selected.get_column(f"continuation_return_bps_{used_horizon}").mean()
            )
            if not selected.is_empty()
            else None,
            "average_trade_pnl_usd": float(rule_trades.get_column("net_pnl_usd").mean())
            if not rule_trades.is_empty()
            else None,
        }

    treatment_accuracy = overall["treatment"]["classification_accuracy"]
    baseline_accuracy = overall["baseline"]["classification_accuracy"]
    comparison = {
        "treatment_minus_baseline_accuracy": (
            float(treatment_accuracy - baseline_accuracy)
            if treatment_accuracy is not None and baseline_accuracy is not None
            else None
        )
    }

    review_summary = [
        {
            "source": "Canonical contract",
            "insight": "Defines a trust-classification problem across tagged 08:30 and 09:30 benchmark windows.",
            "relevance": "Primary source of scope and metrics.",
        },
        {
            "source": "Backtest ledger",
            "insight": "Shows completed equivalents only for MSCI review bridge and ES replenishment fragility.",
            "relevance": "Confirms this strategy is materially new.",
        },
        {
            "source": "Data contract and R2 manifest",
            "insight": "Confirm the shared R2 loader pattern and the ES 1-minute parquet key.",
            "relevance": "Makes the package runnable against the existing data path.",
        },
    ]
    experiments = [
        {
            "name": "Observation And Horizon Stability",
            "purpose": "Check whether the proxy edge is still present when timing moves slightly.",
            "assumption": "The edge is not just a single lucky slice-length and exit-pair artifact.",
            "required_data": "ES 1-minute bars plus the event-family calendars already required by the contract.",
            "success_metric": "Treatment uplift stays directionally positive across neighboring observation and horizon settings.",
            "failure_metric": "Lift disappears once timing moves away from one narrow setting.",
            "decision_rule": "Reject any timing choice that cannot survive adjacent timing variants.",
            "limitation": "Minute bars still hide sub-minute execution differences.",
        },
        {
            "name": "Incremental Value Versus Volatility",
            "purpose": "Test whether the proxy stack adds information beyond a simple impulse-size gate.",
            "assumption": "Dispersion stress, relative volume, and containment are not just restating volatility.",
            "required_data": "The same event set and market bars used for the baseline.",
            "success_metric": "Treatment minus baseline classification accuracy is positive out of sample.",
            "failure_metric": "Treatment matches or trails baseline despite greater complexity.",
            "decision_rule": "Drop the proxy layer if walk-forward uplift is non-positive.",
            "limitation": "A positive result is still tied to the first-pass continuation framing.",
        },
        {
            "name": "Window Family Stability",
            "purpose": "Verify whether 08:30 and 09:30 belong in the same strategy lane.",
            "assumption": "The mechanism behaves consistently across macro-release and cash-open families.",
            "required_data": "A real 08:30 macro calendar plus the deterministic 09:30 family.",
            "success_metric": "Family-level uplift has the same sign and similar magnitude.",
            "failure_metric": "One family works and the other does not.",
            "decision_rule": "Split the strategy by family if the signs conflict.",
            "limitation": "Family imbalance can distort confidence if one side has far fewer usable events.",
        },
    ]

    report_text = build_report_lines(
        strategy_id=config["strategy"]["id"],
        contract_path=config["runtime"]["contract_path"],
        chosen_action="build_new",
        review_summary=review_summary,
        decision_text=(
            "No equivalent generated package exists for "
            "`es_benchmark_window_replenishment_failure_proxy_stack_20260503`, while the existing completed "
            "packages map to different strategies. The contract is new and materially distinct from the completed "
            "ES replenishment-fragility implementation because it frames a benchmark-window trust classifier with "
            "08:30/09:30 family diagnostics instead of the prior replenishment fragility trade package."
        ),
        validation_text=(
            "The contract is sufficient for a first runnable backtest only if the implementation makes its unresolved "
            "execution choices explicit. The added experiments are therefore the minimum useful set needed to test "
            "whether the proxy stack is real, stable across nearby timing choices, and consistent across event families."
        ),
        experiments=experiments,
        approach=(
            "The backtest loads the confirmed ES 1-minute R2 dataset, derives deterministic 09:30 cash-open events "
            "from the market-data dates, optionally loads a real 08:30 macro-release calendar, computes observation-slice "
            "features for 3- and 5-minute windows, labels continuation over 5/10/15-minute horizons, and fits a "
            "volatility-only baseline against a replenishment-proxy treatment in rolling month-based walk-forward splits."
        ),
        run_text=(
            "Run `python backtests/generated/es_benchmark_window_replenishment_failure_proxy_stack_20260503/backtest.py "
            "--config backtests/generated/es_benchmark_window_replenishment_failure_proxy_stack_20260503/config.yaml` "
            "from the repo root. If the macro calendar file is absent, the package skips the 08:30 family and records "
            "that gap in `metrics.json` and `report.md` rather than inventing release dates."
        ),
        next_actions=[
            "Add a real public 08:30 macro-release calendar at the configured package path to enable the macro family.",
            "Run the package and inspect `metrics.json` before treating any threshold choice as meaningful.",
            "Compare treatment uplift separately for `08:30` and `09:30` before pooling conclusions.",
            "Promote only the timing and threshold settings that remain stable across neighboring variants.",
            "Reject the proxy stack quickly if it cannot beat the volatility-only baseline out of sample.",
        ],
    )

    metrics = {
        "strategy_id": config["strategy"]["id"],
        "contract_path": config["runtime"]["contract_path"],
        "overall": overall,
        "comparison": comparison,
        "walk_forward_folds": fold_metrics,
        "chosen_parameters": chosen_parameters,
        "macro_calendar": {
            "available": metadata["macro_calendar_note"] is None,
            "note": metadata["macro_calendar_note"],
        },
        "event_inventory": {
            "cash_open_events": metadata["cash_open_events"],
            "macro_events": metadata["macro_events"],
            "skipped_events": metadata["skipped_events"],
        },
        "implementation_notes": [
            "This first-pass implementation uses continuation direction rather than fade logic.",
            "The 09:30 cash-open family is deterministic from the bar set; the 08:30 macro family requires a real public calendar.",
            "Thresholds are chosen through walk-forward selection from the minimal contract grid rather than hard-coded conclusions.",
        ],
    }

    diagnostics.write_parquet(output_dir / config["outputs"]["diagnostics"])
    trades.write_parquet(output_dir / config["outputs"]["trades"])
    equity_curve.write_parquet(output_dir / config["outputs"]["equity_curve"])
    (output_dir / config["outputs"]["report"]).write_text(report_text, encoding="utf-8")
    (output_dir / config["outputs"]["metrics"]).write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
