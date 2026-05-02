#!/usr/bin/env python3
"""Backtest for the ES replenishment fragility strategy contract.

This implementation follows the contract in
`strategy-contracts/es-replenishment-fragility/2026-05-02-strategy_contract.yaml`
and uses the shared R2 loader from `backtests.lib.r2_data`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable
from zoneinfo import ZoneInfo

try:
    import polars as pl
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This backtest requires the `polars` package. Install project dependencies "
        "before running it."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This backtest requires the `PyYAML` package. Install project dependencies "
        "before running it."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns

NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EventDefinition:
    name: str
    anchor: time
    decision: time


EVENT_DEFINITIONS = {
    "macro_release_0830": EventDefinition(
        name="macro_release_0830",
        anchor=time(8, 30),
        decision=time(8, 32),
    ),
    "cash_open_0930": EventDefinition(
        name="cash_open_0930",
        anchor=time(9, 30),
        decision=time(9, 32),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_timestamp_column(bars: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    timezone_name = config.get("runtime", {}).get("timezone", "America/New_York")
    timestamp_dtype = bars.schema.get("timestamp")
    expr = pl.col("timestamp")

    if timestamp_dtype == pl.Utf8:
        expr = expr.str.to_datetime(strict=False)

    if getattr(timestamp_dtype, "time_zone", None):
        expr = expr.dt.convert_time_zone(timezone_name)
    else:
        expr = expr.dt.replace_time_zone(timezone_name)

    return (
        bars.with_columns(expr.alias("timestamp"))
        .sort("timestamp")
        .with_columns(
            pl.col("timestamp").dt.date().alias("trade_date"),
            pl.col("timestamp").dt.year().alias("calendar_year"),
        )
    )


def read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported calendar file format: {path}")


def parse_date_column(df: pl.DataFrame, column: str) -> pl.Series:
    series = df.get_column(column)
    if series.dtype == pl.Date:
        return series
    if series.dtype == pl.Utf8:
        return series.str.strptime(pl.Date, strict=False)
    return series.cast(pl.Date)


def build_event_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    calendar_cfg = config["event_calendars"]

    macro_df = read_table(Path(calendar_cfg["macro_release_calendar"]["path"]))
    macro_date_col = calendar_cfg["macro_release_calendar"].get("date_column", "event_date")
    macro_dates = parse_date_column(macro_df, macro_date_col).drop_nulls().to_list()

    cash_df = read_table(Path(calendar_cfg["cash_open_calendar"]["path"]))
    cash_date_col = calendar_cfg["cash_open_calendar"].get("date_column", "event_date")
    cash_dates = parse_date_column(cash_df, cash_date_col).drop_nulls().to_list()

    events: list[dict[str, Any]] = []
    for event_date in macro_dates:
        events.append(
            {
                "event_window_name": "macro_release_0830",
                "event_date": event_date,
            }
        )
    for event_date in cash_dates:
        events.append(
            {
                "event_window_name": "cash_open_0930",
                "event_date": event_date,
            }
        )

    events.sort(key=lambda row: (row["event_date"], row["event_window_name"]))
    return events


def expected_minute_range(start: datetime, end: datetime) -> list[datetime]:
    current = start
    out: list[datetime] = []
    while current <= end:
        out.append(current)
        current += timedelta(minutes=1)
    return out


def compute_true_range(high: float, low: float, prev_close: float | None) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def safe_div(numerator: float, denominator: float, floor: float = 1e-9) -> float:
    return numerator / max(abs(denominator), floor)


def median_or_none(values: Iterable[float]) -> float | None:
    values = [value for value in values if value is not None and not math.isnan(value)]
    return median(values) if values else None


def extract_event_features(
    bars: pl.DataFrame,
    event_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    max_exit_horizon = max(config["strategy"]["parameter_grid"]["exit_horizon_minutes"])

    records = bars.select(
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_date",
            "calendar_year",
        ]
    ).to_dicts()
    ts_to_idx = {row["timestamp"]: index for index, row in enumerate(records)}

    raw_events: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for event in event_rows:
        definition = EVENT_DEFINITIONS[event["event_window_name"]]
        anchor_dt = datetime.combine(event["event_date"], definition.anchor, tzinfo=NY_TZ)
        decision_dt = datetime.combine(event["event_date"], definition.decision, tzinfo=NY_TZ)
        entry_dt = decision_dt + timedelta(minutes=1)
        exit_dt = anchor_dt + timedelta(minutes=max_exit_horizon)
        pre_event_start = anchor_dt - timedelta(minutes=20)

        required_timestamps = expected_minute_range(pre_event_start, exit_dt)
        missing = [ts for ts in required_timestamps if ts not in ts_to_idx]
        if missing:
            dropped.append(
                {
                    "event_date": event["event_date"],
                    "event_window_name": event["event_window_name"],
                    "reason": "missing_required_bars",
                    "missing_count": len(missing),
                }
            )
            continue

        anchor_idx = ts_to_idx[anchor_dt]
        decision_idx = ts_to_idx[decision_dt]
        entry_idx = ts_to_idx[entry_dt]
        exit_idx = ts_to_idx[exit_dt]

        pre_event_rows = records[anchor_idx - 20 : anchor_idx]
        prev_close = None
        true_ranges: list[float] = []
        for row in pre_event_rows:
            true_ranges.append(compute_true_range(row["high"], row["low"], prev_close))
            prev_close = row["close"]
        atr20 = sum(true_ranges) / len(true_ranges)

        event_window_rows = records[anchor_idx : decision_idx + 1]
        open_anchor = records[anchor_idx]["open"]
        close_decision = records[decision_idx]["close"]
        impulse_return = (close_decision - open_anchor) / open_anchor
        sign_impulse = 0 if impulse_return == 0 else (1 if impulse_return > 0 else -1)
        window_high = max(row["high"] for row in event_window_rows)
        window_low = min(row["low"] for row in event_window_rows)
        window_range = window_high - window_low
        event_volume = sum(float(row["volume"]) for row in event_window_rows)
        dispersion_stress = safe_div(window_high - window_low, max(atr20, 0.25))
        containment_score = safe_div(abs(close_decision - open_anchor), max(window_range, 0.25))
        continuation_return = (
            sign_impulse * ((records[exit_idx]["close"] - close_decision) / close_decision)
            if sign_impulse
            else 0.0
        )
        trustworthy_label = 1 if continuation_return > 0 else 0

        raw_events.append(
            {
                "event_date": event["event_date"],
                "event_window_name": event["event_window_name"],
                "anchor_ts": anchor_dt,
                "decision_ts": decision_dt,
                "entry_ts": entry_dt,
                "max_exit_ts": exit_dt,
                "entry_idx": entry_idx,
                "anchor_idx": anchor_idx,
                "decision_idx": decision_idx,
                "exit_idx_max": exit_idx,
                "open_anchor": open_anchor,
                "close_decision": close_decision,
                "impulse_return_0_2": impulse_return,
                "sign_impulse": sign_impulse,
                "atr20_pre_event": atr20,
                "event_window_range": window_range,
                "event_window_volume": event_volume,
                "dispersion_stress_0_2": dispersion_stress,
                "containment_score_0_2": containment_score,
                "continuation_return_2_15": continuation_return,
                "trustworthy_continuation_label": trustworthy_label,
                "calendar_year": event["event_date"].year,
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in raw_events:
        by_family.setdefault(row["event_window_name"], []).append(row)

    enriched: list[dict[str, Any]] = []
    for family_rows in by_family.values():
        family_rows.sort(key=lambda row: row["event_date"])
        prior_volumes: list[float] = []
        for row in family_rows:
            volume_baseline = median_or_none(prior_volumes[-20:])
            row["volume_rate_ratio_0_2"] = (
                safe_div(row["event_window_volume"], volume_baseline)
                if volume_baseline
                else None
            )
            row["pre_event_atr20_tercile"] = None
            enriched.append(row)
            prior_volumes.append(row["event_window_volume"])

    feature_df = pl.DataFrame(enriched).sort(["event_date", "event_window_name"])
    if feature_df.is_empty():
        return feature_df, dropped

    atr_rank = (
        feature_df.with_columns(
            pl.col("atr20_pre_event")
            .rank(method="ordinal")
            .over("event_window_name")
            .alias("atr_rank"),
            pl.len().over("event_window_name").alias("family_count"),
        )
        .with_columns(
            (
                (
                    (pl.col("atr_rank") - 1)
                    * 3
                    / pl.when(pl.col("family_count") > 0)
                    .then(pl.col("family_count"))
                    .otherwise(1)
                )
                .floor()
                .clip(0, 2)
                + 1
            )
            .cast(pl.Int64)
            .alias("pre_event_atr20_tercile")
        )
        .drop(["atr_rank", "family_count"])
    )
    return atr_rank, dropped


def simulate_trade(
    event: dict[str, Any],
    bars_by_index: list[dict[str, Any]],
    exit_horizon_minutes: int,
    stop_range_multiple: float,
    point_value: float,
    tick_size: float,
    commission_per_side: float,
) -> dict[str, Any]:
    direction = int(event["sign_impulse"])
    exit_idx = event["entry_idx"] + max((exit_horizon_minutes - 3), 0)
    if exit_idx >= len(bars_by_index):
        exit_idx = len(bars_by_index) - 1

    entry_row = bars_by_index[event["entry_idx"]]
    entry_raw = float(entry_row["open"])
    if direction > 0:
        entry_price = entry_raw + tick_size
    else:
        entry_price = entry_raw - tick_size

    stop_distance = stop_range_multiple * float(event["event_window_range"])
    stop_hit = False
    exit_price = float(bars_by_index[exit_idx]["close"])
    exit_reason = "time_exit"
    exit_time = bars_by_index[exit_idx]["timestamp"]
    max_adverse = 0.0
    max_favorable = 0.0

    for row in bars_by_index[event["entry_idx"] : exit_idx + 1]:
        high = float(row["high"])
        low = float(row["low"])
        if direction > 0:
            adverse = max(0.0, entry_price - low)
            favorable = max(0.0, high - entry_price)
            stop_price = entry_price - stop_distance
            if low <= stop_price:
                stop_hit = True
                exit_reason = "stop_hit"
                exit_time = row["timestamp"]
                exit_price = stop_price - tick_size
                break
        else:
            adverse = max(0.0, high - entry_price)
            favorable = max(0.0, entry_price - low)
            stop_price = entry_price + stop_distance
            if high >= stop_price:
                stop_hit = True
                exit_reason = "stop_hit"
                exit_time = row["timestamp"]
                exit_price = stop_price + tick_size
                break
        max_adverse = max(max_adverse, adverse)
        max_favorable = max(max_favorable, favorable)

    if not stop_hit:
        if direction > 0:
            exit_price = exit_price - tick_size
        else:
            exit_price = exit_price + tick_size

    gross_points = direction * (exit_price - entry_price)
    net_pnl_usd = gross_points * point_value - (2.0 * commission_per_side)
    return {
        "entry_ts": entry_row["timestamp"],
        "exit_ts": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_points": gross_points,
        "net_pnl_usd": net_pnl_usd,
        "stop_hit": stop_hit,
        "exit_reason": exit_reason,
        "max_adverse_excursion_points": max_adverse,
        "max_favorable_excursion_points": max_favorable,
    }


def cartesian_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(grid)
    values = [grid[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def evaluate_rule(
    feature_df: pl.DataFrame,
    bars_by_index: list[dict[str, Any]],
    rule_name: str,
    params: dict[str, Any],
    execution_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for event in feature_df.to_dicts():
        impulse_abs = abs(float(event["impulse_return_0_2"]))
        atr_gate = params["impulse_atr_threshold"] * float(event["atr20_pre_event"]) / float(
            event["open_anchor"]
        )
        baseline_pass = event["sign_impulse"] != 0 and impulse_abs >= atr_gate

        volume_ratio = event["volume_rate_ratio_0_2"]
        fragility_flag = None
        if volume_ratio is not None:
            fragility_flag = int(
                float(event["dispersion_stress_0_2"]) > params["dispersion_max"]
                or float(volume_ratio) < params["volume_min"]
                or float(event["containment_score_0_2"]) < params["containment_min"]
            )

        take_trade = baseline_pass
        if rule_name == "treatment":
            take_trade = baseline_pass and volume_ratio is not None and fragility_flag == 0

        diagnostics.append(
            {
                "event_date": event["event_date"],
                "event_window_name": event["event_window_name"],
                "rule": rule_name,
                "impulse_pass": baseline_pass,
                "fragility_flag": fragility_flag,
                "take_trade": take_trade,
                "trustworthy_continuation_label": event["trustworthy_continuation_label"],
            }
        )

        if not take_trade:
            continue

        trade = simulate_trade(
            event=event,
            bars_by_index=bars_by_index,
            exit_horizon_minutes=int(params["exit_horizon_minutes"]),
            stop_range_multiple=float(params["stop_range_multiple"]),
            point_value=float(execution_cfg["point_value"]),
            tick_size=float(execution_cfg["tick_size"]),
            commission_per_side=float(execution_cfg["commissions"]),
        )
        trade.update(
            {
                "event_date": event["event_date"],
                "event_window_name": event["event_window_name"],
                "calendar_year": event["calendar_year"],
                "rule": rule_name,
                "direction": int(event["sign_impulse"]),
                "impulse_return_0_2": float(event["impulse_return_0_2"]),
                "atr20_pre_event": float(event["atr20_pre_event"]),
                "dispersion_stress_0_2": float(event["dispersion_stress_0_2"]),
                "volume_rate_ratio_0_2": volume_ratio,
                "containment_score_0_2": float(event["containment_score_0_2"]),
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        trades.append(trade)

    metrics = summarize_rule_metrics(trades)
    confusion = confusion_from_diagnostics(diagnostics)
    metrics["confusion_matrix"] = confusion
    metrics["rule"] = rule_name
    return trades, metrics


def confusion_from_diagnostics(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = bool(row["take_trade"])
        actual = bool(row["trustworthy_continuation_label"])
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def summarize_rule_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "expectancy_per_trade": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    pnls = [float(trade["net_pnl_usd"]) for trade in trades]
    gross_profit = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = sum(pnl for pnl in pnls if pnl < 0)
    expectancy = sum(pnls) / len(pnls)
    hit_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
    mean = expectancy
    variance = sum((pnl - mean) ** 2 for pnl in pnls) / len(pnls)
    stddev = math.sqrt(variance)
    sharpe_like = mean / stddev if stddev else 0.0

    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        running += pnl
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)

    return {
        "expectancy_per_trade": expectancy,
        "hit_rate": hit_rate,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else float("inf"),
        "sharpe_like": sharpe_like,
        "trade_count": len(trades),
        "max_drawdown": abs(max_drawdown),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def build_folds(feature_df: pl.DataFrame) -> list[dict[str, Any]]:
    years = sorted(feature_df.get_column("calendar_year").unique().to_list())
    if not years:
        return []

    min_year = years[0]
    max_year = years[-1]
    folds: list[dict[str, Any]] = []
    train_start = min_year
    train_end = train_start + 2
    test_year = train_end + 1

    while test_year <= max_year:
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_year,
                "test_end": test_year,
                "fold_name": f"{train_start}-{train_end}_to_{test_year}",
            }
        )
        train_end += 1
        test_year += 1

    if max_year >= 2024:
        folds.append(
            {
                "train_start": min_year,
                "train_end": 2023,
                "test_start": 2024,
                "test_end": max_year,
                "fold_name": "holdout_2024_onward",
            }
        )
    return folds


def filter_years(feature_df: pl.DataFrame, start_year: int, end_year: int) -> pl.DataFrame:
    return feature_df.filter(
        (pl.col("calendar_year") >= start_year) & (pl.col("calendar_year") <= end_year)
    )


def select_best_params(
    feature_df: pl.DataFrame,
    bars_by_index: list[dict[str, Any]],
    rule_name: str,
    search_space: list[dict[str, Any]],
    execution_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None

    for params in search_space:
        _, metrics = evaluate_rule(feature_df, bars_by_index, rule_name, params, execution_cfg)
        score = (
            float(metrics["expectancy_per_trade"]),
            float(metrics["profit_factor"]) if math.isfinite(metrics["profit_factor"]) else 999.0,
            float(metrics["trade_count"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError(f"Unable to select parameters for {rule_name}")
    return best_params, best_metrics


def compute_family_metrics(trades_df: pl.DataFrame) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if trades_df.is_empty():
        return results

    for (rule, family), subset in trades_df.group_by(["rule", "event_window_name"]):
        metrics = summarize_rule_metrics(subset.to_dicts())
        metrics["rule"] = rule
        metrics["event_window_name"] = family
        results.append(metrics)
    return results


def build_equity_curve(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame(
            {"rule": [], "exit_ts": [], "net_pnl_usd": [], "equity_usd": []},
            schema={
                "rule": pl.Utf8,
                "exit_ts": pl.Datetime(time_zone="America/New_York"),
                "net_pnl_usd": pl.Float64,
                "equity_usd": pl.Float64,
            },
        )

    pieces: list[pl.DataFrame] = []
    for rule in trades_df.get_column("rule").unique().to_list():
        subset = trades_df.filter(pl.col("rule") == rule).sort("exit_ts")
        pieces.append(
            subset.with_columns(pl.col("net_pnl_usd").cum_sum().alias("equity_usd")).select(
                ["rule", "exit_ts", "net_pnl_usd", "equity_usd"]
            )
        )
    return pl.concat(pieces, how="vertical_relaxed")


def write_report(
    output_dir: Path,
    strategy_id: str,
    metrics: dict[str, Any],
    chosen_params: list[dict[str, Any]],
    dropped_df: pl.DataFrame,
) -> None:
    lines = [
        f"# {strategy_id}",
        "",
        "## Summary",
        "",
        f"- Baseline expectancy per trade: {metrics['overall']['baseline']['expectancy_per_trade']:.2f} USD",
        f"- Treatment expectancy per trade: {metrics['overall']['treatment']['expectancy_per_trade']:.2f} USD",
        f"- Baseline trade count: {metrics['overall']['baseline']['trade_count']}",
        f"- Treatment trade count: {metrics['overall']['treatment']['trade_count']}",
        "",
        "## Parameter Winners By Fold",
        "",
    ]

    for row in chosen_params:
        lines.append(
            f"- {row['fold_name']} | {row['rule']} | train={row['train_window']} | "
            f"test={row['test_window']} | params={row['params_json']}"
        )

    if dropped_df.height:
        lines.extend(["", "## Dropped Events", ""])
        for row in dropped_df.to_dicts():
            lines.append(
                f"- {row['event_window_name']} on {row['event_date']}: {row['reason']} "
                f"({row['missing_count']} missing bars)"
            )

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_output_dir(Path(config["runtime"]["output_dir"]))
    strategy_cfg = config["strategy"]

    bars = normalize_ohlcv_columns(load_market_data_from_config(config), config)
    bars = normalize_timestamp_column(bars, config)
    event_rows = build_event_rows(config)
    feature_df, dropped_events = extract_event_features(bars, event_rows, config)
    if feature_df.is_empty():
        raise RuntimeError("No valid events remained after feature extraction.")

    folds = build_folds(feature_df)
    bars_by_index = bars.select(
        ["timestamp", "open", "high", "low", "close", "volume"]
    ).to_dicts()
    param_grid = strategy_cfg["parameter_grid"]
    search_space = cartesian_product(param_grid)
    execution_cfg = config["execution"]

    all_trades: list[dict[str, Any]] = []
    chosen_params: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in folds:
        train_df = filter_years(feature_df, fold["train_start"], fold["train_end"])
        test_df = filter_years(feature_df, fold["test_start"], fold["test_end"])
        if train_df.is_empty() or test_df.is_empty():
            continue

        for rule_name in ("baseline", "treatment"):
            best_params, train_metrics = select_best_params(
                feature_df=train_df,
                bars_by_index=bars_by_index,
                rule_name=rule_name,
                search_space=search_space,
                execution_cfg=execution_cfg,
            )
            test_trades, test_metrics = evaluate_rule(
                feature_df=test_df,
                bars_by_index=bars_by_index,
                rule_name=rule_name,
                params=best_params,
                execution_cfg=execution_cfg,
            )
            for trade in test_trades:
                trade["fold_name"] = fold["fold_name"]
            all_trades.extend(test_trades)
            chosen_params.append(
                {
                    "fold_name": fold["fold_name"],
                    "rule": rule_name,
                    "train_window": f"{fold['train_start']}-{fold['train_end']}",
                    "test_window": f"{fold['test_start']}-{fold['test_end']}",
                    "params_json": json.dumps(best_params, sort_keys=True),
                    "train_expectancy_per_trade": train_metrics["expectancy_per_trade"],
                    "test_expectancy_per_trade": test_metrics["expectancy_per_trade"],
                }
            )
            fold_summaries.append(
                {
                    "fold_name": fold["fold_name"],
                    "rule": rule_name,
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    **test_metrics,
                }
            )

    trades_df = pl.DataFrame(all_trades).sort(["rule", "exit_ts"]) if all_trades else pl.DataFrame()
    equity_curve_df = build_equity_curve(trades_df)
    dropped_df = pl.DataFrame(dropped_events) if dropped_events else pl.DataFrame()

    overall = {}
    for rule_name in ("baseline", "treatment"):
        subset = trades_df.filter(pl.col("rule") == rule_name) if not trades_df.is_empty() else pl.DataFrame()
        overall[rule_name] = summarize_rule_metrics(subset.to_dicts()) if subset.height else summarize_rule_metrics([])

    metrics_payload = {
        "strategy_id": strategy_cfg["id"],
        "contract_path": config["runtime"]["contract_path"],
        "overall": overall,
        "by_event_family": compute_family_metrics(trades_df),
        "folds": fold_summaries,
        "chosen_parameters": chosen_params,
        "notes": [
            "The ES R2 dataset key is wired from the confirmed manifest.",
            "Public macro-release and cash-open calendars must be supplied locally via config.",
            "No additional experiments were required before first-pass backtesting because the contract already defines the core robustness path.",
        ],
    }

    if not trades_df.is_empty():
        trades_df.write_parquet(output_dir / "trades.parquet")
    else:
        pl.DataFrame({"rule": [], "net_pnl_usd": []}).write_parquet(output_dir / "trades.parquet")
    equity_curve_df.write_parquet(output_dir / "equity_curve.parquet")
    feature_df.write_parquet(output_dir / "diagnostics.parquet")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)
    write_report(
        output_dir=output_dir,
        strategy_id=strategy_cfg["id"],
        metrics=metrics_payload,
        chosen_params=chosen_params,
        dropped_df=dropped_df,
    )


if __name__ == "__main__":
    main()
