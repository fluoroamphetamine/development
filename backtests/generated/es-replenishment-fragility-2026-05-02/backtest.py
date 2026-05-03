#!/usr/bin/env python3
"""Runnable backtest for the ES replenishment-fragility contract.

This package follows the exact contract strategy_id and keeps the data-loading
path compatible with the shared R2 helper while freezing the contract's
unresolved mechanics as explicit first-pass assumptions:

- continuation-only direction mapping
- observation window sensitivity
- continuation horizon sensitivity
- volatility-only baseline quantiles
- proxy-stack trust quantiles
- family-specific vs merged-family evaluation
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns

NY_TZ = ZoneInfo("America/New_York")
EPSILON = 1e-9


@dataclass(frozen=True)
class EventDefinition:
    family: str
    anchor: time


EVENT_DEFINITIONS = {
    "macro_release_0830": EventDefinition("macro_release_0830", time(8, 30)),
    "cash_open_0930": EventDefinition("cash_open_0930", time(9, 30)),
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


def normalize_timestamp_column(bars: pl.DataFrame, timezone_name: str) -> pl.DataFrame:
    dtype = bars.schema.get("timestamp")
    expr = pl.col("timestamp")
    if dtype == pl.Utf8:
        expr = expr.str.to_datetime(strict=False)
    if getattr(dtype, "time_zone", None):
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
    raise ValueError(f"Unsupported table format: {path}")


def parse_date_column(df: pl.DataFrame, column: str) -> list[date]:
    series = df.get_column(column)
    if series.dtype == pl.Date:
        out = series
    elif series.dtype == pl.Utf8:
        out = series.str.strptime(pl.Date, strict=False)
    else:
        out = series.cast(pl.Date)
    return out.drop_nulls().to_list()


def build_event_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    calendars = config["event_calendars"]
    macro_cfg = calendars["macro_release_calendar"]
    cash_cfg = calendars["cash_open_calendar"]
    macro_dates = parse_date_column(read_table(Path(macro_cfg["path"])), macro_cfg["date_column"])
    cash_dates = parse_date_column(read_table(Path(cash_cfg["path"])), cash_cfg["date_column"])
    rows: list[dict[str, Any]] = []
    for event_date in macro_dates:
        rows.append({"event_date": event_date, "event_window_name": "macro_release_0830"})
    for event_date in cash_dates:
        rows.append({"event_date": event_date, "event_window_name": "cash_open_0930"})
    rows.sort(key=lambda row: (row["event_date"], row["event_window_name"]))
    return rows


def expected_minute_range(start: datetime, end: datetime) -> list[datetime]:
    current = start
    out: list[datetime] = []
    while current <= end:
        out.append(current)
        current += timedelta(minutes=1)
    return out


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / max(abs(denominator), EPSILON)


def quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if value is not None and not math.isnan(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(clean) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return clean[low]
    weight = pos - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def compute_true_range(rows: list[dict[str, Any]]) -> float:
    prev_close: float | None = None
    true_ranges: list[float] = []
    for row in rows:
        high = float(row["high"])
        low = float(row["low"])
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
        prev_close = float(row["close"])
    return sum(true_ranges) / max(len(true_ranges), 1)


def path_length_from_rows(rows: list[dict[str, Any]], anchor_open: float) -> float:
    closes = [anchor_open] + [float(row["close"]) for row in rows]
    return sum(abs(curr - prev) for prev, curr in zip(closes, closes[1:]))


def impact_containment(rows: list[dict[str, Any]], impulse_sign: int, last_close: float) -> float:
    highs = [float(row["high"]) for row in rows]
    lows = [float(row["low"]) for row in rows]
    slice_high = max(highs)
    slice_low = min(lows)
    slice_range = max(slice_high - slice_low, EPSILON)
    if impulse_sign >= 0:
        return safe_div(last_close - slice_low, slice_range)
    return safe_div(slice_high - last_close, slice_range)


def extract_event_features(
    bars: pl.DataFrame,
    event_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    observation_windows = sorted(config["strategy"]["parameter_grid"]["observation_window_minutes"])
    continuation_horizons = sorted(
        config["strategy"]["parameter_grid"]["continuation_horizon_minutes"]
    )
    volume_lookback = int(config["strategy"].get("volume_lookback_events", 20))
    max_obs = max(observation_windows)
    max_horizon = max(continuation_horizons)

    records = bars.select(
        ["timestamp", "open", "high", "low", "close", "volume", "calendar_year"]
    ).to_dicts()
    ts_to_idx = {row["timestamp"]: idx for idx, row in enumerate(records)}

    raw_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    for event in event_rows:
        definition = EVENT_DEFINITIONS[event["event_window_name"]]
        anchor_ts = datetime.combine(event["event_date"], definition.anchor, tzinfo=NY_TZ)
        required_start = anchor_ts - timedelta(minutes=20)
        required_end = anchor_ts + timedelta(minutes=max_obs + max_horizon)
        missing = [ts for ts in expected_minute_range(required_start, required_end) if ts not in ts_to_idx]
        if missing:
            dropped_rows.append(
                {
                    "event_date": event["event_date"],
                    "event_window_name": event["event_window_name"],
                    "reason": "missing_required_bars",
                    "missing_count": len(missing),
                }
            )
            continue

        anchor_idx = ts_to_idx[anchor_ts]
        pre_event_rows = records[anchor_idx - 20 : anchor_idx]
        atr20 = compute_true_range(pre_event_rows)
        anchor_open = float(records[anchor_idx]["open"])

        for obs_minutes in observation_windows:
            decision_idx = anchor_idx + obs_minutes
            decision_ts = records[decision_idx]["timestamp"]
            entry_idx = decision_idx + 1
            slice_rows = records[anchor_idx : decision_idx + 1]
            last_close = float(records[decision_idx]["close"])
            impulse_raw = last_close - anchor_open
            impulse_sign = 0 if impulse_raw == 0 else (1 if impulse_raw > 0 else -1)
            impulse_size_bp = 10000.0 * safe_div(last_close - anchor_open, anchor_open)
            path_length = path_length_from_rows(slice_rows, anchor_open)
            net_displacement = abs(last_close - anchor_open)
            dispersion_stress = safe_div(path_length, net_displacement if net_displacement else 1.0)
            event_volume = sum(float(row["volume"]) for row in slice_rows)
            containment = impact_containment(slice_rows, impulse_sign, last_close)

            row: dict[str, Any] = {
                "event_date": event["event_date"],
                "calendar_year": event["event_date"].year,
                "event_window_name": event["event_window_name"],
                "anchor_ts": anchor_ts,
                "decision_ts": decision_ts,
                "entry_ts": records[entry_idx]["timestamp"],
                "anchor_idx": anchor_idx,
                "decision_idx": decision_idx,
                "entry_idx": entry_idx,
                "observation_window_minutes": obs_minutes,
                "atr20_pre_event": atr20,
                "anchor_open": anchor_open,
                "decision_close": last_close,
                "entry_open": float(records[entry_idx]["open"]),
                "impulse_sign": impulse_sign,
                "impulse_size_bp": impulse_size_bp,
                "short_window_dispersion_stress": dispersion_stress,
                "event_volume": event_volume,
                "impact_containment": containment,
            }
            for horizon in continuation_horizons:
                exit_idx = decision_idx + horizon
                exit_close = float(records[exit_idx]["close"])
                continuation_return_bp = (
                    10000.0 * impulse_sign * safe_div(exit_close - row["entry_open"], row["entry_open"])
                    if impulse_sign
                    else 0.0
                )
                row[f"continuation_return_bp_{horizon}"] = continuation_return_bp
                row[f"trust_label_{horizon}"] = int(continuation_return_bp > 0.0)
            raw_rows.append(row)

    feature_df = pl.DataFrame(raw_rows).sort(
        ["event_window_name", "observation_window_minutes", "event_date"]
    )
    if feature_df.is_empty():
        return feature_df, pl.DataFrame(dropped_rows)

    by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_df.to_dicts():
        by_key[(row["event_window_name"], row["observation_window_minutes"])].append(row)

    enriched_rows: list[dict[str, Any]] = []
    for rows in by_key.values():
        rows.sort(key=lambda row: row["event_date"])
        prior_volumes: list[float] = []
        for row in rows:
            baseline = median(prior_volumes[-volume_lookback:]) if prior_volumes else None
            row["volume_rate_context"] = (
                safe_div(row["event_volume"], baseline) if baseline not in {None, 0.0} else None
            )
            enriched_rows.append(row)
            prior_volumes.append(row["event_volume"])

    return pl.DataFrame(enriched_rows).sort(
        ["event_date", "event_window_name", "observation_window_minutes"]
    ), pl.DataFrame(dropped_rows)


def build_folds(feature_df: pl.DataFrame, initial_train_years: int, test_years: int) -> list[dict[str, Any]]:
    years = sorted(feature_df.get_column("calendar_year").unique().to_list())
    if not years:
        return []
    min_year = years[0]
    max_year = years[-1]
    train_end = min_year + initial_train_years - 1
    test_start = train_end + 1
    folds: list[dict[str, Any]] = []
    while test_start <= max_year:
        test_end = min(test_start + test_years - 1, max_year)
        folds.append(
            {
                "fold_name": f"{min_year}-{train_end}_to_{test_start}-{test_end}",
                "train_start": min_year,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_end += test_years
        test_start += test_years
    return folds


def row_matches_family_mode(row: dict[str, Any], family_mode: str) -> bool:
    if family_mode == "08:30_only":
        return row["event_window_name"] == "macro_release_0830"
    if family_mode == "09:30_only":
        return row["event_window_name"] == "cash_open_0930"
    return True


def training_group_key(row: dict[str, Any], family_mode: str) -> str:
    if family_mode == "separate_models":
        return row["event_window_name"]
    return family_mode


def compute_thresholds(train_rows: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[training_group_key(row, params["window_family_mode"])].append(row)

    thresholds: dict[str, dict[str, float]] = {}
    for group, rows in grouped.items():
        impulse_abs = [abs(float(row["impulse_size_bp"])) for row in rows]
        dispersion = [float(row["short_window_dispersion_stress"]) for row in rows]
        volume = [float(row["volume_rate_context"]) for row in rows if row["volume_rate_context"] is not None]
        containment = [float(row["impact_containment"]) for row in rows]
        thresholds[group] = {
            "volatility_gate_bp": quantile(impulse_abs, params["volatility_threshold_quantile"]),
            "dispersion_max": quantile(dispersion, 1.0 - params["proxy_threshold_quantile"]),
            "volume_min": quantile(volume, params["proxy_threshold_quantile"]) if volume else 0.0,
            "containment_min": quantile(containment, params["proxy_threshold_quantile"]),
        }
    return thresholds


def threshold_for_row(
    thresholds: dict[str, dict[str, float]], row: dict[str, Any], family_mode: str
) -> dict[str, float]:
    if family_mode == "separate_models":
        return thresholds[row["event_window_name"]]
    return thresholds.get(family_mode) or next(iter(thresholds.values()))


def simulate_trade(
    row: dict[str, Any],
    bars_by_index: list[dict[str, Any]],
    horizon_minutes: int,
    stop_range_multiple: float,
    tick_size: float,
    point_value: float,
    commission_per_side: float,
) -> dict[str, Any]:
    direction = int(row["impulse_sign"])
    entry_idx = int(row["entry_idx"])
    exit_idx = int(row["decision_idx"]) + horizon_minutes
    entry_row = bars_by_index[entry_idx]
    entry_raw = float(entry_row["open"])
    entry_price = entry_raw + tick_size if direction > 0 else entry_raw - tick_size
    slice_rows = bars_by_index[int(row["anchor_idx"]) : int(row["decision_idx"]) + 1]
    slice_high = max(float(bar["high"]) for bar in slice_rows)
    slice_low = min(float(bar["low"]) for bar in slice_rows)
    stop_distance = stop_range_multiple * max(slice_high - slice_low, tick_size)

    exit_row = bars_by_index[exit_idx]
    exit_price = float(exit_row["close"])
    exit_reason = "time_exit"
    max_adverse = 0.0

    for bar in bars_by_index[entry_idx : exit_idx + 1]:
        high = float(bar["high"])
        low = float(bar["low"])
        if direction > 0:
            adverse = max(0.0, entry_price - low)
            stop_level = entry_price - stop_distance
            if low <= stop_level:
                exit_price = stop_level - tick_size
                exit_reason = "stop_hit"
                exit_row = bar
                max_adverse = max(max_adverse, adverse)
                break
        else:
            adverse = max(0.0, high - entry_price)
            stop_level = entry_price + stop_distance
            if high >= stop_level:
                exit_price = stop_level + tick_size
                exit_reason = "stop_hit"
                exit_row = bar
                max_adverse = max(max_adverse, adverse)
                break
        max_adverse = max(max_adverse, adverse)

    if exit_reason == "time_exit":
        exit_price = exit_price - tick_size if direction > 0 else exit_price + tick_size

    gross_points = direction * (exit_price - entry_price)
    net_pnl_usd = gross_points * point_value - (2.0 * commission_per_side)
    return {
        "entry_ts": entry_row["timestamp"],
        "exit_ts": exit_row["timestamp"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_points": gross_points,
        "net_pnl_usd": net_pnl_usd,
        "exit_reason": exit_reason,
        "max_adverse_excursion_points": max_adverse,
    }


def summarize_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "average_trade_return_usd": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_usd": 0.0,
        }
    pnls = [float(trade["net_pnl_usd"]) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0.0]
    losses = [pnl for pnl in pnls if pnl < 0.0]
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        running += pnl
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)
    return {
        "trade_count": len(trades),
        "average_trade_return_usd": sum(pnls) / len(pnls),
        "hit_rate": sum(1 for pnl in pnls if pnl > 0.0) / len(pnls),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("inf"),
        "max_drawdown_usd": abs(max_drawdown),
    }


def confusion_and_accuracy(
    rows: list[dict[str, Any]],
    predicted_key: str,
    label_getter,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = bool(row[predicted_key])
        actual = bool(label_getter(row))
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": accuracy, "count": total}


def evaluate_rule(
    rows: list[dict[str, Any]],
    bars_by_index: list[dict[str, Any]],
    params: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
    execution_cfg: dict[str, Any],
    rule_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluation_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for row in rows:
        if not row_matches_family_mode(row, params["window_family_mode"]):
            continue
        family_thresholds = threshold_for_row(thresholds, row, params["window_family_mode"])
        baseline_pass = (
            row["impulse_sign"] != 0
            and abs(float(row["impulse_size_bp"])) >= family_thresholds["volatility_gate_bp"]
        )
        proxy_pass = (
            row["volume_rate_context"] is not None
            and float(row["short_window_dispersion_stress"]) <= family_thresholds["dispersion_max"]
            and float(row["volume_rate_context"]) >= family_thresholds["volume_min"]
            and float(row["impact_containment"]) >= family_thresholds["containment_min"]
        )
        predicted_trustworthy = baseline_pass if rule_name == "baseline" else baseline_pass and proxy_pass
        evaluation = dict(row)
        evaluation.update(
            {
                "rule": rule_name,
                "window_family_mode": params["window_family_mode"],
                "continuation_horizon_minutes": params["continuation_horizon_minutes"],
                "predicted_trustworthy": int(predicted_trustworthy),
                "baseline_pass": int(baseline_pass),
                "proxy_pass": int(proxy_pass),
                "volatility_gate_bp": family_thresholds["volatility_gate_bp"],
                "dispersion_max": family_thresholds["dispersion_max"],
                "volume_min": family_thresholds["volume_min"],
                "containment_min": family_thresholds["containment_min"],
            }
        )
        evaluation_rows.append(evaluation)
        if not predicted_trustworthy:
            continue
        trade = simulate_trade(
            row=row,
            bars_by_index=bars_by_index,
            horizon_minutes=int(params["continuation_horizon_minutes"]),
            stop_range_multiple=float(params["stop_range_multiple"]),
            tick_size=float(execution_cfg["tick_size"]),
            point_value=float(execution_cfg["point_value"]),
            commission_per_side=float(execution_cfg["commissions"]),
        )
        trade.update(
            {
                "event_date": row["event_date"],
                "event_window_name": row["event_window_name"],
                "rule": rule_name,
                "window_family_mode": params["window_family_mode"],
                "observation_window_minutes": params["observation_window_minutes"],
                "continuation_horizon_minutes": params["continuation_horizon_minutes"],
                "params_json": json.dumps(params, sort_keys=True),
            }
        )
        trades.append(trade)

    confusion = confusion_and_accuracy(
        evaluation_rows,
        "predicted_trustworthy",
        lambda r: r[f"trust_label_{params['continuation_horizon_minutes']}"],
    )
    trade_summary = summarize_trades(trades)
    metrics = {
        "classification_accuracy": confusion["accuracy"],
        "confusion_matrix": confusion,
        "average_trade_return_usd": trade_summary["average_trade_return_usd"],
        "trade_count": trade_summary["trade_count"],
        "hit_rate": trade_summary["hit_rate"],
        "profit_factor": trade_summary["profit_factor"],
        "max_drawdown_usd": trade_summary["max_drawdown_usd"],
    }
    return evaluation_rows, trades, metrics


def score_metrics(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["classification_accuracy"]),
        float(metrics["average_trade_return_usd"]),
        float(metrics["trade_count"]),
    )


def cartesian_product(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(parameter_grid)
    values = [parameter_grid[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def params_match_row(row: dict[str, Any], params: dict[str, Any]) -> bool:
    return int(row["observation_window_minutes"]) == int(params["observation_window_minutes"])


def select_best_params(
    train_rows: list[dict[str, Any]],
    bars_by_index: list[dict[str, Any]],
    search_space: list[dict[str, Any]],
    execution_cfg: dict[str, Any],
    rule_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    best_params: dict[str, Any] | None = None
    best_thresholds: dict[str, dict[str, float]] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None

    for params in search_space:
        scoped_rows = [row for row in train_rows if params_match_row(row, params)]
        if not scoped_rows:
            continue
        filtered_rows = [row for row in scoped_rows if row_matches_family_mode(row, params["window_family_mode"])]
        if not filtered_rows:
            continue
        thresholds = compute_thresholds(filtered_rows, params)
        if not thresholds:
            continue
        _, _, metrics = evaluate_rule(
            rows=scoped_rows,
            bars_by_index=bars_by_index,
            params=params,
            thresholds=thresholds,
            execution_cfg=execution_cfg,
            rule_name=rule_name,
        )
        score = score_metrics(metrics)
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_thresholds = thresholds
            best_metrics = metrics

    if best_params is None or best_thresholds is None or best_metrics is None:
        raise RuntimeError(f"Unable to select parameters for {rule_name}")
    return best_params, best_thresholds, best_metrics


def build_equity_curve(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame({"rule": [], "exit_ts": [], "equity_usd": []})
    pieces: list[pl.DataFrame] = []
    for rule in trades_df.get_column("rule").unique().to_list():
        subset = trades_df.filter(pl.col("rule") == rule).sort("exit_ts")
        pieces.append(
            subset.with_columns(pl.col("net_pnl_usd").cum_sum().alias("equity_usd")).select(
                ["rule", "exit_ts", "net_pnl_usd", "equity_usd"]
            )
        )
    return pl.concat(pieces, how="vertical_relaxed")


def summarize_by_family(evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[(row["rule"], row["event_window_name"])].append(row)
    out: list[dict[str, Any]] = []
    for (rule, family), rows in sorted(grouped.items()):
        summary = confusion_and_accuracy(
            rows,
            "predicted_trustworthy",
            lambda r: r[f"trust_label_{r['continuation_horizon_minutes']}"],
        )
        out.append({"rule": rule, "event_window_name": family, **summary})
    return out


def summarize_by_observation_and_horizon(
    evaluation_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[
            (
                row["rule"],
                int(row["observation_window_minutes"]),
                int(row["continuation_horizon_minutes"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (rule, obs, horizon), rows in sorted(grouped.items()):
        summary = confusion_and_accuracy(
            rows,
            "predicted_trustworthy",
            lambda r: r[f"trust_label_{r['continuation_horizon_minutes']}"],
        )
        out.append(
            {
                "rule": rule,
                "observation_window_minutes": obs,
                "continuation_horizon_minutes": horizon,
                "classification_accuracy": summary["accuracy"],
                "event_count": summary["count"],
            }
        )
    return out


def write_report(
    output_dir: Path,
    strategy_id: str,
    contract_review: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> None:
    added = contract_review["added_experiments"]
    lines = [
        "# Strategy Contract Review & Backtesting Brief",
        "",
        "## 1. Contract Review Summary",
        "",
        "| Source | Key Insight | Relevance |",
        "| --- | --- | --- |",
    ]
    for row in contract_review["summary_rows"]:
        lines.append(f"| {row['source']} | {row['insight']} | {row['relevance']} |")

    lines.extend(
        [
            "",
            "## 2. Strategy Interpretation",
            "",
            f"- Contract: `{contract_review['contract_path']}`",
            f"- Strategy ID: `{strategy_id}`",
            f"- Core thesis: {contract_review['core_thesis']}",
            f"- Key assumptions: {contract_review['assumptions']}",
            f"- Inputs and outputs: {contract_review['inputs_outputs']}",
            f"- Constraints: {contract_review['constraints']}",
            "",
            "## 3. Validation Assessment",
            "",
            contract_review["validation_assessment"],
            "",
            "## 4. Added Experiments",
            "",
        ]
    )
    for experiment in added:
        lines.extend(
            [
                f"- {experiment['name']}: {experiment['purpose']}",
                f"  Assumption tested: {experiment['assumption']}",
                f"  Success metric: {experiment['success_metric']}",
                f"  Failure metric: {experiment['failure_metric']}",
                f"  Decision rule: {experiment['decision_rule']}",
                f"  Limitation: {experiment['limitation']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 5. Backtesting Approach",
            "",
            contract_review["approach"],
            "",
            "## 6. Backtesting Code",
            "",
            "- Run from repo root with the package config in this directory.",
            "- Outputs written: `trades.parquet`, `equity_curve.parquet`, `metrics.json`, `diagnostics.parquet`, `report.md`.",
            "",
            "## 7. Immediate Next Actions",
            "",
        ]
    )
    for step in contract_review["next_actions"]:
        lines.append(f"- {step}")
    lines.extend(
        [
            "",
            "## Result Snapshot",
            "",
            f"- Baseline accuracy: {metrics_payload['overall']['baseline']['classification_accuracy']:.4f}",
            f"- Treatment accuracy: {metrics_payload['overall']['treatment']['classification_accuracy']:.4f}",
            f"- Treatment minus baseline accuracy: {metrics_payload['comparison']['treatment_minus_baseline_accuracy']:.4f}",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_output_dir(Path(config["runtime"]["output_dir"]))

    bars = normalize_ohlcv_columns(load_market_data_from_config(config), config)
    bars = normalize_timestamp_column(bars, config["runtime"]["timezone"])
    events = build_event_rows(config)
    features_df, dropped_df = extract_event_features(bars, events, config)
    if features_df.is_empty():
        raise RuntimeError("No valid events remained after feature extraction.")

    execution_cfg = config["execution"]
    strategy_cfg = config["strategy"]
    walk_forward_cfg = config["walk_forward"]
    search_space = cartesian_product(strategy_cfg["parameter_grid"])
    folds = build_folds(
        features_df,
        int(walk_forward_cfg["initial_train_years"]),
        int(walk_forward_cfg["test_years"]),
    )
    bars_by_index = bars.select(["timestamp", "open", "high", "low", "close", "volume"]).to_dicts()

    all_trades: list[dict[str, Any]] = []
    all_evaluations: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    chosen_parameters: list[dict[str, Any]] = []

    feature_rows = features_df.to_dicts()

    for fold in folds:
        train_rows = [
            row
            for row in feature_rows
            if fold["train_start"] <= row["calendar_year"] <= fold["train_end"]
        ]
        test_rows = [
            row
            for row in feature_rows
            if fold["test_start"] <= row["calendar_year"] <= fold["test_end"]
        ]
        if not train_rows or not test_rows:
            continue
        for rule_name in ("baseline", "treatment"):
            best_params, thresholds, train_metrics = select_best_params(
                train_rows=train_rows,
                bars_by_index=bars_by_index,
                search_space=search_space,
                execution_cfg=execution_cfg,
                rule_name=rule_name,
            )
            scoped_test_rows = [row for row in test_rows if params_match_row(row, best_params)]
            evaluations, trades, test_metrics = evaluate_rule(
                rows=scoped_test_rows,
                bars_by_index=bars_by_index,
                params=best_params,
                thresholds=thresholds,
                execution_cfg=execution_cfg,
                rule_name=rule_name,
            )
            for row in evaluations:
                row["fold_name"] = fold["fold_name"]
            for row in trades:
                row["fold_name"] = fold["fold_name"]
            all_evaluations.extend(evaluations)
            all_trades.extend(trades)
            chosen_parameters.append(
                {
                    "fold_name": fold["fold_name"],
                    "rule": rule_name,
                    "train_window": f"{fold['train_start']}-{fold['train_end']}",
                    "test_window": f"{fold['test_start']}-{fold['test_end']}",
                    "params_json": json.dumps(best_params, sort_keys=True),
                    "thresholds_json": json.dumps(thresholds, sort_keys=True),
                    "train_classification_accuracy": train_metrics["classification_accuracy"],
                    "test_classification_accuracy": test_metrics["classification_accuracy"],
                }
            )
            fold_summaries.append({"fold_name": fold["fold_name"], "rule": rule_name, **test_metrics})

    evaluations_df = pl.DataFrame(all_evaluations) if all_evaluations else pl.DataFrame()
    trades_df = (
        pl.DataFrame(all_trades).sort(["rule", "exit_ts"])
        if all_trades
        else pl.DataFrame({"rule": [], "net_pnl_usd": []})
    )
    equity_curve_df = build_equity_curve(trades_df)

    overall: dict[str, Any] = {}
    for rule_name in ("baseline", "treatment"):
        rule_rows = [row for row in all_evaluations if row["rule"] == rule_name]
        confusion = confusion_and_accuracy(
            rule_rows,
            "predicted_trustworthy",
            lambda r: r[f"trust_label_{r['continuation_horizon_minutes']}"],
        )
        trade_rows = [row for row in all_trades if row["rule"] == rule_name]
        trade_summary = summarize_trades(trade_rows)
        overall[rule_name] = {
            "classification_accuracy": confusion["accuracy"],
            "confusion_matrix": confusion,
            "average_trade_return_usd": trade_summary["average_trade_return_usd"],
            "trade_count": trade_summary["trade_count"],
            "hit_rate": trade_summary["hit_rate"],
            "profit_factor": trade_summary["profit_factor"],
            "max_drawdown_usd": trade_summary["max_drawdown_usd"],
        }

    by_family = summarize_by_family(all_evaluations)
    comparison = {
        "treatment_minus_baseline_accuracy": overall["treatment"]["classification_accuracy"]
        - overall["baseline"]["classification_accuracy"]
    }

    experiment_summary = {
        "observation_horizon_sensitivity": summarize_by_observation_and_horizon(all_evaluations),
        "window_family_stability": by_family,
    }

    contract_review = {
        "contract_path": config["runtime"]["contract_path"],
        "summary_rows": [
            {
                "source": "Canonical contract",
                "insight": "Defines the ES benchmark-window trust-filter question and named proxy stack.",
                "relevance": "Primary implementation source.",
            },
            {
                "source": "Data contract",
                "insight": "Binds the R2 loader, explicit OHLCV column mapping, and missing-macro-calendar rule.",
                "relevance": "Keeps the package runnable against the shared development runtime.",
            },
            {
                "source": "R2 manifest",
                "insight": "Confirms the ES 1-minute parquet key in Cloudflare R2.",
                "relevance": "Lets the config stay runnable against the shared data path.",
            },
        ],
        "core_thesis": (
            "A public proxy stack built from short-window dispersion stress, volume-rate context, "
            "and impact containment should separate trustworthy benchmark-window continuation "
            "from fragile failure better than a volatility-only gate."
        ),
        "assumptions": (
            "The first-pass test stays ES-only, uses 1-minute bars, treats the signal as a "
            "continuation filter, and evaluates 08:30 and 09:30 windows separately before any merge."
        ),
        "inputs_outputs": (
            "Inputs are ES 1-minute OHLCV bars from R2 plus public macro-release and cash-open calendars. "
            "Outputs are event diagnostics, trades, equity, metrics, and a written report."
        ),
        "constraints": (
            "The contract remains incomplete, so entry timing, exit timing, thresholds, and walk-forward "
            "protocol must be frozen as explicit implementation assumptions."
        ),
        "validation_assessment": (
            "The contract is strong enough for a first runnable backtest, but it still needs a small set "
            "of validation experiments because its main risks are structural: whether the proxy adds "
            "information beyond volatility, whether the best observation and holding windows are stable, "
            "and whether 08:30 and 09:30 behave like the same family."
        ),
        "added_experiments": [
            {
                "name": "Observation And Exit Stability",
                "purpose": "Test whether the strategy only works for one narrow timing choice.",
                "assumption": "Decision-time and exit-horizon choices are not hiding a brittle implementation.",
                "success_metric": "Treatment accuracy and average trade return remain competitive across multiple observation and continuation windows.",
                "failure_metric": "The edge collapses outside one single timing pair.",
                "decision_rule": "Reject any timing choice whose treatment uplift is isolated and not reproducible in neighboring windows.",
                "limitation": "This is still based on 1-minute bars, so sub-minute microstructure remains unobserved.",
            },
            {
                "name": "Proxy Incremental Value Vs Volatility",
                "purpose": "Check whether the proxy stack adds real information beyond the volatility gate.",
                "assumption": "Dispersion, volume-rate context, and containment are not just restating impulse size.",
                "success_metric": "Treatment minus baseline classification accuracy is positive overall and by window family.",
                "failure_metric": "Treatment matches or trails the baseline despite extra complexity.",
                "decision_rule": "Demote the proxy layer if uplift is non-positive after walk-forward testing.",
                "limitation": "A positive result is still conditional on the chosen first-pass continuation mapping.",
            },
            {
                "name": "Window Family Stability",
                "purpose": "Confirm whether 08:30 and 09:30 can share a single concept.",
                "assumption": "The mechanism behaves similarly across macro-release and cash-open benchmark windows.",
                "success_metric": "Treatment uplift is directionally consistent in both families.",
                "failure_metric": "One family works while the other is flat or negative.",
                "decision_rule": "Split the strategy by family or reject the merged framing if instability persists.",
                "limitation": "Family imbalance can still reduce confidence if one side has much fewer valid events.",
            },
        ],
        "approach": (
            "The backtest loads the confirmed ES R2 dataset, tags 08:30 and 09:30 event dates from local "
            "calendar files, computes feature rows for observation windows of 1, 3, and 5 minutes, labels "
            "continuation outcomes over 5, 15, and 30 minute horizons, fits quantile thresholds in an anchored "
            "5-year train / 1-year test walk-forward, and compares a volatility-only baseline with a proxy-stack treatment."
        ),
        "next_actions": [
            "Populate the macro-release and cash-open input calendars in the package inputs directory.",
            "Run the package against the confirmed ES R2 key and inspect `metrics.json` first.",
            "Check treatment uplift separately for `macro_release_0830` and `cash_open_0930` before pooling anything.",
            "Promote the winning observation window, horizon, and family mode back into the canonical contract if the results are stable.",
            "Reject the replenishment layer quickly if it does not beat the volatility-only baseline after costs.",
        ],
    }

    metrics_payload = {
        "strategy_id": strategy_cfg["id"],
        "contract_path": config["runtime"]["contract_path"],
        "overall": overall,
        "comparison": comparison,
        "by_event_family": by_family,
        "folds": fold_summaries,
        "chosen_parameters": chosen_parameters,
        "experiments": experiment_summary,
        "dropped_events": dropped_df.to_dicts() if dropped_df.height else [],
        "implementation_notes": [
            "This first-pass implementation freezes a continuation-only direction rule.",
            "Volatility and proxy thresholds are fit from training quantiles instead of being hard-coded.",
            "The contract still depends on external public event calendars supplied through the package config.",
        ],
    }

    trades_df.write_parquet(output_dir / "trades.parquet")
    equity_curve_df.write_parquet(output_dir / "equity_curve.parquet")
    evaluations_df.write_parquet(output_dir / "diagnostics.parquet")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)
    write_report(output_dir, strategy_cfg["id"], contract_review, metrics_payload)


if __name__ == "__main__":
    main()
