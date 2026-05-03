#!/usr/bin/env python3
"""Backtest for the ES replenishment fragility contract."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns


TIMEZONE = "America/New_York"
BASELINE_RULE = "volatility_only_benchmark_filter"
TREATMENT_RULE = "replenishment_failure_proxy_stack"


@dataclass(frozen=True)
class RuleParams:
    rule_id: str
    observation_window_minutes: int
    continuation_horizon_minutes: int
    window_family_mode: str
    threshold_quantile: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def ensure_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    runtime = config.setdefault("runtime", {})
    output_dir = Path(runtime.get("output_dir", config_path.parent / "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_calendar(path: Path, preferred_columns: list[str]) -> list[pl.Date] | list[Any]:
    if not path.exists():
        return []

    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path)

    for name in preferred_columns:
        if name in df.columns:
            series = (
                df.select(pl.col(name).cast(pl.Date, strict=False).alias("event_date"))
                .drop_nulls()
                .unique()
                .sort("event_date")
                .get_column("event_date")
            )
            return series.to_list()
    return []


def safe_sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def pct_to_bp(raw: float) -> float:
    return raw * 10000.0


def round_trip_cost_bp(entry_price: float, execution: dict[str, Any]) -> float:
    total_cost_usd = (
        2.0 * float(execution["slippage_ticks_per_side"]) * float(execution["tick_size"]) * float(execution["point_value"])
        + 2.0 * float(execution["commissions_per_side_usd"])
    )
    return total_cost_usd / (entry_price * float(execution["point_value"])) * 10000.0


def load_bars(config: dict[str, Any]) -> pl.DataFrame:
    bars = load_market_data_from_config(config)
    bars = normalize_ohlcv_columns(bars, config)
    bars = (
        bars.with_columns(
            pl.col("timestamp").str.to_datetime(strict=False, time_zone="UTC").alias("timestamp")
        )
        if bars.schema.get("timestamp") == pl.Utf8
        else bars
    )
    return bars.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).alias("event_ts_et"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ).with_columns(
        pl.col("event_ts_et").dt.date().alias("event_date"),
        (pl.col("event_ts_et").dt.hour() * 60 + pl.col("event_ts_et").dt.minute()).alias("minute_of_day"),
    )


def build_price_lookup(bars: pl.DataFrame) -> dict[tuple[Any, int], dict[str, float]]:
    mapping: dict[tuple[Any, int], dict[str, float]] = {}
    for row in bars.iter_rows(named=True):
        mapping[(row["event_date"], row["minute_of_day"])] = row
    return mapping


def minute_value(hour: int, minute: int) -> int:
    return hour * 60 + minute


def build_window_events(config: dict[str, Any], bars: pl.DataFrame) -> pl.DataFrame:
    calendars = config["event_calendars"]
    macro_dates = read_calendar(Path(calendars["macro_release_calendar"]["path"]), ["event_date", "release_date", "date"])
    cash_dates = read_calendar(Path(calendars["cash_open_calendar"]["path"]), ["event_date", "session_date", "date"])

    if not macro_dates and not cash_dates:
        raise RuntimeError(
            "No usable event dates were found. Provide a real macro release calendar, a cash-open calendar, or both."
        )

    lookup = build_price_lookup(
        bars.filter(
            ((pl.col("minute_of_day") >= minute_value(8, 25)) & (pl.col("minute_of_day") <= minute_value(10, 0)))
        )
    )

    event_rows: list[dict[str, Any]] = []
    family_specs = [
        ("08:30", "macro_release", macro_dates, minute_value(8, 30)),
        ("09:30", "cash_open", cash_dates, minute_value(9, 30)),
    ]

    for family_name, family_tag, event_dates, anchor_minute in family_specs:
        for event_date in event_dates:
            required = []
            for offset in range(0, 31):
                required.append((event_date, anchor_minute + offset))
            if not all(key in lookup for key in required):
                continue

            event_slice = [
                lookup[(event_date, anchor_minute + offset)]
                for offset in range(0, 31)
            ]
            event_frame = pl.DataFrame(event_slice).sort("minute_of_day")
            anchor_row = lookup[(event_date, anchor_minute)]
            row: dict[str, Any] = {
                "event_date": event_date,
                "window_family": family_name,
                "family_tag": family_tag,
                "anchor_minute": anchor_minute,
                "anchor_open": anchor_row["open"],
                "anchor_close": anchor_row["close"],
            }

            pre_window = bars.filter(
                (pl.col("event_date") == event_date)
                & (pl.col("minute_of_day") >= anchor_minute - 15)
                & (pl.col("minute_of_day") < anchor_minute)
            )
            pre_mean_volume = float(pre_window.get_column("volume").mean()) if not pre_window.is_empty() else None

            for obs in config["parameter_grid"]["observation_window_minutes"]:
                obs_row = lookup[(event_date, anchor_minute + obs)]
                obs_frame = event_frame.filter(pl.col("minute_of_day") <= anchor_minute + obs)
                impulse_bp = pct_to_bp(obs_row["close"] / anchor_row["close"] - 1.0)
                row[f"initial_impulse_size_bp_{obs}"] = impulse_bp
                row[f"initial_impulse_sign_{obs}"] = safe_sign(impulse_bp)

                close_path = obs_frame.get_column("close").to_list()
                minute_returns = []
                previous = anchor_row["close"]
                for current in close_path:
                    minute_returns.append(pct_to_bp(current / previous - 1.0))
                    previous = current
                gross_path = sum(abs(value) for value in minute_returns)
                dispersion = float(pl.Series(minute_returns).std()) if len(minute_returns) > 1 else 0.0
                impact_containment = abs(impulse_bp) / gross_path if gross_path > 0 else 0.0
                obs_mean_volume = float(obs_frame.get_column("volume").mean())
                volume_rate = obs_mean_volume / pre_mean_volume if pre_mean_volume not in (None, 0.0) else 1.0
                row[f"dispersion_stress_{obs}"] = dispersion
                row[f"volume_rate_context_{obs}"] = volume_rate
                row[f"impact_containment_score_{obs}"] = max(0.0, min(1.0, impact_containment))
                row[f"entry_price_{obs}"] = obs_row["open"]

                for horizon in config["parameter_grid"]["continuation_horizon_minutes"]:
                    exit_row = lookup[(event_date, anchor_minute + horizon)]
                    signed_return_bp = pct_to_bp(exit_row["close"] / obs_row["open"] - 1.0) * safe_sign(impulse_bp)
                    adverse_frame = event_frame.filter(
                        (pl.col("minute_of_day") >= anchor_minute + obs)
                        & (pl.col("minute_of_day") <= anchor_minute + horizon)
                    )
                    if safe_sign(impulse_bp) >= 0:
                        mae_bp = pct_to_bp(adverse_frame.get_column("low").min() / obs_row["open"] - 1.0)
                    else:
                        mae_bp = pct_to_bp(obs_row["open"] / adverse_frame.get_column("high").max() - 1.0)
                    row[f"signed_return_bp_{obs}_{horizon}"] = signed_return_bp
                    row[f"trust_label_{obs}_{horizon}"] = 1 if signed_return_bp > 0 else 0
                    row[f"mae_bp_{obs}_{horizon}"] = mae_bp

            event_rows.append(row)

    if not event_rows:
        raise RuntimeError("No usable 08:30 or 09:30 events survived the local bar coverage checks.")

    return pl.DataFrame(event_rows).sort(["window_family", "event_date"])


def filter_family(frame: pl.DataFrame, mode: str, family: str) -> pl.DataFrame:
    if mode == "08:30_only":
        return frame.filter(pl.col("window_family") == "08:30")
    if mode == "09:30_only":
        return frame.filter(pl.col("window_family") == "09:30")
    if mode == "separate_models":
        return frame.filter(pl.col("window_family") == family)
    return frame


def evaluate_rule(
    frame: pl.DataFrame,
    params: RuleParams,
    execution: dict[str, Any],
    family: str,
    split_name: str,
) -> pl.DataFrame:
    scoped = filter_family(frame, params.window_family_mode, family)
    if scoped.is_empty():
        return pl.DataFrame(
            {
                "event_date": pl.Series([], dtype=pl.Date),
                "window_family": pl.Series([], dtype=pl.Utf8),
                "rule_id": pl.Series([], dtype=pl.Utf8),
                "took_trade": pl.Series([], dtype=pl.Boolean),
                "correct_label": pl.Series([], dtype=pl.Boolean),
                "net_return_bp": pl.Series([], dtype=pl.Float64),
            }
        )

    obs = params.observation_window_minutes
    horizon = params.continuation_horizon_minutes
    label_col = f"trust_label_{obs}_{horizon}"
    signal_col = f"initial_impulse_size_bp_{obs}"
    direction_col = f"initial_impulse_sign_{obs}"
    entry_col = f"entry_price_{obs}"
    return_col = f"signed_return_bp_{obs}_{horizon}"

    if params.rule_id == BASELINE_RULE:
        threshold = float(scoped.select(pl.col(signal_col).abs().quantile(params.threshold_quantile)).item())
        verdict = (pl.col(signal_col).abs() >= threshold)
        strength_expr = pl.col(signal_col).abs()
    else:
        score_col = f"proxy_score_{obs}"
        scoped = scoped.with_columns(
            (
                pl.col(f"impact_containment_score_{obs}")
                + pl.col(f"volume_rate_context_{obs}").log1p()
                - pl.col(f"dispersion_stress_{obs}").log1p()
            ).alias(score_col)
        )
        threshold = float(scoped.select(pl.col(score_col).quantile(params.threshold_quantile)).item())
        verdict = pl.col(score_col) >= threshold
        strength_expr = pl.col(score_col)

    return scoped.with_columns(
        pl.lit(params.rule_id).alias("rule_id"),
        pl.lit(split_name).alias("split"),
        pl.lit(params.window_family_mode).alias("window_family_mode"),
        pl.lit(params.observation_window_minutes).alias("observation_window_minutes"),
        pl.lit(params.continuation_horizon_minutes).alias("continuation_horizon_minutes"),
        pl.lit(params.threshold_quantile).alias("threshold_quantile"),
        strength_expr.alias("signal_strength"),
        verdict.alias("took_trade"),
        pl.col(direction_col).alias("predicted_direction"),
        pl.col(label_col).cast(pl.Boolean).alias("correct_label"),
        pl.col(return_col).alias("gross_return_bp"),
        pl.col(entry_col).map_elements(
            lambda value: round_trip_cost_bp(float(value), execution),
            return_dtype=pl.Float64,
        ).alias("cost_bp"),
    ).with_columns(
        (pl.col("gross_return_bp") - pl.col("cost_bp")).alias("net_return_bp")
    )


def choose_params(train: pl.DataFrame, config: dict[str, Any], rule_id: str, family: str) -> RuleParams:
    best: RuleParams | None = None
    best_score = (-math.inf, -math.inf)
    threshold_grid_name = "volatility_threshold_quantile" if rule_id == BASELINE_RULE else "proxy_threshold_quantile"

    for obs in config["parameter_grid"]["observation_window_minutes"]:
        for horizon in config["parameter_grid"]["continuation_horizon_minutes"]:
            for mode in config["parameter_grid"]["window_family_mode"]:
                for quantile in config["parameter_grid"][threshold_grid_name]:
                    candidate = RuleParams(rule_id, obs, horizon, mode, quantile)
                    evaluated = evaluate_rule(train, candidate, config["execution"], family, "train")
                    active = evaluated.filter(pl.col("took_trade"))
                    accuracy = active.get_column("correct_label").mean() if not active.is_empty() else None
                    avg_return = active.get_column("net_return_bp").mean() if not active.is_empty() else None
                    score = (
                        float(accuracy) if accuracy is not None else -math.inf,
                        float(avg_return) if avg_return is not None else -math.inf,
                    )
                    if score > best_score:
                        best = candidate
                        best_score = score

    if best is None:
        raise RuntimeError(f"Could not choose parameters for {rule_id} in family {family}.")
    return best


def build_walk_forward(events: pl.DataFrame, config: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    train_years = int(config["walk_forward"]["train_years"])
    test_years = int(config["walk_forward"]["test_years"])
    years = sorted({value.year for value in events.get_column("event_date").to_list()})
    trades: list[pl.DataFrame] = []
    metadata: dict[str, Any] = {"windows": []}

    for start_idx in range(0, max(0, len(years) - train_years - test_years + 1)):
        train_year_set = years[start_idx : start_idx + train_years]
        test_year_set = years[start_idx + train_years : start_idx + train_years + test_years]
        if not test_year_set:
            continue

        split_id = len(metadata["windows"]) + 1
        train = events.filter(pl.col("event_date").dt.year().is_in(train_year_set))
        test = events.filter(pl.col("event_date").dt.year().is_in(test_year_set))
        if train.is_empty() or test.is_empty():
            continue

        window_meta = {"split_id": split_id, "train_years": train_year_set, "test_years": test_year_set, "selected_params": []}
        for family in ["08:30", "09:30"]:
            for rule_id in [BASELINE_RULE, TREATMENT_RULE]:
                family_train = train.filter(pl.col("window_family") == family)
                family_test = test.filter(pl.col("window_family") == family)
                if family_train.is_empty() or family_test.is_empty():
                    continue
                params = choose_params(family_train, config, rule_id, family)
                window_meta["selected_params"].append(params.__dict__ | {"family": family})
                trades.append(evaluate_rule(family_test, params, config["execution"], family, "test").with_columns(pl.lit(split_id).alias("split_id")))
        metadata["windows"].append(window_meta)

    if not trades:
        raise RuntimeError("Walk-forward produced no test trades. Check event coverage and training windows.")
    return pl.concat(trades, how="diagonal_relaxed"), metadata


def build_equity_curve(trades: pl.DataFrame) -> pl.DataFrame:
    active = trades.filter(pl.col("took_trade")).sort(["rule_id", "window_family", "event_date"])
    if active.is_empty():
        return pl.DataFrame({"rule_id": [], "window_family": [], "event_date": [], "equity_bp": [], "drawdown_bp": []})

    frames: list[pl.DataFrame] = []
    for key in active.select("rule_id", "window_family").unique().iter_rows():
        rule_id, family = key
        subset = active.filter((pl.col("rule_id") == rule_id) & (pl.col("window_family") == family)).sort("event_date")
        running = 0.0
        peak = 0.0
        equity = []
        drawdown = []
        for value in subset.get_column("net_return_bp").to_list():
            running += float(value)
            peak = max(peak, running)
            equity.append(running)
            drawdown.append(running - peak)
        frames.append(
            subset.select("rule_id", "window_family", "event_date", "net_return_bp").with_columns(
                pl.Series("equity_bp", equity),
                pl.Series("drawdown_bp", drawdown),
            )
        )
    return pl.concat(frames, how="vertical")


def summarise_metrics(trades: pl.DataFrame, equity_curve: pl.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"walk_forward": metadata["windows"], "rules": {}}
    for key in trades.select("rule_id", "window_family").unique().iter_rows(named=True):
        rule_id = key["rule_id"]
        family = key["window_family"]
        scoped = trades.filter((pl.col("rule_id") == rule_id) & (pl.col("window_family") == family))
        active = scoped.filter(pl.col("took_trade"))
        eq = equity_curve.filter((pl.col("rule_id") == rule_id) & (pl.col("window_family") == family))
        summary["rules"][f"{rule_id}:{family}"] = {
            "events_seen": scoped.height,
            "trades_taken": active.height,
            "trust_classification_accuracy": active.get_column("correct_label").mean() if not active.is_empty() else None,
            "average_trade_return_bp": active.get_column("net_return_bp").mean() if not active.is_empty() else None,
            "max_drawdown_bp": eq.get_column("drawdown_bp").min() if not eq.is_empty() else None,
        }

    for family in ["08:30", "09:30"]:
        base = summary["rules"].get(f"{BASELINE_RULE}:{family}")
        treat = summary["rules"].get(f"{TREATMENT_RULE}:{family}")
        if base and treat:
            treat["treatment_minus_baseline_accuracy"] = (
                (treat.get("trust_classification_accuracy") or 0.0)
                - (base.get("trust_classification_accuracy") or 0.0)
            )
    return summary


def write_report(path: Path, config: dict[str, Any], metrics: dict[str, Any], trades: pl.DataFrame) -> None:
    text = f"""# ES Replenishment Fragility Backtest

## Contract

- Strategy ID: {config["strategy"]["id"]}
- Source contract: {config["strategy"]["source_contract_path"]}
- Data source: {config["data"]["market_data"]["r2_key"]}

## Validation Assessment

The contract is materially testable, but this package still treats three contract gaps as explicit experiments:

1. Observation-slice sensitivity across 1, 3, and 5 minutes.
2. Outcome-horizon sensitivity across 5, 15, and 30 minutes.
3. Separate-vs-shared family handling for 08:30 and 09:30 windows.

## Interpretation Guidance

- If treatment only wins in one family, the strategy should be narrowed instead of merged.
- If treatment accuracy rises but returns do not, the proxy is classifying conditions better than it is monetizing them.
- If macro dates are missing, 08:30 results are intentionally skipped rather than invented.

## Run Notes

- Trades evaluated: {trades.filter(pl.col("took_trade")).height}
- Walk-forward windows: {len(metrics["walk_forward"])}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml(config_path)
    output_dir = ensure_output_dir(config, config_path)

    bars = load_bars(config)
    events = build_window_events(config, bars)
    trades, metadata = build_walk_forward(events, config)
    diagnostics = events
    equity_curve = build_equity_curve(trades)
    metrics = summarise_metrics(trades, equity_curve, metadata)

    trades.write_parquet(output_dir / "trades.parquet")
    diagnostics.write_parquet(output_dir / "diagnostics.parquet")
    equity_curve.write_parquet(output_dir / "equity_curve.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_report(output_dir / "report.md", config, metrics, trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
