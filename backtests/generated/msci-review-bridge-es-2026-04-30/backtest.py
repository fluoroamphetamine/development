#!/usr/bin/env python3
"""Backtest for the MSCI review bridge ES strategy contract.

This implementation follows the contract's first executable assumptions:
- ES-only
- continuation direction from the chosen checkpoint
- R2-backed 1-minute OHLCV via backtests.lib.r2_data

It also makes the minimum explicit choices needed to be runnable:
- month-end overlap is approximated as the last 2 business days of a month
- month-end overlap policy is treated as a tunable filter
- checkpoint execution happens on the first minute after the checkpoint closes
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import polars as pl
import yaml


TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class RuleChoice:
    rule_name: str
    bridge_end_time: str | None
    month_end_overlap_policy: str
    minimum_bridge_absorption_ratio: float | None
    next_open_exit_variant: str


@dataclass(frozen=True)
class WalkForwardWindow:
    split_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def read_event_schedule(config: dict[str, Any]) -> pl.DataFrame:
    event_cfg = config["event_data"]["msci_review_schedule"]
    schedule_path = Path(event_cfg["path"])
    fmt = event_cfg.get("format", "csv")
    date_column = event_cfg.get("date_column", "review_date")

    if fmt == "csv":
        schedule = pl.read_csv(schedule_path)
    elif fmt == "parquet":
        schedule = pl.read_parquet(schedule_path)
    else:
        raise ValueError(f"Unsupported schedule format: {fmt}")

    if date_column not in schedule.columns:
        raise ValueError(
            f"MSCI review schedule is missing '{date_column}'. "
            f"Available columns: {schedule.columns}"
        )

    schedule = schedule.select(pl.col(date_column).alias("review_date"))
    if schedule.schema.get("review_date") != pl.Date:
        schedule = schedule.with_columns(
            pl.col("review_date").cast(pl.Utf8).str.strptime(pl.Date, strict=False)
        )
    return schedule.drop_nulls().unique().sort("review_date")


def to_minutes(time_text: str) -> int:
    hour, minute = time_text.split(":")
    return int(hour) * 60 + int(minute)


def remaining_business_days_in_month(anchor: date) -> int:
    current = anchor
    remaining = 0
    while True:
        nxt = date.fromordinal(current.toordinal() + 1)
        if nxt.month != anchor.month:
            break
        current = nxt
        if current.weekday() < 5:
            remaining += 1
    return remaining


def make_month_end_overlap_tag(review_date: date, overlap_business_days: int) -> bool:
    if review_date.weekday() >= 5:
        return False
    return remaining_business_days_in_month(review_date) < overlap_business_days


def signed_return_bp(entry_price: float, exit_price: float, direction: int) -> float:
    raw = (exit_price / entry_price) - 1.0
    return 10000.0 * raw * float(direction)


def roundtrip_cost_bp(entry_price: float, config: dict[str, Any]) -> float:
    execution = config["execution"]
    tick_size = float(execution["tick_size"])
    point_value = float(execution["point_value"])
    slippage_ticks_per_side = float(execution["slippage_ticks_per_side"])
    commissions_per_side_usd = float(execution["commissions_per_side_usd"])

    slippage_points = slippage_ticks_per_side * tick_size * 2.0
    slippage_usd = slippage_points * point_value
    total_cost_usd = slippage_usd + (2.0 * commissions_per_side_usd)
    notional_usd = entry_price * point_value
    return 10000.0 * total_cost_usd / notional_usd


def safe_sign(value: float | None) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def load_market_bars(config: dict[str, Any]) -> pl.DataFrame:
    from backtests.lib.r2_data import load_market_data_from_config, normalize_ohlcv_columns

    bars = load_market_data_from_config(config)
    bars = normalize_ohlcv_columns(bars, config)
    if "timestamp" not in bars.columns:
        raise ValueError("Normalized bars are missing the canonical 'timestamp' column.")

    bars = bars.with_columns(
        pl.col("timestamp").dt.convert_time_zone(TIMEZONE).alias("timestamp_et")
    )
    return bars.sort("timestamp_et")


def build_price_table(bars: pl.DataFrame) -> pl.DataFrame:
    return (
        bars.select(
            pl.col("timestamp_et").dt.date().alias("session_date"),
            (
                pl.col("timestamp_et").dt.hour() * 60
                + pl.col("timestamp_et").dt.minute()
            ).alias("minute_of_day"),
            "open",
            "close",
            "volume",
        )
        .group_by(["session_date", "minute_of_day"], maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        )
    )


def build_session_map(price_table: pl.DataFrame) -> dict[tuple[date, int], dict[str, float]]:
    rows = price_table.iter_rows(named=True)
    session_map: dict[tuple[date, int], dict[str, float]] = {}
    for row in rows:
        session_map[(row["session_date"], row["minute_of_day"])] = {
            "open": float(row["open"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
    return session_map


def extract_event_rows(
    config: dict[str, Any],
    bars: pl.DataFrame,
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    overlap_business_days = int(config["event_data"].get("month_end_overlap_business_days", 2))
    price_table = build_price_table(bars)
    session_map = build_session_map(price_table)
    available_dates = sorted({key[0] for key in session_map if key[1] == to_minutes("09:30")})

    next_session_lookup: dict[date, date] = {}
    for idx, session_date in enumerate(available_dates[:-1]):
        next_session_lookup[session_date] = available_dates[idx + 1]

    bridge_end_times = config["parameters"]["bridge_end_times"]
    rows: list[dict[str, Any]] = []

    for schedule_row in schedule.iter_rows(named=True):
        review_date = schedule_row["review_date"]
        if review_date not in next_session_lookup:
            continue
        next_date = next_session_lookup[review_date]

        key_times = [
            "16:00",
            "16:05",
            "16:06",
            "16:15",
            "16:16",
            "16:20",
            "16:21",
            "16:30",
            "16:31",
        ]
        exit_times = ["09:30", "09:35", "10:00"]
        missing = False

        def get_px(day: date, time_text: str, field: str) -> float | None:
            return session_map.get((day, to_minutes(time_text)), {}).get(field)

        for time_text in key_times:
            if get_px(review_date, time_text, "open") is None or get_px(review_date, time_text, "close") is None:
                missing = True
                break
        for time_text in exit_times:
            if get_px(next_date, time_text, "open") is None:
                missing = True
                break
        if missing:
            continue

        close_1600 = get_px(review_date, "16:00", "close")
        close_1605 = get_px(review_date, "16:05", "close")
        assert close_1600 is not None and close_1605 is not None
        first5_return_bp = 10000.0 * ((close_1605 / close_1600) - 1.0)
        baseline_direction = safe_sign(first5_return_bp)
        if baseline_direction == 0:
            continue

        record: dict[str, Any] = {
            "review_date": review_date,
            "next_session_date": next_date,
            "month_end_overlap_tag": make_month_end_overlap_tag(review_date, overlap_business_days),
            "event_date_regime": "post_2020" if review_date.year >= 2020 else "pre_2020",
            "first5_return_bp": first5_return_bp,
            "baseline_direction": baseline_direction,
            "baseline_entry_time": "16:06",
            "baseline_entry_price": get_px(review_date, "16:06", "open"),
            "next_open_price": get_px(next_date, "09:30", "open"),
            "exit_open_0930": get_px(next_date, "09:30", "open"),
            "exit_open_0935": get_px(next_date, "09:35", "open"),
            "exit_open_1000": get_px(next_date, "10:00", "open"),
        }
        record["first5_vs_next_open_alignment"] = int(
            baseline_direction == safe_sign(record["next_open_price"] - close_1605)
        )

        for bridge_end_time in bridge_end_times:
            bridge_close = get_px(review_date, bridge_end_time, "close")
            bridge_entry_time = f"{int(bridge_end_time[:2]):02d}:{int(bridge_end_time[3:]) + 1:02d}"
            bridge_entry_price = get_px(review_date, bridge_entry_time, "open")
            if bridge_close is None or bridge_entry_price is None:
                continue
            bridge_return_bp = 10000.0 * ((bridge_close / close_1605) - 1.0)
            bridge_direction = safe_sign(bridge_return_bp)
            denom = abs(close_1605 - close_1600)
            absorption = None if denom == 0 else abs(bridge_close - close_1605) / denom
            record[f"bridge_close_{bridge_end_time}"] = bridge_close
            record[f"bridge_entry_price_{bridge_end_time}"] = bridge_entry_price
            record[f"bridge_return_bp_{bridge_end_time}"] = bridge_return_bp
            record[f"bridge_direction_{bridge_end_time}"] = bridge_direction
            record[f"bridge_absorption_ratio_{bridge_end_time}"] = absorption
            record[f"bridge_vs_next_open_alignment_{bridge_end_time}"] = int(
                bridge_direction != 0 and bridge_direction == safe_sign(record["next_open_price"] - bridge_close)
            )

        rows.append(record)

    if not rows:
        return pl.DataFrame(schema={"review_date": pl.Date})
    return pl.DataFrame(rows).sort("review_date")


def apply_overlap_policy(events: pl.DataFrame, policy: str) -> pl.DataFrame:
    if policy == "all_events":
        return events
    if policy == "exclude_overlap":
        return events.filter(~pl.col("month_end_overlap_tag"))
    if policy == "stratify_overlap":
        return events
    raise ValueError(f"Unsupported month_end_overlap_policy: {policy}")


def score_choice(
    train_events: pl.DataFrame,
    choice: RuleChoice,
    config: dict[str, Any],
) -> tuple[float, float, int]:
    evaluated = build_trades_for_choice(train_events, choice, for_scoring=True)
    if evaluated.is_empty():
        return (-1.0, float("-inf"), 0)
    evaluated = compute_trade_metrics(evaluated, config, split_id=0)
    return (
        float(evaluated["alignment_hit"].mean()),
        float(evaluated["net_return_bp"].mean()),
        evaluated.height,
    )


def choose_best_parameters(train_events: pl.DataFrame, rule_name: str, config: dict[str, Any]) -> RuleChoice:
    policies = config["parameters"]["month_end_overlap_policies"]
    exit_variants = config["parameters"]["next_open_exit_variants"]

    candidates: list[RuleChoice] = []
    if rule_name == "baseline":
        for policy, exit_variant in product(policies, exit_variants):
            candidates.append(
                RuleChoice(
                    rule_name=rule_name,
                    bridge_end_time=None,
                    month_end_overlap_policy=policy,
                    minimum_bridge_absorption_ratio=None,
                    next_open_exit_variant=exit_variant,
                )
            )
    else:
        bridge_end_times = config["parameters"]["bridge_end_times"]
        absorption_thresholds = config["parameters"]["minimum_bridge_absorption_ratios"]
        for policy, exit_variant, bridge_end_time, absorption in product(
            policies,
            exit_variants,
            bridge_end_times,
            absorption_thresholds,
        ):
            candidates.append(
                RuleChoice(
                    rule_name=rule_name,
                    bridge_end_time=bridge_end_time,
                    month_end_overlap_policy=policy,
                    minimum_bridge_absorption_ratio=float(absorption),
                    next_open_exit_variant=exit_variant,
                )
            )

    best_choice = candidates[0]
    best_score = (-1.0, float("-inf"), 0)
    for candidate in candidates:
        candidate_score = score_choice(train_events, candidate, config)
        if candidate_score > best_score:
            best_score = candidate_score
            best_choice = candidate
    return best_choice


def build_trades_for_choice(events: pl.DataFrame, choice: RuleChoice, *, for_scoring: bool) -> pl.DataFrame:
    scoped = apply_overlap_policy(events, choice.month_end_overlap_policy)
    if scoped.is_empty():
        return pl.DataFrame()

    if choice.rule_name == "baseline":
        trades = scoped.filter(pl.col("baseline_direction") != 0).with_columns(
            pl.lit("baseline").alias("rule_name"),
            pl.col("baseline_direction").alias("direction"),
            pl.col("baseline_entry_price").alias("entry_price"),
            pl.lit("16:06").alias("entry_time"),
            pl.col("first5_vs_next_open_alignment").alias("alignment_hit"),
        )
    else:
        bridge = choice.bridge_end_time
        assert bridge is not None
        absorption_threshold = float(choice.minimum_bridge_absorption_ratio or 0.0)
        trades = scoped.filter(
            pl.col(f"bridge_direction_{bridge}") != 0,
            pl.col(f"bridge_absorption_ratio_{bridge}").fill_null(-1.0) >= absorption_threshold,
        ).with_columns(
            pl.lit("treatment").alias("rule_name"),
            pl.col(f"bridge_direction_{bridge}").alias("direction"),
            pl.col(f"bridge_entry_price_{bridge}").alias("entry_price"),
            pl.lit(f"{bridge}->+1m").alias("entry_time"),
            pl.col(f"bridge_vs_next_open_alignment_{bridge}").alias("alignment_hit"),
        )

    if trades.is_empty():
        return trades

    exit_column = {
        "09:30": "exit_open_0930",
        "09:35": "exit_open_0935",
        "10:00": "exit_open_1000",
    }[choice.next_open_exit_variant]
    trades = trades.with_columns(
        pl.col(exit_column).alias("exit_price"),
        pl.lit(choice.next_open_exit_variant).alias("exit_variant"),
        pl.lit(choice.month_end_overlap_policy).alias("month_end_overlap_policy"),
        pl.lit(choice.bridge_end_time).alias("bridge_end_time"),
        pl.lit(choice.minimum_bridge_absorption_ratio).alias("minimum_bridge_absorption_ratio"),
    )

    if for_scoring:
        return trades
    return trades


def compute_trade_metrics(trades: pl.DataFrame, config: dict[str, Any], split_id: int) -> pl.DataFrame:
    if trades.is_empty():
        return trades
    return trades.with_columns(
        pl.lit(split_id).alias("split_id"),
        pl.struct(["entry_price", "exit_price", "direction"]).map_elements(
            lambda row: signed_return_bp(
                float(row["entry_price"]),
                float(row["exit_price"]),
                int(row["direction"]),
            ),
            return_dtype=pl.Float64,
        ).alias("gross_return_bp"),
    ).with_columns(
        pl.struct(["entry_price", "exit_price", "direction"]).map_elements(
            lambda row: (
                float(row["exit_price"]) - float(row["entry_price"])
            ) * float(row["direction"]) * float(config["execution"]["point_value"]),
            return_dtype=pl.Float64,
        ).alias("gross_pnl_usd"),
        pl.col("entry_price").map_elements(
            lambda value: roundtrip_cost_bp(float(value), config),
            return_dtype=pl.Float64,
        ).alias("roundtrip_cost_bp"),
    ).with_columns(
        (pl.col("gross_return_bp") - pl.col("roundtrip_cost_bp")).alias("net_return_bp"),
        (
            pl.col("gross_pnl_usd")
            - ((pl.col("roundtrip_cost_bp") / 10000.0) * pl.col("entry_price") * float(config["execution"]["point_value"]))
        ).alias("net_pnl_usd"),
    )


def make_walk_forward_windows(event_count: int, config: dict[str, Any]) -> list[WalkForwardWindow]:
    wf = config["walk_forward"]
    train_events = int(wf["train_events"])
    test_events = int(wf["test_events"])
    step_events = int(wf["step_events"])

    windows: list[WalkForwardWindow] = []
    split_id = 1
    train_start = 0
    while train_start + train_events + test_events <= event_count:
        windows.append(
            WalkForwardWindow(
                split_id=split_id,
                train_start=train_start,
                train_end=train_start + train_events,
                test_start=train_start + train_events,
                test_end=train_start + train_events + test_events,
            )
        )
        train_start += step_events
        split_id += 1
    return windows


def summarize_by_rule(trades: pl.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for rule_name in ["baseline", "treatment"]:
        scoped = trades.filter(pl.col("rule_name") == rule_name)
        if scoped.is_empty():
            summary[rule_name] = {
                "trade_count": 0,
                "alignment_rate": None,
                "average_net_return_bp": None,
                "average_net_pnl_usd": None,
            }
            continue
        equity = scoped.sort("review_date").with_columns(
            pl.col("net_return_bp").cum_sum().alias("cum_net_return_bp")
        )
        peak = equity["cum_net_return_bp"].cum_max()
        drawdown = equity["cum_net_return_bp"] - peak
        summary[rule_name] = {
            "trade_count": scoped.height,
            "alignment_rate": float(scoped["alignment_hit"].mean()),
            "average_net_return_bp": float(scoped["net_return_bp"].mean()),
            "average_net_pnl_usd": float(scoped["net_pnl_usd"].mean()),
            "max_drawdown_bp": float(drawdown.min()),
        }
    baseline = summary["baseline"]
    treatment = summary["treatment"]
    summary["treatment_minus_baseline_alignment"] = None
    if baseline["alignment_rate"] is not None and treatment["alignment_rate"] is not None:
        summary["treatment_minus_baseline_alignment"] = (
            treatment["alignment_rate"] - baseline["alignment_rate"]
        )
    return summary


def build_diagnostics(events: pl.DataFrame, selected_choices: list[dict[str, Any]]) -> pl.DataFrame:
    if events.is_empty():
        return pl.DataFrame()
    return events.select(
        "review_date",
        "month_end_overlap_tag",
        "event_date_regime",
        "first5_return_bp",
        "first5_vs_next_open_alignment",
    )


def write_report(
    output_dir: Path,
    strategy_id: str,
    trades: pl.DataFrame,
    metrics: dict[str, Any],
    selected_choices: list[dict[str, Any]],
    experiments: list[dict[str, str]],
) -> None:
    lines = [
        f"# {strategy_id}",
        "",
        "## Contract interpretation",
        "- Baseline: trade continuation of the 16:00-16:05 move and hold to the chosen next-session exit.",
        "- Treatment: trade continuation of the broader post-close bridge and require retained absorption before entry.",
        "- Explicit fallback assumption: month-end overlap is approximated as the last 2 business days of the month.",
        "",
        "## Added validation experiments",
    ]
    for experiment in experiments:
        lines.extend(
            [
                f"### {experiment['name']}",
                f"- Purpose: {experiment['purpose']}",
                f"- Assumption: {experiment['assumption']}",
                f"- Success metric: {experiment['success_metric']}",
                f"- Failure metric: {experiment['failure_metric']}",
                f"- Decision rule: {experiment['decision_rule']}",
                f"- Limitation: {experiment['limitation']}",
                "",
            ]
        )

    lines.append("## Walk-forward summary")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Selected parameters")
    for choice in selected_choices:
        lines.append(f"- Split {choice['split_id']} {choice['rule_name']}: {json.dumps(choice, sort_keys=True)}")
    lines.append("")
    lines.append("## Output notes")
    if trades.is_empty():
        lines.append("- No trades were generated. This usually means the MSCI schedule file is still empty or no events survived the filters.")
    else:
        lines.append(f"- Trades written: {trades.height}")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    schedule = read_event_schedule(config)
    if schedule.is_empty():
        empty = pl.DataFrame({"review_date": [], "rule_name": [], "net_return_bp": []}, schema_overrides={
            "review_date": pl.Date,
            "rule_name": pl.Utf8,
            "net_return_bp": pl.Float64,
        })
        empty.write_parquet(output_dir / "trades.parquet")
        empty.write_parquet(output_dir / "equity_curve.parquet")
        empty.write_parquet(output_dir / "diagnostics.parquet")
        metrics = {
            "status": "missing_schedule_data",
            "reason": "The packaged MSCI review schedule template is empty. Replace it with real review dates to run the backtest against R2 market data.",
            "required_inputs": [
                "MSCI review schedule CSV or Parquet with a review_date column",
                "R2_ENDPOINT",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
                "R2_BUCKET",
            ],
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_report(
            output_dir=output_dir,
            strategy_id=config["strategy"]["id"],
            trades=empty,
            metrics=metrics,
            selected_choices=[],
            experiments=build_added_experiments(),
        )
        return 0

    bars = load_market_bars(config)
    events = extract_event_rows(config, bars, schedule)

    if events.is_empty():
        empty = pl.DataFrame({"review_date": [], "rule_name": [], "net_return_bp": []}, schema_overrides={
            "review_date": pl.Date,
            "rule_name": pl.Utf8,
            "net_return_bp": pl.Float64,
        })
        empty.write_parquet(output_dir / "trades.parquet")
        empty.write_parquet(output_dir / "equity_curve.parquet")
        empty.write_parquet(output_dir / "diagnostics.parquet")
        metrics = {
            "status": "no_events",
            "reason": "The MSCI review schedule file is empty or the required event windows are missing from the ES dataset.",
            "required_inputs": [
                "Validated MSCI review schedule dates",
                "ES 1-minute R2 dataset with 15:55-next-session 10:00 coverage on retained dates",
            ],
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        write_report(
            output_dir=output_dir,
            strategy_id=config["strategy"]["id"],
            trades=empty,
            metrics=metrics,
            selected_choices=[],
            experiments=build_added_experiments(),
        )
        return 0

    windows = make_walk_forward_windows(events.height, config)
    all_trades: list[pl.DataFrame] = []
    selected_choices: list[dict[str, Any]] = []

    for window in windows:
        train = events.slice(window.train_start, window.train_end - window.train_start)
        test = events.slice(window.test_start, window.test_end - window.test_start)
        baseline_choice = choose_best_parameters(train, "baseline", config)
        treatment_choice = choose_best_parameters(train, "treatment", config)

        for choice in [baseline_choice, treatment_choice]:
            trades = build_trades_for_choice(test, choice, for_scoring=False)
            trades = compute_trade_metrics(trades, config, split_id=window.split_id)
            if not trades.is_empty():
                all_trades.append(trades)
            selected_choices.append(
                {
                    "split_id": window.split_id,
                    "rule_name": choice.rule_name,
                    "bridge_end_time": choice.bridge_end_time,
                    "month_end_overlap_policy": choice.month_end_overlap_policy,
                    "minimum_bridge_absorption_ratio": choice.minimum_bridge_absorption_ratio,
                    "next_open_exit_variant": choice.next_open_exit_variant,
                }
            )

    trades = pl.concat(all_trades, how="vertical_relaxed") if all_trades else pl.DataFrame(
        {
            "review_date": [],
            "rule_name": [],
            "alignment_hit": [],
            "net_return_bp": [],
            "net_pnl_usd": [],
        },
        schema_overrides={
            "review_date": pl.Date,
            "rule_name": pl.Utf8,
            "alignment_hit": pl.Int64,
            "net_return_bp": pl.Float64,
            "net_pnl_usd": pl.Float64,
        },
    )
    if not trades.is_empty():
        trades = trades.sort(["review_date", "rule_name", "split_id"])
        equity_curve = trades.sort(["rule_name", "review_date"]).with_columns(
            pl.col("net_return_bp").cum_sum().over("rule_name").alias("cum_net_return_bp")
        ).with_columns(
            pl.col("cum_net_return_bp").cum_max().over("rule_name").alias("cum_peak_bp")
        ).with_columns(
            (pl.col("cum_net_return_bp") - pl.col("cum_peak_bp")).alias("drawdown_bp")
        )
    else:
        equity_curve = pl.DataFrame()

    metrics = {
        "strategy_id": config["strategy"]["id"],
        "contract_path": config["strategy"]["contract_path"],
        "walk_forward_windows": len(windows),
        "event_count": events.height,
        "rule_summary": summarize_by_rule(trades) if not trades.is_empty() else {},
        "selected_choices": selected_choices,
    }

    diagnostics = build_diagnostics(events, selected_choices)

    trades.write_parquet(output_dir / "trades.parquet")
    equity_curve.write_parquet(output_dir / "equity_curve.parquet")
    diagnostics.write_parquet(output_dir / "diagnostics.parquet")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    write_report(
        output_dir=output_dir,
        strategy_id=config["strategy"]["id"],
        trades=trades,
        metrics=metrics,
        selected_choices=selected_choices,
        experiments=build_added_experiments(),
    )
    return 0


def build_added_experiments() -> list[dict[str, str]]:
    return [
        {
            "name": "Month-end overlap sensitivity",
            "purpose": "Test whether the apparent bridge edge survives different contamination handling around month-end flows.",
            "assumption": "The signal is MSCI-transfer specific, not mostly ordinary month-end rebalance noise.",
            "success_metric": "Treatment-minus-baseline alignment remains positive on non-overlap events and under exclusion.",
            "failure_metric": "The positive effect disappears once overlap dates are excluded.",
            "decision_rule": "Keep the bridge thesis only if non-overlap results stay directionally positive after costs.",
            "limitation": "The overlap tag is still a public-calendar heuristic until the repo freezes an exact tagging rule.",
        },
        {
            "name": "Execution timing sensitivity",
            "purpose": "Measure whether the edge belongs to the bridge checkpoint itself or only to a fragile overnight/open print.",
            "assumption": "The bridge contains transfer information before the next official cash open, not just at one exit print.",
            "success_metric": "Results stay stable across 09:30, 09:35, and 10:00 exit variants.",
            "failure_metric": "Performance is concentrated in one print and collapses on nearby exits.",
            "decision_rule": "Reject the bridge-first trade framing if the apparent edge vanishes outside a single exit timestamp.",
            "limitation": "This still does not resolve the separate question of entering at 16:30 versus deferring entry to the next open.",
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
