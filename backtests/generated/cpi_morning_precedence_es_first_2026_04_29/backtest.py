#!/usr/bin/env python3
"""Backtest for CPI morning precedence ES-first strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml


TIMEZONE = "America/New_York"
NY_TZ = ZoneInfo(TIMEZONE)
EPSILON = 1e-9


@dataclass(frozen=True)
class ParameterChoice:
    branch_score_abs_min: float
    reentry_noise_ratio_max: float
    reentry_dispersion_stress_max: float
    stop_range_frac: float


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
    dtype = bars.schema.get("timestamp")
    expr = pl.col("timestamp")
    if dtype == pl.Utf8:
        expr = expr.str.to_datetime(strict=False)
    bars = bars.with_columns(expr.alias("timestamp"))
    if getattr(bars.schema["timestamp"], "time_zone", None):
        bars = bars.with_columns(pl.col("timestamp").dt.convert_time_zone(TIMEZONE))
    else:
        bars = bars.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone(TIMEZONE))
    return bars.sort("timestamp").with_columns(
        pl.col("timestamp").dt.date().alias("session_date"),
        pl.col("timestamp").dt.year().alias("event_year"),
    )


def load_csv_table(path: Path, name: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required {name} at {path.as_posix()}. Supply a real public input file before running."
        )
    return pl.read_csv(path, try_parse_dates=True)


def normalize_release_date_frame(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = frame.schema.get(column)
    expr = pl.col(column)
    if dtype == pl.Utf8:
        expr = expr.str.strptime(pl.Date, strict=False)
    elif dtype != pl.Date:
        expr = expr.cast(pl.Date)
    return frame.with_columns(expr.alias(column))


def load_inputs(config: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    inputs_cfg = config["inputs"]
    calendar = load_csv_table(Path(inputs_cfg["cpi_release_calendar"]["path"]), "CPI release calendar")
    components = load_csv_table(Path(inputs_cfg["cpi_component_table"]["path"]), "CPI component table")

    calendar = calendar.rename({inputs_cfg["cpi_release_calendar"]["date_column"]: "release_date"})
    calendar = normalize_release_date_frame(calendar, "release_date").drop_nulls()

    comp_map = inputs_cfg["cpi_component_table"]["columns"]
    components = components.rename(
        {
            comp_map["date"]: "release_date",
            comp_map["headline_mom"]: "headline_mom",
            comp_map["core_mom"]: "core_mom",
            comp_map["shelter_mom"]: "shelter_mom",
            comp_map["core_services_ex_shelter_mom"]: "core_services_ex_shelter_mom",
            comp_map["core_goods_mom"]: "core_goods_mom",
        }
    )
    components = normalize_release_date_frame(components, "release_date").with_columns(
        pl.col("headline_mom").cast(pl.Float64),
        pl.col("core_mom").cast(pl.Float64),
        pl.col("shelter_mom").cast(pl.Float64),
        pl.col("core_services_ex_shelter_mom").cast(pl.Float64),
        pl.col("core_goods_mom").cast(pl.Float64),
    ).drop_nulls()
    return calendar, components


def ensure_event_minutes(day_bars: pl.DataFrame, required_times: list[str]) -> bool:
    lookup = {ts.strftime("%H:%M") for ts in day_bars.get_column("timestamp").to_list()}
    return all(t in lookup for t in required_times)


def compute_event_rows(
    bars: pl.DataFrame,
    calendar: pl.DataFrame,
    components: pl.DataFrame,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    joined = calendar.join(components, on="release_date", how="inner").sort("release_date")
    if joined.is_empty():
        raise RuntimeError("No CPI events remained after joining the release calendar to the component table.")

    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event in joined.iter_rows(named=True):
        release_date = event["release_date"]
        day_bars = bars.filter(pl.col("session_date") == release_date)
        required = ["08:29", "08:30", "08:34", "09:29", "09:30", "10:00"]
        if day_bars.is_empty() or not ensure_event_minutes(day_bars, required):
            skipped.append({"release_date": str(release_date), "reason": "missing_required_bars"})
            continue

        bars_by_time = {ts.strftime("%H:%M"): row for ts, row in zip(day_bars["timestamp"], day_bars.to_dicts())}
        if any(t not in bars_by_time for t in required):
            skipped.append({"release_date": str(release_date), "reason": "missing_anchor_minutes"})
            continue

        bars_0830_0929 = day_bars.filter(
            ((pl.col("timestamp").dt.hour() == 8) & (pl.col("timestamp").dt.minute() >= 30))
            | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() <= 29))
        )
        reentry_window = day_bars.filter(
            ((pl.col("timestamp").dt.hour() == 8) & (pl.col("timestamp").dt.minute() >= 35))
            | ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() <= 29))
        )
        if bars_0830_0929.height < 60 or reentry_window.height < 55:
            skipped.append({"release_date": str(release_date), "reason": "incomplete_window"})
            continue

        component_branch_score = (
            0.5 * event["core_services_ex_shelter_mom"]
            + 0.5 * event["shelter_mom"]
            - 0.5 * event["core_goods_mom"]
        )
        direction = -1 if component_branch_score > 0 else (1 if component_branch_score < 0 else 0)
        close_0829 = float(bars_by_time["08:29"]["close"])
        close_0834 = float(bars_by_time["08:34"]["close"])
        close_0929 = float(bars_by_time["09:29"]["close"])
        open_0930 = float(bars_by_time["09:30"]["open"])
        close_1000 = float(bars_by_time["10:00"]["close"])
        reentry_returns = (
            reentry_window.select((pl.col("close") / pl.col("close").shift(1) - 1.0).abs().alias("r"))
            .drop_nulls()
            .get_column("r")
            .to_list()
        )
        net_displacement = abs(close_0929 / close_0834 - 1.0)
        reentry_noise_ratio = sum(float(x) for x in reentry_returns) / max(net_displacement, 0.0001)
        reentry_dispersion_stress = (
            float(bars_0830_0929.get_column("high").max()) - float(bars_0830_0929.get_column("low").min())
        ) / max(close_0829, EPSILON)
        branch_path_sign = 1 if close_0929 > close_0834 else (-1 if close_0929 < close_0834 else 0)
        branch_coherence_flag = int(branch_path_sign == direction)
        official_open_carry_return_points = direction * (close_1000 - open_0930)
        stop_reference_range = float(bars_0830_0929.get_column("high").max()) - float(bars_0830_0929.get_column("low").min())
        post_open_window = day_bars.filter(
            ((pl.col("timestamp").dt.hour() == 9) & (pl.col("timestamp").dt.minute() >= 30))
            | ((pl.col("timestamp").dt.hour() == 10) & (pl.col("timestamp").dt.minute() == 0))
        )
        adverse_excursion_points = (
            max(open_0930 - float(post_open_window.get_column("low").min()), 0.0)
            if direction > 0
            else max(float(post_open_window.get_column("high").max()) - open_0930, 0.0)
        )
        rows.append(
            {
                "release_date": release_date,
                "event_year": release_date.year,
                "headline_mom": float(event["headline_mom"]),
                "core_mom": float(event["core_mom"]),
                "shelter_mom": float(event["shelter_mom"]),
                "core_services_ex_shelter_mom": float(event["core_services_ex_shelter_mom"]),
                "core_goods_mom": float(event["core_goods_mom"]),
                "component_branch_score": float(component_branch_score),
                "provisional_macro_direction": direction,
                "reentry_noise_ratio": float(reentry_noise_ratio),
                "reentry_dispersion_stress": float(reentry_dispersion_stress),
                "branch_coherence_flag": branch_coherence_flag,
                "open_0930": open_0930,
                "close_1000": close_1000,
                "official_open_carry_return_points": float(official_open_carry_return_points),
                "official_open_carry_success": int(official_open_carry_return_points > 0.0),
                "stop_reference_range": float(stop_reference_range),
                "adverse_excursion_points": float(adverse_excursion_points),
            }
        )

    events = pl.DataFrame(rows).sort("release_date") if rows else pl.DataFrame(schema={"release_date": pl.Date})
    return events, skipped


def make_parameter_choices(config: dict[str, Any]) -> list[ParameterChoice]:
    grid = config["strategy"]["parameter_grid"]
    return [
        ParameterChoice(float(branch), float(noise), float(stress), float(stop))
        for branch in grid["branch_score_abs_min"]
        for noise in grid["reentry_noise_ratio_max"]
        for stress in grid["reentry_dispersion_stress_max"]
        for stop in grid["stop_range_frac"]
    ]


def apply_rule(events: pl.DataFrame, choice: ParameterChoice, rule_name: str) -> pl.DataFrame:
    base = (
        (pl.col("provisional_macro_direction") != 0)
        & (pl.col("component_branch_score").abs() >= choice.branch_score_abs_min)
    )
    if rule_name == "baseline":
        return events.with_columns(pl.lit(rule_name).alias("rule_name"), base.cast(pl.Int64).alias("selected"))
    treatment = (
        base
        & (pl.col("branch_coherence_flag") == 1)
        & (pl.col("reentry_noise_ratio") <= choice.reentry_noise_ratio_max)
        & (pl.col("reentry_dispersion_stress") <= choice.reentry_dispersion_stress_max)
    )
    return events.with_columns(pl.lit(rule_name).alias("rule_name"), treatment.cast(pl.Int64).alias("selected"))


def net_trade_pnl(event: dict[str, Any], choice: ParameterChoice, config: dict[str, Any]) -> float:
    direction = int(event["provisional_macro_direction"])
    if direction == 0:
        return 0.0
    tick_size = float(config["execution"]["tick_size"])
    point_value = float(config["execution"]["point_value"])
    commissions = float(config["execution"]["commissions_round_turn_usd"])
    slippage_ticks = float(config["execution"]["slippage_ticks_per_side"])
    stop_threshold = max(6.0, choice.stop_range_frac * float(event["stop_reference_range"]))
    gross_points = float(event["official_open_carry_return_points"])
    stopped = float(event["adverse_excursion_points"]) >= stop_threshold
    realized_points = -stop_threshold if stopped else gross_points
    slippage_points = 2.0 * slippage_ticks * tick_size
    return (realized_points - slippage_points) * point_value - commissions


def choose_best(train_events: pl.DataFrame, choices: list[ParameterChoice], config: dict[str, Any], rule_name: str) -> ParameterChoice:
    best = choices[0]
    best_score = (-10000.0, -10000.0)
    for choice in choices:
        scored = apply_rule(train_events, choice, rule_name).filter(pl.col("selected") == 1)
        if scored.is_empty():
            continue
        pnls = [net_trade_pnl(row, choice, config) for row in scored.iter_rows(named=True)]
        expectancy = sum(pnls) / len(pnls)
        win_rate = sum(1 for pnl in pnls if pnl > 0.0) / len(pnls)
        score = (expectancy, win_rate)
        if score > best_score:
            best = choice
            best_score = score
    return best


def make_walk_forward_splits(events: pl.DataFrame) -> list[dict[str, Any]]:
    years = sorted(events.get_column("event_year").unique().to_list())
    splits: list[dict[str, Any]] = []
    split_id = 1
    for idx in range(5, len(years) - 1):
        splits.append(
            {
                "split_id": split_id,
                "train_years": years[:idx],
                "validation_year": years[idx],
                "test_year": years[idx + 1],
            }
        )
        split_id += 1
    return splits


def build_trade_rows(scored: pl.DataFrame, choice: ParameterChoice, config: dict[str, Any], split_id: int) -> pl.DataFrame:
    selected = scored.filter(pl.col("selected") == 1)
    rows: list[dict[str, Any]] = []
    for event in selected.iter_rows(named=True):
        pnl = net_trade_pnl(event, choice, config)
        rows.append(
            {
                "split_id": split_id,
                "release_date": event["release_date"],
                "rule_name": event["rule_name"],
                "branch_score_abs_min": choice.branch_score_abs_min,
                "reentry_noise_ratio_max": choice.reentry_noise_ratio_max,
                "reentry_dispersion_stress_max": choice.reentry_dispersion_stress_max,
                "stop_range_frac": choice.stop_range_frac,
                "provisional_macro_direction": event["provisional_macro_direction"],
                "official_open_carry_return_points": event["official_open_carry_return_points"],
                "net_pnl_usd": pnl,
                "win_flag": int(pnl > 0.0),
                "branch_sign": "positive" if event["component_branch_score"] > 0 else "negative",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def summarize_trades(trades: pl.DataFrame) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trade_count": 0,
            "expectancy_per_trade": None,
            "win_rate": None,
            "sharpe_like_event_ratio": None,
            "max_drawdown": None,
        }
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    returns = trades.sort("release_date").get_column("net_pnl_usd").to_list()
    for pnl in returns:
        running += float(pnl)
        peak = max(peak, running)
        max_drawdown = min(max_drawdown, running - peak)
    mean_return = sum(returns) / len(returns)
    variance = sum((float(r) - mean_return) ** 2 for r in returns) / max(len(returns) - 1, 1)
    sharpe_like = mean_return / max(variance ** 0.5, EPSILON)
    return {
        "trade_count": trades.height,
        "expectancy_per_trade": mean_return,
        "win_rate": float(trades.get_column("win_flag").mean()),
        "sharpe_like_event_ratio": sharpe_like,
        "max_drawdown": max_drawdown,
    }


def build_equity_curve(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame()
    return trades.sort(["rule_name", "release_date"]).with_columns(
        pl.col("net_pnl_usd").cum_sum().over("rule_name").alias("equity_usd")
    )


def render_report(config: dict[str, Any], metrics: dict[str, Any], skipped: list[dict[str, Any]]) -> str:
    strategy_id = config["strategy"]["id"]
    contract_path = config["runtime"]["contract_path"]
    lines = [
        "# Strategy Contract Review & Backtesting Brief",
        "",
        "## 1. Contract Review Summary",
        "",
        "| Source | Key Insight | Relevance |",
        "| --- | --- | --- |",
        f"| `{contract_path}` | CPI component branch drives direction; reentry quality decides promotion or veto into the 09:30 carry. | Primary implementation source. |",
        "| `backtests/BACKTEST_LEDGER.md` | No equivalent completed package exists for the CPI morning precedence strategy. | Confirms `build_new` is the right action. |",
        "| `data/DATA_CONTRACT.md` and `data/r2_manifest.yaml` | ES 1-minute bars are available through the shared R2 helper. | Defines the runtime data path and canonical columns. |",
        "",
        "## 2. Strategy Interpretation",
        "",
        f"- Contract file: `{contract_path}`",
        "- Core thesis: CPI component direction alone is a useful 09:30 carry baseline, and a clean 08:30-to-open reentry path should improve that baseline by vetoing weak setups.",
        "- Assumptions: The package uses the contract's weighted component branch formula, a 09:30 open entry, a 10:00 close exit, and the contract's stop framework with the provided parameter grid.",
        "- Required inputs and outputs: ES 1-minute R2 bars, a real CPI release calendar CSV, a real CPI component CSV, and the standard trades/equity/metrics/diagnostics/report outputs.",
        "- Key constraints: The package does not invent BLS event data; missing public CPI files stop execution clearly.",
        "",
        "## 3. Build / Skip / Repair Decision",
        "",
        "- Chosen action: `build_new`",
        "- Decision evidence: The ledger and generated packages cover the ES replenishment-fragility family and MSCI review bridge, but not the CPI morning precedence strategy. The CPI contract is materially different in mechanism, required inputs, and validation design.",
        "",
        "## 4. Validation Assessment",
        "",
        "The contract is sufficiently specified for a first runnable backtest once real CPI release and component files are provided. The main remaining validation risk is whether the public reentry-quality proxies add value beyond the component-only baseline rather than just overfitting a small event sample.",
        "",
        "## 5. Added Experiments",
        "",
        "- Name: Threshold Stability Across Walk-Forward Splits",
        "  Purpose: Check whether the veto thresholds hold up across event chronology instead of only in one period.",
        "  Assumption being tested: The treatment filter is not just fitting one CPI regime.",
        "  Required data or inputs: The same ES bars and CPI inputs already required by the contract.",
        "  Success metric: Positive treatment lift versus baseline across multiple test splits.",
        "  Failure metric: Lift turns negative or disappears in most out-of-sample years.",
        "  Decision rule: Reject the veto layer if it cannot maintain positive out-of-sample lift.",
        "  Key limitation: CPI sample sizes are naturally small.",
        "- Name: Branch Sign Stratification",
        "  Purpose: Verify that positive and negative branch-score events are not hiding opposite behavior.",
        "  Assumption being tested: The mechanism is directionally symmetric enough to pool.",
        "  Required data or inputs: The same event-level backtest output.",
        "  Success metric: Similar treatment behavior across positive and negative branch-sign subsets.",
        "  Failure metric: One side carries the entire result.",
        "  Decision rule: Split interpretation by branch sign if the subsets diverge materially.",
        "  Key limitation: One sign may have fewer usable events.",
        "",
        "## 6. Backtesting Approach",
        "",
        "The backtest loads ES 1-minute bars through the shared R2 helper, joins a real CPI release calendar to a real CPI component table, computes the contract's component-branch and reentry-quality features for each CPI morning, evaluates the baseline and treatment rules across the supplied parameter grid, and uses expanding walk-forward yearly splits to compare out-of-sample expectancy, win rate, drawdown, and treatment lift.",
        "",
        "## 7. Backtesting Code or Repair Output",
        "",
        f"Run `python backtests/generated/{strategy_id}/backtest.py --config backtests/generated/{strategy_id}/config.yaml` from the repo root after placing the CPI calendar and component CSVs at the configured paths. The package stops with a clear missing-input error if those public files are absent.",
        "",
        "## 8. Immediate Next Actions",
        "",
        "- Place a real CPI release calendar CSV at the configured input path.",
        "- Place a real CPI component table CSV at the configured input path.",
        "- Run the package and inspect treatment lift versus baseline in `metrics.json`.",
        "- Check branch-sign splits in `diagnostics.parquet` before trusting aggregate results.",
        "- Treat any result with fewer than 40 out-of-sample treatment trades as inconclusive.",
        "",
        "## Runtime Notes",
        "",
        f"- Skipped events due to missing bars: {len(skipped)}",
        f"- Metrics summary: `{json.dumps(metrics, default=str)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = load_market_bars(config)
    calendar, components = load_inputs(config)
    events, skipped = compute_event_rows(bars, calendar, components)
    if events.is_empty():
        raise RuntimeError("No usable CPI events remained after applying the contract's data-quality rules.")

    splits = make_walk_forward_splits(events)
    if not splits:
        raise RuntimeError("Not enough event years are available to form the required expanding walk-forward splits.")

    choices = make_parameter_choices(config)
    diagnostics_parts: list[pl.DataFrame] = []
    trade_parts: list[pl.DataFrame] = []
    split_metrics: list[dict[str, Any]] = []

    for split in splits:
        train = events.filter(pl.col("event_year").is_in(split["train_years"]))
        test = events.filter(pl.col("event_year") == split["test_year"])
        if train.is_empty() or test.is_empty():
            continue
        split_summary = {"split_id": split["split_id"], "test_year": split["test_year"]}
        for rule_name in ["baseline", "treatment"]:
            choice = choose_best(train, choices, config, rule_name)
            scored = apply_rule(test, choice, rule_name)
            diagnostics_parts.append(
                scored.with_columns(
                    pl.lit(split["split_id"]).alias("split_id"),
                    pl.lit(choice.branch_score_abs_min).alias("branch_score_abs_min"),
                    pl.lit(choice.reentry_noise_ratio_max).alias("reentry_noise_ratio_max"),
                    pl.lit(choice.reentry_dispersion_stress_max).alias("reentry_dispersion_stress_max"),
                    pl.lit(choice.stop_range_frac).alias("stop_range_frac"),
                )
            )
            trades = build_trade_rows(scored, choice, config, split["split_id"])
            if not trades.is_empty():
                trade_parts.append(trades)
            split_summary[rule_name] = summarize_trades(trades)
        if (
            split_summary.get("baseline", {}).get("expectancy_per_trade") is not None
            and split_summary.get("treatment", {}).get("expectancy_per_trade") is not None
        ):
            split_summary["treatment_lift_vs_baseline"] = (
                split_summary["treatment"]["expectancy_per_trade"] - split_summary["baseline"]["expectancy_per_trade"]
            )
        else:
            split_summary["treatment_lift_vs_baseline"] = None
        split_metrics.append(split_summary)

    diagnostics = pl.concat(diagnostics_parts, how="vertical_relaxed") if diagnostics_parts else pl.DataFrame()
    trades = pl.concat(trade_parts, how="vertical_relaxed") if trade_parts else pl.DataFrame()
    equity_curve = build_equity_curve(trades)

    overall = {}
    for rule_name in ["baseline", "treatment"]:
        rule_trades = trades.filter(pl.col("rule_name") == rule_name) if not trades.is_empty() else pl.DataFrame()
        overall[rule_name] = summarize_trades(rule_trades)
        by_sign = {}
        for sign in ["positive", "negative"]:
            sign_trades = rule_trades.filter(pl.col("branch_sign") == sign) if not rule_trades.is_empty() else pl.DataFrame()
            by_sign[sign] = summarize_trades(sign_trades)
        overall[rule_name]["by_branch_sign"] = by_sign

    treatment_expectancy = overall["treatment"]["expectancy_per_trade"]
    baseline_expectancy = overall["baseline"]["expectancy_per_trade"]
    metrics = {
        "strategy_id": config["strategy"]["id"],
        "contract_path": config["runtime"]["contract_path"],
        "overall": overall,
        "treatment_lift_vs_baseline": (
            treatment_expectancy - baseline_expectancy
            if treatment_expectancy is not None and baseline_expectancy is not None
            else None
        ),
        "walk_forward_summaries": split_metrics,
        "skipped_events": skipped,
        "input_files": {
            "cpi_release_calendar": config["inputs"]["cpi_release_calendar"]["path"],
            "cpi_component_table": config["inputs"]["cpi_component_table"]["path"],
        },
    }

    diagnostics.write_parquet(output_dir / config["outputs"]["diagnostics"])
    trades.write_parquet(output_dir / config["outputs"]["trades"])
    equity_curve.write_parquet(output_dir / config["outputs"]["equity_curve"])
    (output_dir / config["outputs"]["metrics"]).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (output_dir / config["outputs"]["report"]).write_text(render_report(config, metrics, skipped), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
