#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--contract-path", default="")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def ensure_fallback_event_calendars(config: dict[str, Any], log_path: Path) -> None:
    """Create minimal event calendars when a generated package omitted them.

    The macro calendar is left intentionally empty because exact macro-release
    dates should not be invented. The cash-open calendar is generated as
    weekdays across the current ES R2 data range so 09:30 tests can run.
    """

    calendars = config.get("event_calendars") or {}
    messages: list[str] = []

    macro = calendars.get("macro_release_calendar") or {}
    macro_path = Path(macro.get("path", "")) if macro.get("path") else None
    if macro_path and not macro_path.exists():
        macro_path.parent.mkdir(parents=True, exist_ok=True)
        macro_path.write_text("event_date\n", encoding="utf-8")
        messages.append(f"Created empty macro calendar placeholder: {macro_path}")

    cash = calendars.get("cash_open_calendar") or {}
    cash_path = Path(cash.get("path", "")) if cash.get("path") else None
    if cash_path and not cash_path.exists():
        cash_path.parent.mkdir(parents=True, exist_ok=True)
        start = date(2010, 6, 6)
        end = date(2026, 3, 15)
        lines = ["event_date"]
        current = start
        while current <= end:
            if current.weekday() < 5:
                lines.append(current.isoformat())
            current += timedelta(days=1)
        cash_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        messages.append(f"Created weekday cash-open calendar: {cash_path}")

    if messages:
        with log_path.open("a", encoding="utf-8") as log_file:
            for message in messages:
                log_file.write(message + "\n")


def write_runtime_config(
    config_yaml: Path,
    output_dir: Path,
    contract_path: Path | None,
    log_path: Path,
) -> Path:
    """Copy generated config and inject workflow runtime paths."""

    config = load_yaml(config_yaml)
    ensure_fallback_event_calendars(config, log_path)

    runtime = config.setdefault("runtime", {})
    runtime["output_dir"] = output_dir.as_posix()
    if contract_path is not None:
        runtime["contract_path"] = contract_path.as_posix()

    runtime_config = output_dir / "config.runtime.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return runtime_config


def main() -> int:
    args = parse_args()

    strategy_id = args.strategy_id
    contract_path = Path(args.contract_path) if args.contract_path else None
    output_dir = Path(args.output_dir)
    package_dir = Path("backtests/generated") / strategy_id
    backtest_py = package_dir / "backtest.py"
    config_yaml = package_dir / "config.yaml"
    requirements = package_dir / "requirements.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_path.write_text("", encoding="utf-8")

    if not backtest_py.exists():
        (output_dir / "missing_backtest_package.flag").write_text(
            f"Missing generated backtest: {backtest_py}\n",
            encoding="utf-8",
        )
        log_path.write_text(f"Missing generated backtest: {backtest_py}\n", encoding="utf-8")
        return 3

    if contract_path is not None and not contract_path.exists():
        log_path.write_text(
            f"Contract path was provided but does not exist: {contract_path}\n",
            encoding="utf-8",
        )
        return 2

    if requirements.exists():
        install_cmd = [sys.executable, "-m", "pip", "install", "-r", requirements.as_posix()]
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"Installing requirements: {' '.join(install_cmd)}\n")
            subprocess.run(install_cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    cmd = [sys.executable, backtest_py.as_posix()]

    if config_yaml.exists():
        runtime_config = write_runtime_config(
            config_yaml=config_yaml,
            output_dir=output_dir,
            contract_path=contract_path,
            log_path=log_path,
        )
        cmd.extend(["--config", runtime_config.as_posix()])
    else:
        cmd.extend(["--output-dir", output_dir.as_posix()])
        if contract_path is not None:
            cmd.extend(["--contract", contract_path.as_posix()])

    env = os.environ.copy()

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Running command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
