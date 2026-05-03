#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


def write_runtime_config(
    config_yaml: Path,
    output_dir: Path,
    contract_path: Path | None,
) -> Path:
    """Copy generated config and inject workflow runtime paths.

    Some generated backtests only accept `--config`, not `--output-dir` or
    `--contract`. This wrapper normalizes those values into a temporary config
    file so generated backtests can remain simple.
    """

    config = load_yaml(config_yaml)
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
        )
        cmd.extend(["--config", runtime_config.as_posix()])
    else:
        # Fallback for future generated scripts that do not use config.yaml.
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
