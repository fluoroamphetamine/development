#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


COMPLETED_STATUSES = {"completed", "skipped", "needs_review", "missing_backtest_package"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_ledger_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    statuses: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "Strategy ID" in line or "---" in line:
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue

        strategy_id = cells[0]
        status = cells[2]
        if strategy_id:
            statuses[strategy_id] = status

    return statuses


def write_output(path: Path, has_strategy: bool, strategy_id: str = "", contract_path: str = "") -> None:
    content = "\n".join(
        [
            f"has_strategy={'true' if has_strategy else 'false'}",
            f"strategy_id={strategy_id}",
            f"contract_path={contract_path}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    strategies_dir = Path(args.strategies_dir)
    ledger_path = Path(args.ledger)
    output_path = Path(args.output)

    strategies_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    statuses = read_ledger_statuses(ledger_path)

    strategy_files = sorted(strategies_dir.glob("*.yaml")) + sorted(strategies_dir.glob("*.yml"))

    for strategy_file in strategy_files:
        strategy_id = strategy_file.stem
        status = statuses.get(strategy_id)

        if status not in COMPLETED_STATUSES:
            write_output(
                output_path,
                has_strategy=True,
                strategy_id=strategy_id,
                contract_path=strategy_file.as_posix(),
            )
            return

    write_output(output_path, has_strategy=False)


if __name__ == "__main__":
    main()
