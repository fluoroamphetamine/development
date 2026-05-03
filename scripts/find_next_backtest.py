#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


COMPLETED_STATUSES = {"completed", "skipped", "needs_review", "missing_backtest_package"}
DEFAULT_CONTRACT_DIRS = [
    "strategies/pending",
    "strategies/contracts",
    "strategy_contracts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies-dir", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-dir", default="backtests/generated")
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


def discover_contracts(primary_dir: Path) -> dict[str, Path]:
    """Return strategy_id -> contract path from known contract directories."""

    dirs = [primary_dir]
    for raw_dir in DEFAULT_CONTRACT_DIRS:
        candidate = Path(raw_dir)
        if candidate not in dirs:
            dirs.append(candidate)

    contracts: dict[str, Path] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        for pattern in ("*.yaml", "*.yml"):
            for strategy_file in sorted(directory.glob(pattern)):
                if strategy_file.name.startswith("."):
                    continue
                contracts.setdefault(strategy_file.stem, strategy_file)

    return contracts


def discover_generated_packages(generated_dir: Path) -> set[str]:
    """Return strategy IDs with generated backtest.py packages."""

    if not generated_dir.exists():
        return set()

    strategy_ids: set[str] = set()
    for child in sorted(generated_dir.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir() and (child / "backtest.py").exists():
            strategy_ids.add(child.name)

    return strategy_ids


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
    generated_dir = Path(args.generated_dir)
    ledger_path = Path(args.ledger)
    output_path = Path(args.output)

    strategies_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    statuses = read_ledger_statuses(ledger_path)
    contracts = discover_contracts(strategies_dir)
    generated_packages = discover_generated_packages(generated_dir)

    candidate_ids = sorted(set(contracts) | generated_packages)

    print(f"Discovered contract IDs: {sorted(contracts)}")
    print(f"Discovered generated package IDs: {sorted(generated_packages)}")
    print(f"Candidate IDs: {candidate_ids}")

    for strategy_id in candidate_ids:
        status = statuses.get(strategy_id)
        if status in COMPLETED_STATUSES:
            continue

        contract_path = contracts.get(strategy_id)
        write_output(
            output_path,
            has_strategy=True,
            strategy_id=strategy_id,
            contract_path=contract_path.as_posix() if contract_path else "",
        )
        print(
            "Selected strategy "
            f"{strategy_id} with contract_path={contract_path.as_posix() if contract_path else '<none>'}"
        )
        return

    write_output(output_path, has_strategy=False)
    print("No pending strategies discovered.")


if __name__ == "__main__":
    main()
