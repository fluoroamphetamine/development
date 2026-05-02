#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


HEADER = """# Backtest Ledger

| Strategy ID | Contract Path | Status | Last Run UTC | Result Path | Workflow Run | Notes |
|---|---|---|---|---|---|---|
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--workflow-run", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def parse_existing_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "Strategy ID" in line or "---" in line:
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 7:
            rows.append(cells[:7])

    return rows


def render_rows(rows: list[list[str]]) -> str:
    lines = [HEADER.rstrip()]
    for row in rows:
        safe = [cell.replace("|", "/") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    existing_text = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else HEADER
    rows = parse_existing_rows(existing_text)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    result_path = Path(args.result_dir).as_posix()

    notes = args.notes
    if not notes:
        if args.status == "completed":
            notes = "Backtest completed."
        elif args.status == "missing_backtest_package":
            notes = "Strategy contract exists but generated backtest package is missing."
        elif args.status == "failed":
            notes = "Backtest failed. See run.log."
        else:
            notes = ""

    new_row = [
        args.strategy_id,
        args.contract_path,
        args.status,
        now,
        result_path,
        args.workflow_run,
        notes,
    ]

    replaced = False
    for idx, row in enumerate(rows):
        if row[0] == args.strategy_id:
            rows[idx] = new_row
            replaced = True
            break

    if not replaced:
        rows.append(new_row)

    rows.sort(key=lambda row: row[0])
    ledger_path.write_text(render_rows(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
