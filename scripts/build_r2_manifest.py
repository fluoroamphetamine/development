#!/usr/bin/env python3
"""Build or refresh a simple Cloudflare R2 object manifest.

This script lists objects in the configured R2 bucket and writes a YAML manifest
that generated backtests can use to resolve instrument/timeframe data keys.

It is intentionally conservative: exact dataset records are only created for
Parquet files whose names look like OHLCV time-series files.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import yaml
from botocore.config import Config


OHLCV_PATTERN = re.compile(
    r"^(?P<instrument>[A-Za-z0-9_]+)/(?P<filename>.+?\.ohlcv-(?P<timeframe>[^./]+)\.parquet)$"
)


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/r2_manifest.yaml")
    parser.add_argument("--max-keys", type=int, default=10000)
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def make_client():
    return boto3.client(
        "s3",
        endpoint_url=required_env("R2_ENDPOINT"),
        aws_access_key_id=required_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=required_env("R2_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


def list_keys(bucket: str, prefix: str, max_keys: int) -> list[str]:
    client = make_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= max_keys:
                return keys

    return keys


def dataset_id(instrument: str, timeframe: str) -> str:
    return f"{instrument.upper()}_{timeframe}"


def build_manifest(keys: list[str], bucket: str) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    known_datasets: dict[str, dict[str, Any]] = {}

    for key in sorted(keys):
        top_level = key.split("/", 1)[0] if "/" in key else ""
        if top_level:
            grouped[top_level].append(key)

        match = OHLCV_PATTERN.match(key)
        if not match:
            continue

        instrument = match.group("instrument").upper()
        timeframe = match.group("timeframe")
        ds_id = dataset_id(instrument, timeframe)

        known_datasets[ds_id] = {
            "instrument": instrument,
            "timeframe": timeframe,
            "type": "ohlcv",
            "format": "parquet",
            "r2_key": key,
            "status": "confirmed_by_r2_manifest_refresh",
        }

    prefixes = {
        prefix: {
            "r2_prefix": f"{prefix}/",
            "object_count": len(items),
            "parquet_count": sum(1 for item in items if item.endswith(".parquet")),
        }
        for prefix, items in sorted(grouped.items())
    }

    return {
        "bucket": bucket,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "endpoint_env": "R2_ENDPOINT",
        "bucket_env": "R2_BUCKET",
        "access_key_env": "R2_ACCESS_KEY_ID",
        "secret_key_env": "R2_SECRET_ACCESS_KEY",
        "known_datasets": known_datasets,
        "prefixes": prefixes,
        "all_parquet_keys": [key for key in sorted(keys) if key.endswith(".parquet")],
        "notes": [
            "Generated backtests must not hard-code R2 credentials.",
            "Generated backtests should use backtests.lib.r2_data for runtime loading.",
            "For exact object reads, prefer known_datasets.<dataset_id>.r2_key over prefixes.",
        ],
    }


def main() -> None:
    args = parse_args()
    bucket = required_env("R2_BUCKET")
    keys = list_keys(bucket=bucket, prefix=args.prefix, max_keys=args.max_keys)
    manifest = build_manifest(keys=keys, bucket=bucket)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    print(f"Wrote R2 manifest to {output_path}")
    print(f"Objects listed: {len(keys)}")
    print(f"Parquet keys listed: {len(manifest['all_parquet_keys'])}")


if __name__ == "__main__":
    main()
