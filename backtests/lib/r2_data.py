#!/usr/bin/env python3
"""Reusable Cloudflare R2 market-data helpers for generated backtests.

Generated backtests should import this module instead of hard-coding R2 access.

Expected environment variables, normally provided by GitHub Actions secrets:

- R2_ENDPOINT
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY
- R2_BUCKET

The helpers intentionally download Parquet objects into a local cache before
reading with Polars. This keeps generated backtests simple and avoids depending
on less predictable direct remote scan behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import boto3
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError


DEFAULT_CACHE_DIR = Path(".cache/r2")
DEFAULT_REGION = "auto"


@dataclass(frozen=True)
class R2Config:
    """Cloudflare R2 connection configuration."""

    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region: str = DEFAULT_REGION


@dataclass(frozen=True)
class R2ObjectRef:
    """Parsed R2 object reference."""

    bucket: str
    key: str


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""

    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_r2_config_from_env(
    *,
    endpoint_env: str = "R2_ENDPOINT",
    access_key_env: str = "R2_ACCESS_KEY_ID",
    secret_key_env: str = "R2_SECRET_ACCESS_KEY",
    bucket_env: str = "R2_BUCKET",
) -> R2Config:
    """Load R2 connection settings from environment variables."""

    return R2Config(
        endpoint=require_env(endpoint_env),
        access_key_id=require_env(access_key_env),
        secret_access_key=require_env(secret_key_env),
        bucket=require_env(bucket_env),
        region=os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION),
    )


def make_r2_client(config: R2Config | None = None):
    """Create a boto3 S3 client configured for Cloudflare R2."""

    cfg = config or load_r2_config_from_env()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name=cfg.region,
        config=Config(signature_version="s3v4"),
    )


def parse_r2_uri(uri: str, default_bucket: str | None = None) -> R2ObjectRef:
    """Parse an R2 URI or key into bucket and key.

    Supports:
    - r2://bucket/path/to/file.parquet
    - s3://bucket/path/to/file.parquet
    - path/to/file.parquet with default_bucket
    """

    raw = uri.strip()
    for prefix in ("r2://", "s3://"):
        if raw.startswith(prefix):
            rest = raw[len(prefix) :]
            bucket, sep, key = rest.partition("/")
            if not bucket or not sep or not key:
                raise ValueError(f"Invalid R2 URI: {uri}")
            return R2ObjectRef(bucket=bucket, key=key)

    if not default_bucket:
        raise ValueError(
            f"URI does not include a bucket and no default bucket was provided: {uri}"
        )

    return R2ObjectRef(bucket=default_bucket, key=raw.lstrip("/"))


def safe_cache_path(cache_dir: Path, bucket: str, key: str) -> Path:
    """Convert an R2 bucket/key into a safe local cache path."""

    safe_key = key.strip("/").replace("/", "__")
    return cache_dir / bucket / safe_key


def object_exists(bucket: str, key: str, config: R2Config | None = None) -> bool:
    """Return true when an R2 object exists."""

    client = make_r2_client(config)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status in {403, 404}:
            return False
        raise


def list_r2_objects(
    prefix: str = "",
    *,
    bucket: str | None = None,
    config: R2Config | None = None,
    max_keys: int | None = None,
) -> list[str]:
    """List object keys under a prefix."""

    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    client = make_r2_client(cfg)
    paginator = client.get_paginator("list_objects_v2")

    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if max_keys is not None and len(keys) >= max_keys:
                return keys

    return keys


def download_r2_object(
    bucket: str,
    key: str,
    local_path: Path,
    *,
    config: R2Config | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a single R2 object into a local file path."""

    if local_path.exists() and not overwrite:
        return local_path

    client = make_r2_client(config)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.download_file(bucket, key, str(local_path))
    except ClientError as exc:
        raise RuntimeError(f"Failed to download r2://{bucket}/{key}: {exc}") from exc

    return local_path


def download_r2_prefix(
    prefix: str,
    *,
    bucket: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    suffixes: Iterable[str] = (".parquet",),
    config: R2Config | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download all objects under a prefix that match the requested suffixes."""

    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    suffix_tuple = tuple(suffixes)
    keys = list_r2_objects(prefix, bucket=bucket_name, config=cfg)

    downloaded: list[Path] = []
    for key in keys:
        if suffix_tuple and not key.endswith(suffix_tuple):
            continue
        local_path = safe_cache_path(cache_dir, bucket_name, key)
        downloaded.append(
            download_r2_object(
                bucket_name,
                key,
                local_path,
                config=cfg,
                overwrite=overwrite,
            )
        )

    if not downloaded:
        raise FileNotFoundError(
            f"No matching R2 objects found for prefix r2://{bucket_name}/{prefix}"
        )

    return downloaded


def load_parquet_object(
    uri_or_key: str,
    *,
    bucket: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    config: R2Config | None = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download one Parquet object from R2 and load it with Polars."""

    cfg = config or load_r2_config_from_env()
    ref = parse_r2_uri(uri_or_key, default_bucket=bucket or cfg.bucket)
    cache_path = safe_cache_path(Path(cache_dir), ref.bucket, ref.key)
    local_path = download_r2_object(
        ref.bucket,
        ref.key,
        cache_path,
        config=cfg,
        overwrite=overwrite,
    )
    return pl.read_parquet(local_path)


def scan_parquet_object(
    uri_or_key: str,
    *,
    bucket: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    config: R2Config | None = None,
    overwrite: bool = False,
) -> pl.LazyFrame:
    """Download one Parquet object from R2 and return a Polars LazyFrame."""

    cfg = config or load_r2_config_from_env()
    ref = parse_r2_uri(uri_or_key, default_bucket=bucket or cfg.bucket)
    cache_path = safe_cache_path(Path(cache_dir), ref.bucket, ref.key)
    local_path = download_r2_object(
        ref.bucket,
        ref.key,
        cache_path,
        config=cfg,
        overwrite=overwrite,
    )
    return pl.scan_parquet(local_path)


def load_parquet_prefix(
    prefix: str,
    *,
    bucket: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    config: R2Config | None = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Download all Parquet objects under a prefix and concatenate them."""

    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    paths = download_r2_prefix(
        prefix,
        bucket=bucket_name,
        cache_dir=Path(cache_dir),
        suffixes=(".parquet",),
        config=cfg,
        overwrite=overwrite,
    )
    return pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")


def load_market_data_from_config(config: dict[str, Any]) -> pl.DataFrame:
    """Load market data using a generated backtest config dictionary.

    Expected config shape:

    data:
      source: r2
      market_data:
        r2_key: ES/ES-20100606-20260315.ohlcv-1m.parquet
      local_cache_dir: .cache/r2

    Optional:
      data.market_data.r2_uri: r2://futures-data/ES/file.parquet
      data.market_data.r2_prefix: ES/
    """

    data_cfg = config.get("data", {})
    if data_cfg.get("source", "r2") != "r2":
        raise ValueError("load_market_data_from_config only supports data.source=r2")

    cfg = load_r2_config_from_env(
        endpoint_env=data_cfg.get("endpoint_env", "R2_ENDPOINT"),
        access_key_env=data_cfg.get("access_key_env", "R2_ACCESS_KEY_ID"),
        secret_key_env=data_cfg.get("secret_key_env", "R2_SECRET_ACCESS_KEY"),
        bucket_env=data_cfg.get("bucket_env", "R2_BUCKET"),
    )

    market_cfg = data_cfg.get("market_data", {})
    cache_dir = Path(data_cfg.get("local_cache_dir", DEFAULT_CACHE_DIR.as_posix()))

    if "r2_uri" in market_cfg:
        df = load_parquet_object(
            market_cfg["r2_uri"],
            cache_dir=cache_dir,
            config=cfg,
            overwrite=bool(data_cfg.get("overwrite_cache", False)),
        )
    elif "r2_key" in market_cfg:
        df = load_parquet_object(
            market_cfg["r2_key"],
            bucket=market_cfg.get("bucket", cfg.bucket),
            cache_dir=cache_dir,
            config=cfg,
            overwrite=bool(data_cfg.get("overwrite_cache", False)),
        )
    elif "r2_prefix" in market_cfg:
        df = load_parquet_prefix(
            market_cfg["r2_prefix"],
            bucket=market_cfg.get("bucket", cfg.bucket),
            cache_dir=cache_dir,
            config=cfg,
            overwrite=bool(data_cfg.get("overwrite_cache", False)),
        )
    else:
        raise ValueError(
            "data.market_data must include one of: r2_uri, r2_key, or r2_prefix"
        )

    validate_required_columns(df, config)
    return df


def validate_required_columns(df: pl.DataFrame, config: dict[str, Any]) -> None:
    """Validate required OHLCV columns from a generated backtest config."""

    columns_cfg = config.get("columns", {})
    required_cols = [
        columns_cfg.get("timestamp", "timestamp"),
        columns_cfg.get("open", "open"),
        columns_cfg.get("high", "high"),
        columns_cfg.get("low", "low"),
        columns_cfg.get("close", "close"),
        columns_cfg.get("volume", "volume"),
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise RuntimeError(
            "Market data is missing required columns: "
            f"{missing}. Available columns: {df.columns}"
        )


def normalize_ohlcv_columns(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Rename configured OHLCV columns to canonical names used by backtests."""

    columns_cfg = config.get("columns", {})
    mapping = {
        columns_cfg.get("timestamp", "timestamp"): "timestamp",
        columns_cfg.get("open", "open"): "open",
        columns_cfg.get("high", "high"): "high",
        columns_cfg.get("low", "low"): "low",
        columns_cfg.get("close", "close"): "close",
        columns_cfg.get("volume", "volume"): "volume",
    }

    rename_map = {
        source: target
        for source, target in mapping.items()
        if source in df.columns and source != target
    }
    return df.rename(rename_map) if rename_map else df
