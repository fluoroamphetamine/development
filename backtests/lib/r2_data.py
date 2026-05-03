#!/usr/bin/env python3
"""Cloudflare R2 data helpers for generated backtests."""

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

COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "datetime", "datetime_utc", "DateTime", "Datetime", "time"],
    "open": ["open", "Open", "OPEN", "o"],
    "high": ["high", "High", "HIGH", "h"],
    "low": ["low", "Low", "LOW", "l"],
    "close": ["close", "Close", "CLOSE", "c", "last", "Last"],
    "volume": ["volume", "Volume", "VOLUME", "vol", "Vol"],
}


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region: str = DEFAULT_REGION


@dataclass(frozen=True)
class R2ObjectRef:
    bucket: str
    key: str


def require_env(name: str) -> str:
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
    return R2Config(
        endpoint=require_env(endpoint_env),
        access_key_id=require_env(access_key_env),
        secret_access_key=require_env(secret_key_env),
        bucket=require_env(bucket_env),
        region=os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION),
    )


def make_r2_client(config: R2Config | None = None):
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
    raw = uri.strip()
    for prefix in ("r2://", "s3://"):
        if raw.startswith(prefix):
            rest = raw[len(prefix):]
            bucket, sep, key = rest.partition("/")
            if not bucket or not sep or not key:
                raise ValueError(f"Invalid R2 URI: {uri}")
            return R2ObjectRef(bucket=bucket, key=key)
    if not default_bucket:
        raise ValueError(f"No bucket found in URI and no default bucket provided: {uri}")
    return R2ObjectRef(bucket=default_bucket, key=raw.lstrip("/"))


def safe_cache_path(cache_dir: Path, bucket: str, key: str) -> Path:
    return cache_dir / bucket / key.strip("/").replace("/", "__")


def object_exists(bucket: str, key: str, config: R2Config | None = None) -> bool:
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
    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    client = make_r2_client(cfg)
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
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
    if local_path.exists() and not overwrite:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        make_r2_client(config).download_file(bucket, key, str(local_path))
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
    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    suffix_tuple = tuple(suffixes)
    paths: list[Path] = []
    for key in list_r2_objects(prefix, bucket=bucket_name, config=cfg):
        if suffix_tuple and not key.endswith(suffix_tuple):
            continue
        paths.append(
            download_r2_object(
                bucket_name,
                key,
                safe_cache_path(cache_dir, bucket_name, key),
                config=cfg,
                overwrite=overwrite,
            )
        )
    if not paths:
        raise FileNotFoundError(f"No matching R2 objects found for r2://{bucket_name}/{prefix}")
    return paths


def load_parquet_object(
    uri_or_key: str,
    *,
    bucket: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    config: R2Config | None = None,
    overwrite: bool = False,
) -> pl.DataFrame:
    cfg = config or load_r2_config_from_env()
    ref = parse_r2_uri(uri_or_key, default_bucket=bucket or cfg.bucket)
    local_path = download_r2_object(
        ref.bucket,
        ref.key,
        safe_cache_path(Path(cache_dir), ref.bucket, ref.key),
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
    cfg = config or load_r2_config_from_env()
    ref = parse_r2_uri(uri_or_key, default_bucket=bucket or cfg.bucket)
    local_path = download_r2_object(
        ref.bucket,
        ref.key,
        safe_cache_path(Path(cache_dir), ref.bucket, ref.key),
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
    cfg = config or load_r2_config_from_env()
    bucket_name = bucket or cfg.bucket
    paths = download_r2_prefix(
        prefix,
        bucket=bucket_name,
        cache_dir=Path(cache_dir),
        config=cfg,
        overwrite=overwrite,
    )
    return pl.concat([pl.read_parquet(path) for path in paths], how="vertical_relaxed")


def _find_column(df: pl.DataFrame, configured: str | None, canonical: str) -> str | None:
    cols = list(df.columns)
    casefold = {col.casefold(): col for col in cols}
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(COLUMN_ALIASES[canonical])
    for candidate in candidates:
        if candidate in cols:
            return candidate
        folded = casefold.get(candidate.casefold())
        if folded:
            return folded
    return None


def resolve_ohlcv_column_mapping(df: pl.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    columns_cfg = config.get("columns", {})
    mapping: dict[str, str] = {}
    for canonical in ["timestamp", "open", "high", "low", "close", "volume"]:
        source = _find_column(df, columns_cfg.get(canonical), canonical)
        if source:
            mapping[source] = canonical
    return mapping


def validate_required_columns(df: pl.DataFrame, config: dict[str, Any]) -> None:
    found = set(resolve_ohlcv_column_mapping(df, config).values())
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(
            "Market data is missing required OHLCV columns after alias detection: "
            f"{missing}. Available columns: {df.columns}"
        )


def normalize_ohlcv_columns(df: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    mapping = resolve_ohlcv_column_mapping(df, config)
    rename_map = {source: target for source, target in mapping.items() if source != target}
    return df.rename(rename_map) if rename_map else df


def load_market_data_from_config(config: dict[str, Any]) -> pl.DataFrame:
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
    overwrite = bool(data_cfg.get("overwrite_cache", False))

    if "r2_uri" in market_cfg:
        df = load_parquet_object(market_cfg["r2_uri"], cache_dir=cache_dir, config=cfg, overwrite=overwrite)
    elif "r2_key" in market_cfg:
        df = load_parquet_object(
            market_cfg["r2_key"],
            bucket=market_cfg.get("bucket", cfg.bucket),
            cache_dir=cache_dir,
            config=cfg,
            overwrite=overwrite,
        )
    elif "r2_prefix" in market_cfg:
        df = load_parquet_prefix(
            market_cfg["r2_prefix"],
            bucket=market_cfg.get("bucket", cfg.bucket),
            cache_dir=cache_dir,
            config=cfg,
            overwrite=overwrite,
        )
    else:
        raise ValueError("data.market_data must include r2_uri, r2_key, or r2_prefix")

    validate_required_columns(df, config)
    return df
