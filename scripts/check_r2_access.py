#!/usr/bin/env python3

from __future__ import annotations

import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    access_key = required_env("R2_ACCESS_KEY_ID")
    secret_key = required_env("R2_SECRET_ACCESS_KEY")
    bucket = required_env("R2_BUCKET")
    endpoint = required_env("R2_ENDPOINT")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    try:
        response = client.list_objects_v2(Bucket=bucket, MaxKeys=5)
    except ClientError as exc:
        print(f"R2 access check failed: {exc}", file=sys.stderr)
        return 1

    objects = response.get("Contents", [])

    print("R2 access check passed.")
    print(f"Bucket: {bucket}")
    print(f"Endpoint: {endpoint}")
    print(f"Visible objects: {len(objects)}")

    for obj in objects:
        print(f"- {obj.get('Key')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
