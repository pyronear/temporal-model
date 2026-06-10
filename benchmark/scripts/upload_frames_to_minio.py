#!/usr/bin/env python
"""Upload the pyro-annotator frames to the VM's MinIO `frames` bucket.

Keys = each frame's path relative to the store root — the exact keys the
benchmark client POSTs. Idempotent: skips objects that already exist.

Usage: uv run python scripts/upload_frames_to_minio.py \
    --store data/03_primary/sequences --endpoint http://localhost:9000
"""

import argparse
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from temporal_model.benchmark.dataset import iter_sequences
from temporal_model.benchmark.run_api import frame_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    ap.add_argument("--endpoint", default="http://localhost:9000")
    ap.add_argument("--bucket", default="frames")
    ap.add_argument("--access-key", default="minioadmin")
    ap.add_argument("--secret-key", default="minioadmin")
    args = ap.parse_args()

    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        region_name="us-east-1",
    )
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError:
        s3.create_bucket(Bucket=args.bucket)

    uploaded = skipped = 0
    for seq in iter_sequences(args.store):
        for f in seq.frames:
            key = frame_key(args.store, f)
            try:
                s3.head_object(Bucket=args.bucket, Key=key)
                skipped += 1
                continue
            except ClientError:
                pass
            s3.upload_file(str(f.image_path), args.bucket, key)
            uploaded += 1
        if (uploaded + skipped) % 500 == 0:
            print(f"... {uploaded} uploaded, {skipped} skipped")
    print(f"done: {uploaded} uploaded, {skipped} skipped")


if __name__ == "__main__":
    main()
