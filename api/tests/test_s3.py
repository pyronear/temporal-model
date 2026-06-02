import boto3
import botocore.exceptions as botoexc
import pytest
from moto import mock_aws

from temporal_model.api.errors import FrameNotFound, S3Unavailable
from temporal_model.api.s3 import fetch_frames

BUCKET = "test-frames"
KEYS = [
    "cam12/2023-05-23/adf_2023-05-23T17-18-01.jpg",
    "cam12/2023-05-23/adf_2023-05-23T17-18-31.jpg",
]


@mock_aws
def test_fetch_preserves_order_and_basenames(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    for key in KEYS:
        client.put_object(Bucket=BUCKET, Key=key, Body=b"\xff\xd8\xff\xe0jpegbytes")

    paths = fetch_frames(client, BUCKET, KEYS, tmp_path)

    assert [p.name for p in paths] == [
        "adf_2023-05-23T17-18-01.jpg",
        "adf_2023-05-23T17-18-31.jpg",
    ]
    assert all(p.exists() and p.read_bytes() for p in paths)
    # order is preserved exactly as requested
    assert paths == sorted(paths, key=lambda p: p.parent.name)


@mock_aws
def test_missing_key_raises_frame_not_found(tmp_path):
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket=BUCKET)
    with pytest.raises(FrameNotFound):
        fetch_frames(client, BUCKET, ["cam12/missing.jpg"], tmp_path)


def test_endpoint_failure_raises_s3_unavailable(tmp_path, monkeypatch):
    class _Boom:
        def download_file(self, *a, **k):
            raise botoexc.EndpointConnectionError(endpoint_url="http://nope")

    with pytest.raises(S3Unavailable):
        fetch_frames(_Boom(), BUCKET, ["cam12/a.jpg"], tmp_path)
