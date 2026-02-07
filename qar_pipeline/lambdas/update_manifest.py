import json
import os

import boto3
from botocore.exceptions import ClientError


s3 = boto3.client("s3")


def _manifest_key(batch_id: str) -> str:
    return f"preprocess/{batch_id}/manifest.json"


def _load_manifest(bucket: str, batch_id: str) -> dict:
    key = _manifest_key(batch_id)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return {"batch_id": batch_id, "tails": []}
        raise


def _save_manifest(bucket: str, batch_id: str, manifest: dict) -> None:
    key = _manifest_key(batch_id)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def handler(event, _context):
    # event: { "airline": "...", "tail": "...", "batch_id": "...", "s3_uri": "s3://..." }
    airline = event["airline"]
    tail = event["tail"]
    batch_id = event["batch_id"]
    s3_uri = event["s3_uri"]

    bucket = f"qar-{airline}-data"

    manifest = _load_manifest(bucket, batch_id)
    existing = {t["tail"] for t in manifest.get("tails", [])}

    if tail not in existing:
        manifest.setdefault("tails", []).append({"tail": tail, "input_s3": s3_uri})

    _save_manifest(bucket, batch_id, manifest)

    return {"ok": True, "bucket": bucket, "manifest_key": _manifest_key(batch_id)}


if __name__ == "__main__":
    event = json.loads(os.environ.get("EVENT", "{}"))
    print(json.dumps(handler(event, None), indent=2))
