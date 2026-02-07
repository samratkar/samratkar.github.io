import json
import os

import boto3


s3 = boto3.client("s3")


def _manifest_key(batch_id: str) -> str:
    return f"preprocess/{batch_id}/manifest.json"


def handler(event, _context):
    # event: { "airline": "...", "batch_id": "..." }
    airline = event["airline"]
    batch_id = event["batch_id"]

    bucket = f"qar-{airline}-data"
    key = _manifest_key(batch_id)

    resp = s3.get_object(Bucket=bucket, Key=key)
    manifest = json.loads(resp["Body"].read().decode("utf-8"))

    return {
        "airline": airline,
        "batch_id": batch_id,
        "tails": manifest.get("tails", []),
    }


if __name__ == "__main__":
    event = json.loads(os.environ.get("EVENT", "{}"))
    print(json.dumps(handler(event, None), indent=2))
