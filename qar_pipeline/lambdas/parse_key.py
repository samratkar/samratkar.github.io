import json
import os


def _parse_airline_from_bucket(bucket_name: str) -> str:
    # Expected: qar-{airline}-data
    if not bucket_name.startswith("qar-") or not bucket_name.endswith("-data"):
        raise ValueError(f"Unexpected bucket name: {bucket_name}")
    return bucket_name[len("qar-") : -len("-data")]


def _parse_key(key: str) -> dict:
    # Expected: incoming/{tail}/{batch_id}/file.csv
    parts = key.split("/")
    if len(parts) < 4 or parts[0] != "incoming":
        raise ValueError(f"Unexpected key format: {key}")
    tail = parts[1]
    batch_id = parts[2]
    return {"tail": tail, "batch_id": batch_id}


def handler(event, _context):
    bucket = event["bucket"]
    key = event["key"]

    airline = _parse_airline_from_bucket(bucket)
    parsed = _parse_key(key)
    s3_uri = f"s3://{bucket}/{key}"

    return {
        "airline": airline,
        "tail": parsed["tail"],
        "batch_id": parsed["batch_id"],
        "s3_uri": s3_uri,
    }


if __name__ == "__main__":
    event = json.loads(os.environ.get("EVENT", "{}"))
    print(json.dumps(handler(event, None), indent=2))
