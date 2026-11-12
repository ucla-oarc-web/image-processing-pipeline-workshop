"""Workshop configuration - loaded from cdk.json. Edit values there."""
import json
import os

import boto3
from botocore.exceptions import ClientError

with open(os.path.join(os.path.dirname(__file__), "..", "cdk.json")) as _f:
    _CTX = json.load(_f)["context"]

REGION = _CTX["region"]
ENDPOINT_NAME = _CTX["endpoint_name"]
STACK_NAME = _CTX["stack_name"]
STORAGE_STACK_NAME = "OarcWsStorageStack"
RESOURCE_PREFIX = _CTX["resource_prefix"]
TAGS = [{"Key": k, "Value": v} for k, v in _CTX.get("tags", {}).items()]

INSTANCE_TYPE = _CTX["instance_type"]
PRICE_PER_HOUR = _CTX["instance_price_per_hour"]  # on-demand price in REGION


def get_bucket_name() -> str:
    """Bucket name from the storage stack output (falls back to cdk.json)."""
    cfn = boto3.client("cloudformation", region_name=REGION)
    try:
        outputs = cfn.describe_stacks(StackName=STORAGE_STACK_NAME)["Stacks"][0].get("Outputs", [])
        for output in outputs:
            if output["OutputKey"] == "BucketName":
                return output["OutputValue"]
    except ClientError:
        pass
    if _CTX.get("bucket_name"):
        return _CTX["bucket_name"]
    raise SystemExit(f"ERROR: {STORAGE_STACK_NAME} not found in {REGION}. Run 'cdk deploy --all' first.")
