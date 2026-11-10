"""SageMaker Endpoint Monitor Lambda.

Triggered by:
  1. A CloudWatch alarm when the endpoint has zero invocations for 1 hour.
  2. A daily EventBridge rule (2 AM UTC) as a safety net.

If the SageMaker endpoint exists, is InService, is older than IDLE_MINUTES,
and has processed no invocations in the last IDLE_MINUTES, it deletes the
endpoint to stop costs.

Estimated cost of a running ml.g4dn.xlarge endpoint: ~$0.736/hour (~$530/month).
"""
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client("sagemaker")
cloudwatch_client = boto3.client("cloudwatch")

ENDPOINT_NAME = os.environ["SAGEMAKER_ENDPOINT_NAME"]
IDLE_MINUTES = int(os.environ.get("IDLE_MINUTES", "60"))


def _invocations_since(start: datetime, end: datetime) -> int:
    """Sum of InvocationsProcessed for the endpoint between start and end."""
    response = cloudwatch_client.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName="InvocationsProcessed",
        Dimensions=[
            {"Name": "EndpointName", "Value": ENDPOINT_NAME},
            {"Name": "VariantName", "Value": "primary"},
        ],
        StartTime=start,
        EndTime=end,
        Period=300,
        Statistics=["Sum"],
    )
    return int(sum(dp["Sum"] for dp in response.get("Datapoints", [])))


def lambda_handler(event, _context):
    """Delete the endpoint if it has been idle for IDLE_MINUTES."""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            logger.info("Endpoint '%s' does not exist. Nothing to do.", ENDPOINT_NAME)
            return {"action": "none", "reason": "endpoint_not_found"}
        raise

    status = response["EndpointStatus"]
    now = datetime.now(timezone.utc)
    age_minutes = int((now - response["CreationTime"]).total_seconds() / 60)
    logger.info("Endpoint '%s' status=%s age=%dmin", ENDPOINT_NAME, status, age_minutes)

    if status != "InService":
        return {"action": "none", "reason": f"status_{status}"}
    if age_minutes < IDLE_MINUTES:
        return {"action": "skipped", "reason": "too_new", "age_minutes": age_minutes}

    invocations = _invocations_since(now - timedelta(minutes=IDLE_MINUTES), now)
    if invocations > 0:
        logger.info("Endpoint active: %d invocation(s) in the last %d min.", invocations, IDLE_MINUTES)
        return {"action": "none", "reason": "active", "invocations": invocations}

    for attempt in range(1, 4):
        try:
            logger.info("Idle for %d min. Deleting endpoint '%s' (attempt %d/3).",
                        IDLE_MINUTES, ENDPOINT_NAME, attempt)
            sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
            return {"action": "deleted", "endpoint": ENDPOINT_NAME}
        except ClientError as error:
            logger.warning("Delete attempt %d failed: %s", attempt, error)
            if attempt < 3:
                time.sleep(10)

    logger.error("All 3 delete attempts failed for endpoint '%s'.", ENDPOINT_NAME)
    return {"action": "failed", "endpoint": ENDPOINT_NAME}
