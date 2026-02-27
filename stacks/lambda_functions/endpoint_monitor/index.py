"""SageMaker Endpoint Monitor Lambda.

This Lambda is triggered by:
  1. A CloudWatch alarm when the SageMaker endpoint has zero invocations
     for 1 hour.
  2. A daily EventBridge scheduled rule (2 AM UTC) as a safety net.

When triggered, it checks if the SageMaker endpoint exists and is running,
then deletes it to prevent unnecessary costs.

Estimated cost of a running ml.g4dn.xlarge endpoint: ~$0.736/hour (~$530/month).
"""
import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client("sagemaker")

# The endpoint name is passed via environment variable from CDK
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "sam3-seg-endpoint")


def lambda_handler(event, _context):
    """Check if the SageMaker endpoint is running and delete it if so."""
    logger.info("Endpoint monitor triggered. Event: %s", event)
    logger.info("Checking endpoint: %s", ENDPOINT_NAME)

    try:
        response = sagemaker_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        creation_time = response["CreationTime"]
        logger.info("Endpoint '%s' status: %s, created: %s", ENDPOINT_NAME, status, creation_time)
    except ClientError as error:
        if error.response["Error"]["Code"] == "ValidationException":
            logger.info("Endpoint '%s' does not exist. Nothing to do.", ENDPOINT_NAME)
            return {"action": "none", "reason": "endpoint_not_found"}
        raise

    # Check if endpoint has been running for at least 1 hour
    from datetime import datetime, timezone, timedelta
    age = datetime.now(timezone.utc) - creation_time
    if age < timedelta(hours=1):
        logger.info(
            "Endpoint '%s' is only %d minutes old, skipping deletion.",
            ENDPOINT_NAME, int(age.total_seconds() / 60)
        )
        return {"action": "skipped", "reason": "too_new", "age_minutes": int(age.total_seconds() / 60)}

    # Only delete if the endpoint is InService (stable state)
    if status == "InService":
        for attempt in range(1, 4):
            try:
                logger.info("Deleting endpoint '%s' (attempt %d/3).", ENDPOINT_NAME, attempt)
                sagemaker_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
                logger.info("Delete request accepted on attempt %d.", attempt)
                break
            except ClientError as e:
                logger.warning("Delete attempt %d failed: %s", attempt, e)
                if attempt < 3:
                    import time; time.sleep(10)
        else:
            logger.error("All 3 delete attempts failed for endpoint '%s'.", ENDPOINT_NAME)
            return {"action": "failed", "endpoint": ENDPOINT_NAME}
        return {"action": "deleted", "endpoint": ENDPOINT_NAME}
