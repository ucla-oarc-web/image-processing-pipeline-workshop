"""SageMaker Endpoint Deployment Script.

Manages the SageMaker endpoint lifecycle independently from the CDK stack.
The endpoint is the most expensive resource (see PRICE_PER_HOUR in config.py)
so it is created and deleted on demand rather than as part of the stack.

Usage:
    python scripts/deploy_endpoint.py create   # Create endpoint
    python scripts/deploy_endpoint.py delete   # Delete endpoint
    python scripts/deploy_endpoint.py status   # Check endpoint status
"""
import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError

from config import (REGION, ENDPOINT_NAME, STACK_NAME, TAGS, RESOURCE_PREFIX, INSTANCE_TYPE,
                    PRICE_PER_HOUR)


def get_stack_outputs(stack_name: str, region: str) -> dict:
    """Read CDK stack outputs to get bucket name and other config."""
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        response = cfn.describe_stacks(StackName=stack_name)
        outputs = response["Stacks"][0].get("Outputs", [])
        return {o["OutputKey"]: o["OutputValue"] for o in outputs}
    except ClientError:
        return {}


def create_endpoint(endpoint_name: str):
    """Create SageMaker Endpoint (model and config already exist in CDK stack)."""
    sm = boto3.client("sagemaker", region_name=REGION)

    print(f"\n{'='*60}")
    print(f"  WARNING: This will create a SageMaker endpoint")
    print(f"  Instance type: {INSTANCE_TYPE}")
    print(f"  Estimated cost: ~${PRICE_PER_HOUR}/hour (~${PRICE_PER_HOUR * 24:.2f}/day)")
    print(f"  Auto-shutdown: enabled after 1 hour of inactivity")
    print(f"{'='*60}\n")

    try:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status == "InService":
            print(f"Endpoint '{endpoint_name}' is already InService.")
            return
        print(f"Endpoint '{endpoint_name}' exists with status: {status}")
        if status == "Failed":
            print("Deleting failed endpoint and recreating...")
            sm.delete_endpoint(EndpointName=endpoint_name)
            _wait_for_deletion(sm, endpoint_name)
        else:
            print("Waiting for current operation to complete...")
            _wait_for_in_service(sm, endpoint_name)
            return
    except ClientError:
        pass

    print(f"Creating endpoint '{endpoint_name}'...")
    outputs = get_stack_outputs(STACK_NAME, REGION)
    config_name = outputs.get("EndpointConfigName")
    if not config_name:
        print("ERROR: EndpointConfigName not found in stack outputs. Run 'cdk deploy' first.")
        sys.exit(1)
    print(f"  Config: {config_name}")
    try:
        cfg = sm.describe_endpoint_config(EndpointConfigName=config_name)
        print(f"  Model:  {cfg['ProductionVariants'][0]['ModelName']}")
    except Exception:
        pass
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
        Tags=TAGS,
    )
    _wait_for_in_service(sm, endpoint_name)


def _wait_for_in_service(sm, endpoint_name: str):
    """Poll endpoint status until InService or failure."""
    print("Waiting for endpoint to be ready (this takes 5-10 minutes)...")
    while True:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"  Status: {status}")
        if status == "InService":
            print("\nEndpoint is ready!")
            return
        if status == "Failed":
            reason = response.get("FailureReason", "unknown")
            print(f"\nEndpoint creation failed: {reason}")
            sys.exit(1)
        time.sleep(30)


def _wait_for_deletion(sm, endpoint_name: str):
    """Poll until endpoint no longer exists."""
    print("Waiting for endpoint deletion...")
    while True:
        try:
            sm.describe_endpoint(EndpointName=endpoint_name)
            time.sleep(15)
        except ClientError:
            print("Endpoint deleted.")
            return


def delete_endpoint():
    """Delete SageMaker Endpoint (model and config managed by CDK stack)."""
    sm = boto3.client("sagemaker", region_name=REGION)

    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Deleting endpoint '{ENDPOINT_NAME}'...")
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        _wait_for_deletion(sm, ENDPOINT_NAME)
    except ClientError:
        print(f"Endpoint '{ENDPOINT_NAME}' does not exist.")

    print("\nEndpoint deleted. Model and config remain (managed by CDK stack).")


def check_status():
    """Check the current status of the SageMaker endpoint."""
    sm = boto3.client("sagemaker", region_name=REGION)
    cw = boto3.client("cloudwatch", region_name=REGION)

    try:
        response = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = response["EndpointStatus"]
        creation_time = response["CreationTime"]

        config_name = response.get("EndpointConfigName", "unknown")
        model_name = "unknown"
        try:
            cfg = sm.describe_endpoint_config(EndpointConfigName=config_name)
            model_name = cfg["ProductionVariants"][0]["ModelName"]
        except Exception:
            pass

        print(f"Endpoint: {ENDPOINT_NAME}")
        print(f"Config:   {config_name}")
        print(f"Model:    {model_name}")
        print(f"Status:   {status}")
        print(f"Created:  {creation_time}")

        if status == "InService":
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            hours = (now - creation_time).total_seconds() / 3600
            print(f"\nRuntime:  {hours:.2f} hours")
            print(f"Cost:     ${hours * PRICE_PER_HOUR:.2f}")

            try:
                metric_response = cw.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="InvocationsProcessed",
                    Dimensions=[
                        {"Name": "EndpointName", "Value": ENDPOINT_NAME},
                        {"Name": "VariantName", "Value": "primary"}
                    ],
                    StartTime=creation_time,
                    EndTime=now,
                    Period=60,
                    Statistics=["Sum"]
                )
                datapoints = sorted(metric_response.get("Datapoints", []), key=lambda x: x["Timestamp"])
                last_invocation = next((dp["Timestamp"] for dp in reversed(datapoints) if dp["Sum"] > 0), None)
                total_invocations = int(sum(dp["Sum"] for dp in datapoints))
                if last_invocation:
                    idle_mins = (now - last_invocation).total_seconds() / 60
                    shutdown_mins = max(0, 60 - idle_mins)
                    print(f"Idle:     {idle_mins:.0f} min (last invocation: {last_invocation.strftime('%Y-%m-%d %H:%M:%S %Z')})")
                    print(f"Shutdown: ~{shutdown_mins:.0f} min remaining")
                else:
                    print(f"Idle:     {hours:.2f} hours (no invocations)")
                    print(f"Shutdown: imminent (no activity detected)")
                print(f"Invocations: {total_invocations} (since creation)")
            except Exception as e:
                print(f"Idle:     Unable to determine ({e})")

            # Alarm status
            try:
                alarm_name = f"{RESOURCE_PREFIX}-sagemaker-idle-endpoint"
                alarm_resp = cw.describe_alarms(AlarmNames=[alarm_name])
                alarms = alarm_resp.get("MetricAlarms", [])
                if alarms:
                    a = alarms[0]
                    print(f"\nAuto-shutdown alarm: {a['StateValue']}")
                    print(f"  Alarm:   {alarm_name}")
                    print(f"  Metric:  {a['MetricName']} (threshold < {int(a['Threshold'])} over {a['Period']//60}min)")
                    if a.get("StateUpdatedTimestamp"):
                        print(f"  Updated: {a['StateUpdatedTimestamp']}")
                else:
                    print(f"\nAuto-shutdown alarm: NOT FOUND ({alarm_name})")
                    print("  Deploy the pipeline stack to create it.")
            except Exception as e:
                print(f"\nAuto-shutdown alarm: Unable to check ({e})")

            print(f"\nWARNING: Endpoint is running and incurring costs (~${PRICE_PER_HOUR}/hr).")
            print("Run 'python scripts/deploy_endpoint.py delete' to stop it.")
    except ClientError:
        print(f"Endpoint '{ENDPOINT_NAME}' does not exist (not running, no cost).")


def main():
    parser = argparse.ArgumentParser(description="Manage the SageMaker endpoint.")
    parser.add_argument("action", choices=["create", "delete", "status"])
    action = parser.parse_args().action

    outputs = get_stack_outputs(STACK_NAME, REGION)
    endpoint = outputs.get("EndpointName", ENDPOINT_NAME)

    if action == "create":
        create_endpoint(endpoint)
    elif action == "delete":
        delete_endpoint()
    else:
        check_status()


if __name__ == "__main__":
    main()
