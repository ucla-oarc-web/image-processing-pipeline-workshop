"""Upload test data and watch for pipeline reports.

Uploads input files, then watches for and downloads the generated markdown reports.
For adjuster results, use adjuster_report.py instead.

Usage:
    python scripts/run_pipeline.py

Press Ctrl+C to stop watching.
"""
import os
import sys
import time

import boto3

from config import REGION, get_bucket_name

MARKDOWN_PREFIX = "markdown/"
FAILURE_PREFIX = "async-failure/"
POLL_INTERVAL_SECONDS = 30


def upload_test_data(bucket: str, region: str):
    """Upload images and test input files to S3."""
    s3_client = boto3.client("s3", region_name=region)

    # Upload images first
    images_dir = os.path.join(os.path.dirname(__file__), "..", "files", "images")
    if os.path.exists(images_dir):
        image_files = sorted(os.listdir(images_dir))
        print(f"Uploading {len(image_files)} image(s) to s3://{bucket}/images/")
        for filename in image_files:
            s3_client.upload_file(os.path.join(images_dir, filename), bucket, f"images/{filename}")
            print(f"  {filename}")
        print()

    # Find input files
    inputs_dir = os.path.join(os.path.dirname(__file__), "..", "files", "inputs")

    if not os.path.exists(inputs_dir):
        print(f"ERROR: Input directory not found: {inputs_dir}")
        return 0

    input_files = [f for f in os.listdir(inputs_dir) if f.endswith(".json")]

    if not input_files:
        print(f"ERROR: No JSON files found in {inputs_dir}")
        return 0

    print(f"Uploading {len(input_files)} input file(s) to s3://{bucket}/inputs/\n")

    for i, filename in enumerate(sorted(input_files), 1):
        local_path = os.path.join(inputs_dir, filename)
        s3_key = f"inputs/{filename}"

        print(f"[{i}/{len(input_files)}] Uploading {filename}...")
        s3_client.upload_file(local_path, bucket, s3_key)

        # Add delay between uploads to trigger separate executions
        if i < len(input_files):
            time.sleep(5)

    print(f"\nUploaded {len(input_files)} file(s). Pipeline executions starting...\n")
    return len(input_files)


def list_keys(s3_client, bucket: str, prefix: str = MARKDOWN_PREFIX) -> set:
    """List all object keys under a prefix."""
    keys = set()
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.add(obj["Key"])
    return keys


def watch_for_results(bucket: str, region: str, expected_count: int):
    """Watch for new markdown reports in S3."""
    s3_client = boto3.client("s3", region_name=region)

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(script_dir, "files", "reports", "pipeline")
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Watching for {expected_count} report(s) in s3://{bucket}/{MARKDOWN_PREFIX}")
    print(f"Saving to: {reports_dir}")
    print(f"Poll interval: {POLL_INTERVAL_SECONDS}s. Press Ctrl+C to stop.\n")

    known = list_keys(s3_client, bucket)
    known_failures = list_keys(s3_client, bucket, FAILURE_PREFIX)
    print(f"Found {len(known)} existing report(s).\n")

    # Download any existing reports not already saved locally
    existing_files = set(os.listdir(reports_dir))
    for key in known:
        filename = os.path.basename(key)
        if filename not in existing_files:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode()
            with open(os.path.join(reports_dir, filename), "w") as f:
                f.write(content)

    received = 0
    failed = 0

    try:
        while received + failed < expected_count:
            time.sleep(POLL_INTERVAL_SECONDS)

            # SageMaker writes errors to async-failure/ instead of async-out/
            failures = list_keys(s3_client, bucket, FAILURE_PREFIX)
            for key in sorted(failures - known_failures):
                failed += 1
                error = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode(errors="replace")
                print(f"\n{'='*60}")
                print(f"  SAGEMAKER FAILURE: {key}")
                print(f"{'='*60}\n{error[:2000]}\n")
                print("  Check the endpoint's CloudWatch logs: /aws/sagemaker/Endpoints/<endpoint-name>\n")
            known_failures = failures

            current = list_keys(s3_client, bucket)
            new_keys = current - known

            if new_keys:
                for key in sorted(new_keys):
                    received += 1
                    print(f"\n{'='*60}")
                    print(f"  NEW REPORT ({received}/{expected_count}): {key}")
                    print(f"{'='*60}\n")

                    response = s3_client.get_object(Bucket=bucket, Key=key)
                    content = response["Body"].read().decode()

                    filename = os.path.basename(key)
                    local_path = os.path.join(reports_dir, filename)
                    with open(local_path, "w") as f:
                        f.write(content)
                    print(f"Saved to: {local_path}")

                known = current
            else:
                print(f"  Waiting... ({received}/{expected_count} reports, {failed} failed)")

        print(f"\n{'='*60}")
        print(f"  Done: {received} report(s) received, {failed} failed.")
        print(f"{'='*60}")
        print(f"\nRun 'python scripts/adjuster_report.py' for adjuster results.\n")

    except KeyboardInterrupt:
        print(f"\n\nStopped. {received}/{expected_count} reports received.")
        print(f"Run 'python scripts/adjuster_report.py' for adjuster results.\n")


def main():
    bucket = get_bucket_name()
    count = upload_test_data(bucket, REGION)
    if count == 0:
        sys.exit(1)
    watch_for_results(bucket, REGION, count)


if __name__ == "__main__":
    main()
