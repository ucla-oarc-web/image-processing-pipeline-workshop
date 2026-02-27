"""View adjuster routing decisions and sync reports from S3.

Usage:
    python scripts/adjuster_report.py              # Print routing table + sync reports
    python scripts/adjuster_report.py --table       # Print routing table only
    python scripts/adjuster_report.py --sync        # Sync reports only
"""
import json
import os
import sys
from collections import defaultdict
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

from config import REGION, BUCKET_NAME

ADJUSTER_STACK = "OarcWsAdjusterStack"
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(SCRIPT_DIR, "files", "reports", "pipeline")
ADJUSTER_DIR = os.path.join(SCRIPT_DIR, "files", "reports", "adjuster")


def get_routing_table_name() -> str:
    """Get DynamoDB routing table name from adjuster stack outputs."""
    cfn = boto3.client("cloudformation", region_name=REGION)
    try:
        resp = cfn.describe_stacks(StackName=ADJUSTER_STACK)
        for out in resp["Stacks"][0].get("Outputs", []):
            if out.get("OutputKey") == "RoutingTableName":
                return out["OutputValue"]
    except ClientError:
        pass
    return None


def scan_all(table_name: str) -> list:
    """Scan entire DynamoDB table with pagination."""
    table = boto3.resource("dynamodb", region_name=REGION).Table(table_name)
    resp = table.scan()
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp.get("Items", []))
    return items


def print_routing_table():
    """Print routing decisions grouped by source image."""
    table_name = get_routing_table_name()
    if not table_name:
        print("ERROR: Adjuster stack not found. Deploy OarcWsAdjusterStack first.")
        sys.exit(1)

    items = scan_all(table_name)
    if not items:
        print("No routing decisions found.")
        return

    # Group by source image
    by_image = defaultdict(list)
    for item in items:
        src = item.get("source_image_uri", "unknown")
        # Just keep the filename
        key = src.split("/")[-1] if "/" in src else src
        by_image[key].append(item)

    # Summary counts
    total = len(items)
    decisions = defaultdict(int)
    for item in items:
        decisions[item.get("decision", "unknown")] += 1

    print(f"\n{'='*70}")
    print(f"  ADJUSTER ROUTING DECISIONS  ({total} total)")
    print(f"{'='*70}")
    print(f"  Summary: {', '.join(f'{v} {k}' for k, v in sorted(decisions.items()))}")
    print()

    for image_key in sorted(by_image):
        houses = sorted(by_image[image_key], key=lambda h: h.get("house_id", ""))
        print(f"  {image_key}  ({len(houses)} homes)")
        print(f"  {'─'*66}")
        print(f"  {'ID':<8} {'Decision':<22} {'Conf':>5}  Reason")
        print(f"  {'─'*66}")
        for h in houses:
            hid = h.get("house_id", "?")
            dec = h.get("decision", "?")
            conf = str(h.get("confidence", "?"))
            reason = h.get("reason", "")
            print(f"  {hid:<8} {dec:<22} {conf:>5}  {reason}")
        print()

    # Save JSON
    os.makedirs(ADJUSTER_DIR, exist_ok=True)
    json_path = os.path.join(ADJUSTER_DIR, "routing_decisions.json")
    with open(json_path, "w") as f:
        json.dump(items, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")


def sync_reports():
    """Download pipeline reports and adjuster artifacts from S3."""
    s3 = boto3.client("s3", region_name=REGION)
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(ADJUSTER_DIR, exist_ok=True)

    # Pipeline markdown reports
    count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="markdown/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if not filename:
                continue
            dest = os.path.join(PIPELINE_DIR, filename)
            if not os.path.exists(dest):
                s3.download_file(BUCKET_NAME, key, dest)
                print(f"  Downloaded: {filename}")
                count += 1
    print(f"  Pipeline reports: {count} new file(s) synced to {PIPELINE_DIR}")

    # Adjuster artifacts (annotated images + crops)
    count = 0
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="routing-artifacts/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len("routing-artifacts/"):]
            if not rel_path:
                continue
            dest = os.path.join(ADJUSTER_DIR, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                s3.download_file(BUCKET_NAME, key, dest)
                count += 1
    print(f"  Adjuster artifacts: {count} new file(s) synced to {ADJUSTER_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    show_table = "--table" in args or not args
    show_sync = "--sync" in args or not args

    if show_table:
        print_routing_table()
    if show_sync:
        print(f"\n{'='*70}")
        print("  SYNCING REPORTS FROM S3")
        print(f"{'='*70}")
        sync_reports()
    print()
