"""Image Processing Pipeline - Processor Lambda.

This Lambda is invoked by Step Functions after SageMaker async inference
completes. It:
  1. Reads the SageMaker output JSON from S3 (contains image URIs)
  2. Sends the comparison image to Bedrock Claude for analysis
  3. Generates a styled markdown report with before/after/compared images
  4. Saves the report to S3 (markdown/ prefix)

Environment variables (set by CDK):
  BUCKET_NAME       - S3 bucket for all pipeline data
  BEDROCK_MODEL_ID  - Bedrock model identifier for Claude
"""
import json
import os
import logging
import base64

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from io import BytesIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "oarc-image-processing-pipeline")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-opus-4-5-20251101-v1:0")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def load_json_from_s3(bucket, key):
    """Load and parse a JSON file from S3."""
    try:
        return json.loads(s3_client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as e:
        raise RuntimeError(f"Error fetching s3://{bucket}/{key}: {e}") from e


def load_s3_binary(uri):
    """Load raw bytes from an s3:// URI."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()


def save_text_to_s3(text, bucket, key):
    """Write a text string to S3 and return the s3:// URI."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode(), ContentType="text/plain")
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image(image_bytes, max_size_bytes=3_900_000):
    """Resize an image to fit within the Bedrock API size limit (5MB after base64 encoding)."""
    if len(image_bytes) <= max_size_bytes:
        return image_bytes

    img = Image.open(BytesIO(image_bytes))

    # Try JPEG at decreasing quality
    for quality in (85, 70, 50, 30):
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        if len(buf.getvalue()) <= max_size_bytes:
            return buf.getvalue()

    # Scale down until it fits
    while True:
        width, height = img.size
        img = img.resize((int(width * 0.8), int(height * 0.8)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        if len(buf.getvalue()) <= max_size_bytes:
            return buf.getvalue()


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

def call_bedrock(compare_bytes, prompt):
    """Send the comparison image to Bedrock Claude for analysis."""
    resized_bytes = resize_image(compare_bytes)
    media_type = "image/jpeg" if resized_bytes[0:2] == b'\xff\xd8' else "image/png"

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(resized_bytes).decode(),
                }},
            ],
        }],
    }

    response = bedrock_client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(payload))
    body = json.loads(response["body"].read())

    for item in body.get("content", []):
        if item.get("type") == "text":
            return item["text"]
    return json.dumps(body)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def process_sam3_analysis(bucket, key):
    """Read SageMaker output, call Bedrock, and save LLM analysis to S3."""
    from datetime import datetime, timezone

    sam3_output = load_json_from_s3(bucket, key)
    prompt = open(os.path.join(os.path.dirname(__file__), "prompt.txt")).read()

    compare_bytes = load_s3_binary(sam3_output["compare"])
    llm_result = call_bedrock(compare_bytes, prompt)

    # Extract descriptive name from before image filename (e.g., "1-before.png" -> "1")
    before_image = sam3_output.get("before", "")
    if before_image:
        image_name = os.path.splitext(os.path.basename(before_image))[0]
        # Remove common suffixes like "-before"
        image_name = image_name.replace("-before", "").replace("_before", "")
        descriptive_name = f"palisades-fire-{image_name}"
    else:
        # Fallback to input filename
        descriptive_name = os.path.splitext(os.path.basename(key))[0]

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
    base_name = f"{descriptive_name}--{timestamp}"

    llm_uri = save_text_to_s3(llm_result, bucket, f"llm_output/{base_name}.txt")

    return {
        "before": sam3_output["before"],
        "after": sam3_output["after"],
        "compared": sam3_output["compare"],
        "llm_output": llm_uri,
        "base_name": base_name,
    }


def generate_markdown_report(data, base_name):
    """Generate a styled markdown/HTML report with embedded image references."""
    llm_text = load_s3_binary(data["llm_output"]).decode()

    return f"""# Palisades Fire Analysis Report

## Before and After Images

<style>
.container {{
    display: flex;
    gap: 20px;
    margin: 20px 0;
    height: 600px;
}}
.left-panel {{
    flex: 0 0 40%;
    display: flex;
    flex-direction: column;
    gap: 10px;
}}
.right-panel {{
    flex: 1 0 auto;
}}
.image-box {{
    width: 100%;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}}
.left-panel .image-box {{
    flex: 1;
}}
.right-panel .image-box {{
    flex: 1;
    height: 100%;
    width: 100%;
}}
.image-box img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}}
.image-title {{
    padding: 10px;
    background: #f5f5f5;
    font-weight: bold;
    text-align: center;
}}
</style>

<div class="container">
    <div class="left-panel">
        <div class="image-box">
            <div class="image-title">Before Image</div>
            <img src="before_image_{base_name}.png" alt="Before Image">
        </div>
        <div class="image-box">
            <div class="image-title">After Image</div>
            <img src="after_image_{base_name}.png" alt="After Image">
        </div>
    </div>
    <div class="right-panel">
        <div class="image-box">
            <div class="image-title">Compared Image</div>
            <img src="compared_image_{base_name}.png" alt="Compared Image">
        </div>
    </div>
</div>

## Analysis Results

{llm_text}

---
_End of Report_
"""


def save_markdown_to_s3(markdown_content, bucket, base_name, image_uris):
    """Save the markdown report with embedded base64 images to S3."""
    # Load and resize images to embed in markdown
    embedded_images = {}
    target_sizes = {"before": 300, "after": 300, "compared": 600}

    for img_type, s3_uri in image_uris.items():
        if s3_uri.startswith("s3://"):
            img_bytes = load_s3_binary(s3_uri)
            img = Image.open(BytesIO(img_bytes))

            # Resize to smaller dimensions
            max_width = target_sizes.get(img_type, 400)
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to JPEG with compression for smaller file size
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=75, optimize=True)
            b64_data = base64.b64encode(buffer.getvalue()).decode()
            embedded_images[img_type] = f"data:image/jpeg;base64,{b64_data}"

    # Replace image src with base64 data URIs
    for img_type, data_uri in embedded_images.items():
        markdown_content = markdown_content.replace(
            f'{img_type}_image_{base_name}.png',
            data_uri
        )

    markdown_key = f"markdown/{base_name}.md"
    s3_client.put_object(
        Bucket=bucket, Key=markdown_key,
        Body=markdown_content.encode(), ContentType="text/markdown",
    )
    logger.info("Saved markdown report with embedded images to s3://%s/%s", bucket, markdown_key)
    return markdown_key


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event, _context):
    """Main handler - processes SageMaker async output files."""
    bucket = event["bucket"]
    key = event["key"]

    if not (key.startswith("async-out/") and key.endswith(".out")):
        raise ValueError(f"Unsupported key pattern: {key}")

    # Step 1: Read SageMaker output and call Bedrock for analysis
    analysis_result = process_sam3_analysis(bucket, key)

    # Step 2: Generate styled markdown report
    markdown_content = generate_markdown_report(analysis_result, analysis_result["base_name"])

    # Step 3: Save report and images to S3
    image_uris = {
        "before": analysis_result["before"],
        "after": analysis_result["after"],
        "compared": analysis_result["compared"],
    }
    markdown_key = save_markdown_to_s3(markdown_content, bucket, analysis_result["base_name"], image_uris)

    return {
        "bucket": bucket,
        "sourceKey": key,
        "before": analysis_result["before"],
        "after": analysis_result["after"],
        "compared": analysis_result["compared"],
        "llm_output": analysis_result["llm_output"],
        "markdown_key": markdown_key,
    }
