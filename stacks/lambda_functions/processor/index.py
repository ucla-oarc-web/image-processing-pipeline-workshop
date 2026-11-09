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
import base64
import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO

import boto3
from anthropic import AnthropicBedrock
from botocore.exceptions import ClientError
from PIL import Image

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
claude = AnthropicBedrock()  # region and credentials come from the Lambda environment

BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
MAX_TOKENS = 16000  # includes Claude's thinking, so leave plenty of room


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

def to_jpeg(image_bytes, max_edge, quality=85):
    """Downscale so the longest edge is at most max_edge pixels and encode as JPEG."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

def call_bedrock(compare_bytes, prompt):
    """Send the comparison image to Claude on Bedrock and return its text answer."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image", "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(to_jpeg(compare_bytes, max_edge=2000)).decode(),
        }},
    ]

    # Stream so long thinking doesn't hit HTTP timeouts
    with claude.messages.stream(
        model=BEDROCK_MODEL_ID,
        max_tokens=MAX_TOKENS,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()
    logger.info("Claude responded (stop_reason=%s, usage=%s)", message.stop_reason, message.usage)

    # Check why Claude stopped before using its answer
    if message.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {message.stop_details}")
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={message.stop_reason})")
    if message.stop_reason == "max_tokens":
        text += "\n\n_(Analysis truncated: the model reached its output limit.)_"
    return text


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def process_sam3_analysis(bucket, key):
    """Read SageMaker output, call Bedrock, and save LLM analysis to S3."""
    sam3_output = load_json_from_s3(bucket, key)
    with open(os.path.join(os.path.dirname(__file__), "prompt.txt")) as f:
        prompt = f.read()

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
    target_widths = {"before": 300, "after": 300, "compared": 600}

    # Replace image src placeholders with small embedded JPEG data URIs
    for img_type, s3_uri in image_uris.items():
        jpeg = to_jpeg(load_s3_binary(s3_uri), max_edge=target_widths[img_type], quality=75)
        data_uri = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
        markdown_content = markdown_content.replace(f"{img_type}_image_{base_name}.png", data_uri)

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
