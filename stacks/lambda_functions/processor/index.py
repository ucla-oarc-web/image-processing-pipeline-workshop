"""Image Processing Pipeline - Processor Lambda.

This Lambda is invoked by Step Functions after SageMaker async inference
completes. It:
  1. Reads the SageMaker output JSON from S3 (contains image URIs)
  2. Sends the before, after and outlined images, area metadata and optional CAL FIRE
     data to Claude on Bedrock, which can call the tools in tools.py
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
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO

import boto3
from anthropic import AnthropicBedrock
from botocore.exceptions import ClientError
from PIL import Image

from tools import TOOLS

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")
claude = AnthropicBedrock()  # region and credentials come from the Lambda environment

BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
MAX_TOKENS = 16000  # per response, including Claude's thinking
MAX_TURNS = 8  # cap on Claude <-> tool round trips, so one image can't run up the bill
# Image tokens grow with pixel area (about 1 token per 28x28 pixels): 1280 px is about 1,600 tokens
# per image instead of about 3,200 at full size. Raise it if Claude needs finer detail.
IMAGE_MAX_EDGE = 1280

# Claude Opus 5.5 list prices in USD per million tokens, for the cost estimate in the logs.
# Check Amazon Bedrock pricing for your region; update these if they differ.
PRICE_PER_MTOK = {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00}


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
# CAL FIRE damage inspection (DINS) data
# ---------------------------------------------------------------------------

# Public CAL FIRE DINS feature service (statewide, one record per inspected structure)
DINS_URL = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/"
            "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query")
# Construction details only. DAMAGE is left out on purpose: SAM3 decides what burned.
DINS_FIELDS = ["INCIDENTNAME", "STRUCTURETYPE", "YEARBUILT", "ROOFCONSTRUCTION", "EAVES", "VENTSCREEN",
               "EXTERIORSIDING", "WINDOWPANE", "DECKPORCHELEVATED", "FENCEATTACHEDTOSTRUCTURE"]


def get_calfire_context(bucket, geo_bbox):
    """Summarize how homes inside geo_bbox [min_lon, min_lat, max_lon, max_lat] were built.

    Records are cached in S3 so repeated runs don't hit the CAL FIRE API again.
    Returns "" when no geo_bbox is given.
    """
    if not geo_bbox:
        return ""
    cache_key = "calfire-cache/" + "_".join(f"{v:.5f}" for v in geo_bbox) + ".json"
    try:
        records = json.loads(s3_client.get_object(Bucket=bucket, Key=cache_key)["Body"].read())
        logger.info("Using cached CAL FIRE data s3://%s/%s", bucket, cache_key)
    except s3_client.exceptions.NoSuchKey:
        params = {
            "where": "1=1",  # every inspected structure in the area, whichever fire it was inspected after
            "geometry": ",".join(str(v) for v in geo_bbox),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(DINS_FIELDS),
            "returnGeometry": "false",
            "f": "json",
        }
        with urllib.request.urlopen(f"{DINS_URL}?{urllib.parse.urlencode(params)}", timeout=30) as resp:
            records = [f["attributes"] for f in json.load(resp).get("features", [])]
        s3_client.put_object(Bucket=bucket, Key=cache_key, Body=json.dumps(records).encode(),
                             ContentType="application/json")
        logger.info("Fetched %d CAL FIRE records, cached at s3://%s/%s", len(records), bucket, cache_key)

    if not records:
        return ""
    lines = [f"CAL FIRE inspection data: how the {len(records)} structures in this area were built "
             "(counts per value; burn outcomes are not included):"]
    for field in DINS_FIELDS:
        counts = Counter(str(r.get(field) or "Unknown") for r in records)
        if field == "YEARBUILT":
            counts = Counter("Unknown" if y == "Unknown" else f"{y[:3]}0s" for y in counts.elements())
        lines.append(f"- {field}: " + ", ".join(f"{v} ({n})" for v, n in counts.most_common(6)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bedrock
# ---------------------------------------------------------------------------

def call_bedrock(content):
    """Send the labeled inputs to Claude on Bedrock, let it call the tools, and return its text answer.

    content: the user message's content blocks (images, metadata, then the prompt).
    The SDK's tool runner runs the loop: when Claude asks for a tool, the runner calls
    the matching function in tools.py and sends the result back, until Claude is done.
    """
    # Stream so long thinking doesn't hit HTTP timeouts
    runner = claude.beta.messages.tool_runner(
        model=BEDROCK_MODEL_ID,
        max_tokens=MAX_TOKENS,
        max_iterations=MAX_TURNS,
        output_config={"effort": "medium"},
        tools=TOOLS,
        messages=[{"role": "user", "content": content}],
        # Every tool round trip resends the whole conversation. Automatic caching moves the cache
        # forward each turn, so the images, prompt and earlier tool results are read from cache
        # at a small fraction of the normal input price.
        cache_control={"type": "ephemeral"},
        stream=True,
    )

    # Each turn is one request; add up the tokens so we can log what the image cost
    tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}
    for turn, stream in enumerate(runner, start=1):
        message = stream.get_final_message()
        usage = message.usage
        tokens["input"] += usage.input_tokens
        tokens["cache_write"] += usage.cache_creation_input_tokens or 0
        tokens["cache_read"] += usage.cache_read_input_tokens or 0
        tokens["output"] += usage.output_tokens
    cost = sum(tokens[k] * PRICE_PER_MTOK[k] / 1_000_000 for k in tokens)
    logger.info("Claude finished in %d turn(s), stop_reason=%s, tokens=%s, estimated cost $%.2f",
                turn, message.stop_reason, tokens, cost)

    # Check why Claude stopped before using its answer
    if message.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {message.stop_details}")
    if message.stop_reason == "tool_use":
        raise RuntimeError(f"Claude was still calling tools after {MAX_TURNS} turns; raise MAX_TURNS")
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={message.stop_reason})")
    if message.stop_reason == "max_tokens":
        text += "\n\n_(Analysis truncated: the model reached its output limit.)_"
    return text


def build_area_metadata(sam3_output):
    """Area metadata for Claude: capture dates, resolution, bounds, and SAM3's burned/survived labels."""
    original_input = sam3_output.get("input", {})
    detection = sam3_output.get("detection", {})
    houses = detection.get("detections", [])
    not_given = "not provided"
    return {
        "before_image_capture_date": original_input.get("before_date", not_given),
        "after_image_capture_date": original_input.get("after_date", not_given),
        "image_size_px": [detection.get("image_width"), detection.get("image_height")],
        "resolution_m_per_px": detection.get("gsd_m") or not_given,
        "location_bounds": original_input.get("geo_bbox") or not_given,
        "location_bounds_format": "[min_longitude, min_latitude, max_longitude, max_latitude]",
        "labels_from": "SAM3 segmentation (an earlier pipeline step)",
        "homes_burned": sum(h["label"] == "burned" for h in houses),
        "homes_survived": sum(h["label"] == "survived" for h in houses),
        "homes": houses,  # id, bbox [x_min, y_min, x_max, y_max] in pixels, label
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def process_sam3_analysis(bucket, key):
    """Read SageMaker output, call Bedrock, and save LLM analysis to S3."""
    sam3_output = load_json_from_s3(bucket, key)
    with open(os.path.join(os.path.dirname(__file__), "prompt.txt")) as f:
        prompt = f.read()

    original_input = sam3_output.get("input", {})
    metadata = build_area_metadata(sam3_output)
    calfire_context = get_calfire_context(bucket, original_input.get("geo_bbox"))

    # Inputs first, each labeled the way the prompt names it; the prompt goes last
    content = []
    for label, uri in [("Before Image", sam3_output["before"]),
                       ("After Image", sam3_output["after"]),
                       ("Outlined Image", sam3_output["compare"])]:
        content.append({"type": "text", "text": f"<{label}>"})
        content.append({"type": "image", "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(to_jpeg(load_s3_binary(uri), max_edge=IMAGE_MAX_EDGE)).decode(),
        }})
    content.append({"type": "text", "text": f"<Area Metadata>\n{json.dumps(metadata, separators=(",", ":"))}\n</Area Metadata>"})
    if calfire_context:
        content.append({"type": "text", "text": f"<CAL FIRE Construction Data>\n{calfire_context}\n</CAL FIRE Construction Data>"})
    content.append({"type": "text", "text": prompt})

    llm_result = call_bedrock(content)

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
        "llm_text": llm_result,
        "base_name": base_name,
    }


def generate_markdown_report(llm_text, base_name):
    """Generate a styled markdown/HTML report with embedded image references."""
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
    markdown_content = generate_markdown_report(analysis_result["llm_text"], analysis_result["base_name"])

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
