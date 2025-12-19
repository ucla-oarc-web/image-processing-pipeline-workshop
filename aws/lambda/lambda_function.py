import os
import json
import boto3
from botocore.exceptions import ClientError
import logging
import base64
import datetime
from PIL import Image
from io import BytesIO

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_ID = "us.anthropic.claude-opus-4-5-20251101-v1:0"


def load_json_from_s3(bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except ClientError as e:
        raise RuntimeError(f"Error fetching s3://{bucket}/{key}: {e}") from e


def load_s3_binary(uri):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def resize_image(image_bytes, max_size_mb=5):
    max_bytes = max_size_mb * 1024 * 1024
    if len(image_bytes) <= max_bytes:
        return image_bytes

    img = Image.open(BytesIO(image_bytes))
    quality = 85
    while quality > 10:
        output = BytesIO()
        img.save(output, format='PNG', optimize=True)
        if len(output.getvalue()) <= max_bytes:
            return output.getvalue()
        quality -= 10
        output = BytesIO()
        img.save(output, format='JPEG', quality=quality)
        if len(output.getvalue()) <= max_bytes:
            return output.getvalue()

    # Final resize if still too large
    width, height = img.size
    img = img.resize((width//2, height//2), Image.Resampling.LANCZOS)
    output = BytesIO()
    img.save(output, format='JPEG', quality=70)
    return output.getvalue()


def load_s3_image_b64(uri, resize):
    image_bytes = load_s3_binary(uri)

    if resize:
        image_bytes = resize_image(image_bytes)

    return base64.b64encode(image_bytes).decode("utf-8")


def save_json_to_s3(data, bucket, key):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2).encode(), ContentType="application/json")
    return f"s3://{bucket}/{key}"


def save_text_to_s3(text, bucket, key):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode(), ContentType="text/plain")
    return f"s3://{bucket}/{key}"


# --- Bedrock ---

def call_bedrock(compare_bytes, before_uri, after_uri, prompt):
    # Resize image to ensure it's under 5MB when base64 encoded
    resized_bytes = resize_image(compare_bytes)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(resized_bytes).decode()}},
            ],
        }],
    }

    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(payload))
    body = json.loads(response["body"].read())

    for item in body.get("content", []):
        if item.get("type") == "text":
            return item["text"]
    return json.dumps(body)


# --- Handlers ---

def process_sam3_analysis(bucket, key):
    """Process SAM3 async output: call Bedrock and save results"""
    sam3 = load_json_from_s3(bucket, key)
    prompt = s3.get_object(Bucket=bucket, Key="prompt.txt")["Body"].read().decode()

    compare_bytes = load_s3_binary(sam3["compare"])
    result = call_bedrock(compare_bytes, sam3["before"], sam3["after"], prompt)

    base_name = os.path.splitext(os.path.basename(sam3["llm_output"]))[0]
    llm_uri = save_text_to_s3(result, bucket, f"llm_output/{base_name}.txt")

    completed = {
        "before": sam3["before"],
        "after": sam3["after"],
        "compared": sam3["compare"],
        "llm_output": llm_uri,
        "base_name": base_name
    }
    # save_json_to_s3(completed, bucket, f"outputs/{base_name}-completed.json")
    return completed

def handle_stage_3(data, bucket, key):
    """Generate markdown report with embedded images and save to S3"""
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Load images and LLM output
    before_img = load_s3_binary(data["before"])
    after_img = load_s3_binary(data["after"])
    compared_img = load_s3_binary(data["compared"])
    llm_text = load_s3_binary(data["llm_output"]).decode()

    # Convert images to base64
    before_b64 = base64.b64encode(before_img).decode()
    after_b64 = base64.b64encode(after_img).decode()
    compared_b64 = base64.b64encode(compared_img).decode()

    # Generate report with embedded images and equal height layout
    report = f"""# Palisades Fire Analysis Report

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
    flex: 0 0 60%;
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
            <img src="data:image/png;base64,{before_b64}" alt="Before Image">
        </div>
        <div class="image-box">
            <div class="image-title">After Image</div>
            <img src="data:image/png;base64,{after_b64}" alt="After Image">
        </div>
    </div>
    <div class="right-panel">
        <div class="image-box">
            <div class="image-title">Compared Image</div>
            <img src="data:image/png;base64,{compared_b64}" alt="Compared Image">
        </div>
    </div>
</div>

## Analysis Results

{llm_text}

---
_End of Report_"""

    # Save markdown to S3
    base_name = data["base_name"]
    markdown_key = f"markdown/{base_name}.md"
    save_text_to_s3(report, bucket, markdown_key)

    return {
        "bucket": bucket,
        "sourceKey": key,
        "markdownKey": markdown_key,
    }

# --- GitHub ---

def commit_to_github(repo, branch, token, file_path, content, message):
    import requests

    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}"}

    encoded = base64.b64encode(content if isinstance(content, bytes) else content.encode()).decode()

    resp = requests.get(api_url, headers=headers)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    payload = {"message": message, "content": encoded, "branch": branch}
    if sha:
        payload["sha"] = sha

    requests.put(api_url, headers=headers, json=payload).raise_for_status()


# --- Entry Point ---

def lambda_handler(event, _ctx):
    bucket = event["bucket"]
    key = event["key"]

    if key.startswith("async-out/") and key.endswith(".out"):
        response = process_sam3_analysis(bucket, key)
        return handle_stage_3(response, bucket, key)
