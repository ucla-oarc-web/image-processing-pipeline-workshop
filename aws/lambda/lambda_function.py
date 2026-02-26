import json
import boto3
from botocore.exceptions import ClientError
import logging
import base64
from PIL import Image
from io import BytesIO
import requests
import os

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
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2).encode(),
                  ContentType="application/json")
    return f"s3://{bucket}/{key}"


def save_text_to_s3(text, bucket, key):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode(), ContentType="text/plain")
    return f"s3://{bucket}/{key}"


# --- Bedrock ---

def call_bedrock(compare_bytes, before_uri, after_uri, prompt):
    resized_bytes = resize_image(compare_bytes)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                 "data": base64.b64encode(resized_bytes).decode()}},
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

    return {
        "before": sam3["before"],
        "after": sam3["after"],
        "compared": sam3["compare"],
        "llm_output": llm_uri,
        "base_name": base_name
    }


def generate_markdown_report(data, base_name):
    """Generate markdown report with styled HTML for images.

    Args:
        data: Dictionary with before, after, compared image URIs and llm_output
        base_name: Base name for image files

    Returns:
        str: Markdown report with inline HTML for image styling
    """
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


def save_markdown_to_s3(markdown_content, bucket, base_name):
    markdown_key = f"markdown/{base_name}.md"
    s3.put_object(Bucket=bucket, Key=markdown_key, Body=markdown_content.encode(), 
                  ContentType="text/markdown")
    logger.info(f"Saved markdown report to s3://{bucket}/{markdown_key}")
    return markdown_key


# --- GitHub ---

def commit_to_github(repo, branch, token, file_path, content, message):
    """Commit a file to GitHub repository using the Contents API.

    Args:
        repo: Repository in format 'owner/repo'
        branch: Target branch name
        token: GitHub personal access token
        file_path: Path in the repository for the file
        content: File content (string or bytes)
        message: Commit message

    Returns:
        dict: GitHub API response with commit details
    """

    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    encoded = base64.b64encode(
        content if isinstance(content, bytes) else content.encode()
    ).decode()

    # Get existing file SHA if it exists (required for updates)
    sha = None
    try:
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
        elif resp.status_code != 404:
            resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Could not check existing file: {e}")

    payload = {"message": message, "content": encoded, "branch": branch}
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload)
    put_resp.raise_for_status()

    logger.info(f"Successfully committed {file_path} to {repo}/{branch}")
    return put_resp.json()


def sync_to_github(data, markdown_content, base_name):
    """Sync images and markdown report to GitHub repository.

    Args:
        data: Dictionary with before, after, compared image URIs
        markdown_content: The markdown string to commit
        base_name: Base name for the files

    Returns:
        dict: Status of GitHub sync operation
    """
    github_repo = os.environ.get("GITHUB_REPO")
    github_token = os.environ.get("GITHUB_TOKEN")
    github_branch = os.environ.get("GITHUB_BRANCH")

    if not github_repo or not github_token or not github_branch:
        logger.info("GitHub credentials not configured, skipping commit")
        return {"github_committed": False}

    try:
        # Load images from S3
        before_img = load_s3_binary(data["before"])
        after_img = load_s3_binary(data["after"])
        compared_img = load_s3_binary(data["compared"])

        # Files to commit: (path, content, message)
        files = [
            (f"markdown/before_image_{base_name}.png", before_img, "Add before image"),
            (f"markdown/after_image_{base_name}.png", after_img, "Add after image"),
            (f"markdown/compared_image_{base_name}.png", compared_img, "Add compared image"),
            (f"markdown/{base_name}.md", markdown_content.encode(), "Add analysis report"),
        ]

        for file_path, content, message in files:
            commit_to_github(
                repo=github_repo,
                branch=github_branch,
                token=github_token,
                file_path=file_path,
                content=content,
                message=f"{message}: {base_name}"
            )

        logger.info(f"Committed {len(files)} files to GitHub: {github_repo}")
        return {"github_committed": True, "github_files": len(files)}

    except Exception as e:
        logger.error(f"Failed to commit to GitHub: {e}")
        return {"github_committed": False, "github_error": str(e)}


# --- Entry Point ---

def lambda_handler(event, _ctx):
    bucket = event["bucket"]
    key = event["key"]

    if key.startswith("async-out/") and key.endswith(".out"):
        # Process SAM3 output and call Bedrock
        response = process_sam3_analysis(bucket, key)

        # Generate markdown report (with styled HTML for images)
        markdown_content = generate_markdown_report(response, response["base_name"])

        # Save markdown to S3
        markdown_key = save_markdown_to_s3(markdown_content, bucket, response["base_name"])

        # Sync images + markdown to GitHub
        github_result = sync_to_github(response, markdown_content, response["base_name"])

        return {
            "bucket": bucket,
            "sourceKey": key,
            "before": response["before"],
            "after": response["after"],
            "compared": response["compared"],
            "llm_output": response["llm_output"],
            "markdown_key": markdown_key,
            **github_result
        }

    raise ValueError(f"Unsupported key pattern: {key}")
