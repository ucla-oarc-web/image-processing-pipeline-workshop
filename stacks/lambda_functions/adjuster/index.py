import base64
import io
import json
import logging
import os
import re
from urllib.parse import unquote_plus
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
from botocore.config import Config

bedrock = boto3.client("bedrock-runtime", config=Config(read_timeout=600))
dynamodb = boto3.resource("dynamodb")


MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
ROUTING_TABLE_NAME = os.environ["ROUTING_TABLE_NAME"]

table = dynamodb.Table(ROUTING_TABLE_NAME)


def _load_prompt() -> str:
    """Load prompt from local prompt.txt file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning("prompt.txt not found, using built-in prompt")
        return _base_prompt()


def _extract_bucket_key(event: Dict[str, Any]) -> Tuple[str, str]:
    # Direct invocation
    if "bucket" in event and "key" in event:
        return event["bucket"], event["key"]

    # EventBridge S3 event
    detail = event.get("detail", {})
    if "bucket" in detail and "object" in detail:
        return detail["bucket"]["name"], unquote_plus(detail["object"]["key"])

    # S3 notification
    records = event.get("Records", [])
    if records:
        first = records[0]
        if first.get("eventSource") == "aws:s3":
            bucket = first["s3"]["bucket"]["name"]
            key = unquote_plus(first["s3"]["object"]["key"])
            return bucket, key

    raise ValueError("Event must contain either {bucket,key}, EventBridge detail, or S3 Records")


def _load_s3_text(bucket: str, key: str) -> str:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except ClientError as error:
        raise RuntimeError(f"Unable to read s3://{bucket}/{key}: {error}") from error


def _load_s3_binary(bucket: str, key: str) -> bytes:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        raise RuntimeError(f"Unable to read s3://{bucket}/{key}: {error}") from error


def _guess_media_type(key: str) -> str:
    lowered = key.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".webp"):
        return "image/webp"
    return "image/png"


def _base_prompt() -> str:
    return (
        "You are an insurance property adjuster reviewing post-fire satellite imagery. "
        "For each visible home, determine whether there is a 5-foot inclusion zone around "
        "the home where no fire encroachment is present. Return STRICT JSON only using this schema: "
        "{\"summary\": {\"total_homes\": int, \"auto_approved_count\": int, \"needs_human_review_count\": int}, "
        "\"homes\": [{\"house_id\": string, \"decision\": \"auto_approved\"|\"needs_human_review\", "
        "\"has_5ft_inclusion_zone\": true|false|null, \"confidence\": number, \"reason\": string, "
        "\"bbox\": {\"x_min\": number, \"y_min\": number, \"x_max\": number, \"y_max\": number}}]}. "
        "Use bbox coordinates normalized from 0.0 to 1.0 and aligned to the referenced home. "
        "Use needs_human_review whenever confidence is low, occlusion exists, or 5-foot clearance cannot be confirmed. "
        "Keep each reason concise (<= 12 words). Return JSON only, no markdown, no commentary."
    )


def _resize_image(image_bytes: bytes, max_size_bytes: int = 3_900_000) -> bytes:
    """Resize an image to fit within the Bedrock API size limit (5MB after base64 encoding)."""
    if len(image_bytes) <= max_size_bytes:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))

    # Try JPEG at decreasing quality
    for quality in (85, 70, 50, 30):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        if len(buf.getvalue()) <= max_size_bytes:
            return buf.getvalue()

    # Scale down until it fits
    while True:
        width, height = img.size
        img = img.resize((int(width * 0.8), int(height * 0.8)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        if len(buf.getvalue()) <= max_size_bytes:
            return buf.getvalue()


def _invoke_bedrock(image_bytes: bytes, image_key: str, prompt_text: str) -> Dict[str, Any]:
    # Resize if needed (same logic as processor Lambda)
    image_bytes = _resize_image(image_bytes)
    media_type = "image/jpeg" if image_bytes[0:2] == b'\xff\xd8' else "image/png"

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        },
                    },
                ],
            }
        ],
    }

    logger.info("Calling Bedrock (max_tokens=32000, image_size=%d bytes)", len(image_bytes))
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(payload))
    body = json.loads(response["body"].read())
    stop_reason = str(body.get("stop_reason", "unknown"))
    logger.info("Bedrock responded (stop_reason=%s)", stop_reason)

    text_chunks = [
        item.get("text", "")
        for item in body.get("content", [])
        if item.get("type") == "text"
    ]
    model_text = "\n".join(text_chunks).strip()
    cleaned = model_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Non-JSON Bedrock response (stop_reason=%s): %s", stop_reason, cleaned)
        raise RuntimeError("Bedrock response is not valid JSON")


def _normalize_decisions(raw: Dict[str, Any]) -> Dict[str, Any]:
    homes: List[Dict[str, Any]] = raw.get("homes", []) if isinstance(raw, dict) else []
    normalized: List[Dict[str, Any]] = []

    for idx, home in enumerate(homes, start=1):
        decision = str(home.get("decision", "")).strip().lower()
        if decision not in {"auto_approved", "needs_human_review"}:
            decision = "needs_human_review"

        house_id = str(home.get("house_id", f"house-{idx:03d}"))
        inclusion = home.get("has_5ft_inclusion_zone", None)
        bbox = _normalize_bbox(home.get("bbox"))

        confidence_raw = home.get("confidence", 0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(home.get("reason", "No reason provided by model"))

        normalized.append(
            {
                "house_id": house_id,
                "decision": decision,
                "has_5ft_inclusion_zone": inclusion,
                "confidence": confidence,
                "reason": reason,
                "bbox": bbox,
            }
        )

    auto_count = len([h for h in normalized if h["decision"] == "auto_approved"])
    review_count = len([h for h in normalized if h["decision"] == "needs_human_review"])

    return {
        "summary": {
            "total_homes": len(normalized),
            "auto_approved_count": auto_count,
            "needs_human_review_count": review_count,
        },
        "homes": normalized,
    }


def _normalize_bbox(raw_bbox: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw_bbox, dict):
        return None

    required = ["x_min", "y_min", "x_max", "y_max"]
    parsed: Dict[str, float] = {}

    for key in required:
        value = raw_bbox.get(key)
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            return None

    x_min = max(0.0, min(1.0, parsed["x_min"]))
    y_min = max(0.0, min(1.0, parsed["y_min"]))
    x_max = max(0.0, min(1.0, parsed["x_max"]))
    y_max = max(0.0, min(1.0, parsed["y_max"]))

    if x_max <= x_min or y_max <= y_min:
        return None

    return {
        "x_min": round(x_min, 6),
        "y_min": round(y_min, 6),
        "x_max": round(x_max, 6),
        "y_max": round(y_max, 6),
    }


def _sanitize_key_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return cleaned[:120] or "home"


def _bbox_to_pixel_box(bbox: Dict[str, float], width: int, height: int) -> Tuple[int, int, int, int]:
    left = int(bbox["x_min"] * width)
    top = int(bbox["y_min"] * height)
    right = int(bbox["x_max"] * width)
    bottom = int(bbox["y_max"] * height)

    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _bbox_to_dynamodb_map(bbox: Optional[Dict[str, float]]) -> Optional[Dict[str, str]]:
    if not bbox:
        return None
    return {
        "x_min": str(bbox["x_min"]),
        "y_min": str(bbox["y_min"]),
        "x_max": str(bbox["x_max"]),
        "y_max": str(bbox["y_max"]),
    }


def _save_visual_artifacts(
    bucket: str,
    image_key: str,
    image_bytes: bytes,
    normalized: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, str]]:
    if not PIL_AVAILABLE:
        logger.warning("Pillow is unavailable; skipping crop/annotation artifact generation")
        return None, {}

    try:
        base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as error:
        logger.warning("Unable to open image for artifact generation: %s", error)
        return None, {}

    width, height = base_image.size
    annotated_image = base_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    line_width = max(2, width // 400)

    crop_uris: Dict[str, str] = {}

    for home in normalized.get("homes", []):
        house_id = str(home.get("house_id", "unknown"))
        bbox = home.get("bbox")
        if not bbox:
            continue

        left, top, right, bottom = _bbox_to_pixel_box(bbox, width, height)
        crop = base_image.crop((left, top, right, bottom))

        safe_house_id = _sanitize_key_component(house_id)
        crop_key = f"routing-artifacts/crops/{image_key}/{safe_house_id}.png"

        crop_bytes = io.BytesIO()
        crop.save(crop_bytes, format="PNG")
        s3.put_object(
            Bucket=bucket,
            Key=crop_key,
            Body=crop_bytes.getvalue(),
            ContentType="image/png",
        )
        crop_uris[house_id] = f"s3://{bucket}/{crop_key}"

        decision = str(home.get("decision", "needs_human_review"))
        color = "green" if decision == "auto_approved" else "red"
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
        label = f"{house_id} {decision}"
        label_top = max(0, top - 14)
        draw.text((left + 2, label_top), label, fill=color)

    annotated_key = f"routing-artifacts/annotated/{image_key}.annotated.png"
    annotated_bytes = io.BytesIO()
    annotated_image.save(annotated_bytes, format="PNG")
    s3.put_object(
        Bucket=bucket,
        Key=annotated_key,
        Body=annotated_bytes.getvalue(),
        ContentType="image/png",
    )

    return f"s3://{bucket}/{annotated_key}", crop_uris


def _write_routing_results(
    bucket: str,
    image_key: str,
    normalized: Dict[str, Any],
    annotated_image_s3_uri: Optional[str],
    crop_uris: Dict[str, str],
) -> None:
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    source_uri = f"s3://{bucket}/{image_key}"

    with table.batch_writer() as writer:
        for home in normalized["homes"]:
            writer.put_item(
                Item={
                    "routing_id": f"{image_key}#{home['house_id']}",
                    "source_image_uri": source_uri,
                    "house_id": home["house_id"],
                    "decision": home["decision"],
                    "has_5ft_inclusion_zone": home["has_5ft_inclusion_zone"],
                    "confidence": str(home["confidence"]),
                    "reason": home["reason"],
                    "bbox": _bbox_to_dynamodb_map(home.get("bbox")),
                    "home_crop_s3_uri": crop_uris.get(home["house_id"]),
                    "annotated_image_s3_uri": annotated_image_s3_uri,
                    "created_at": timestamp,
                }
            )

    logger.info(
        "Wrote %s routing records to DynamoDB table %s for %s",
        len(normalized["homes"]),
        ROUTING_TABLE_NAME,
        source_uri,
    )


def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    try:
        bucket, key = _extract_bucket_key(event)
        logger.info(
            "Received event for s3://%s/%s (table=%s, model=%s)",
            bucket,
            key,
            ROUTING_TABLE_NAME,
            MODEL_ID,
        )

        if not key.startswith("compared/"):
            logger.warning("Skipping object outside compared/ prefix: %s", key)
            return {
                "bucket": bucket,
                "key": key,
                "skipped": True,
                "reason": "unsupported-prefix",
            }

        prompt_text = _load_prompt()

        image_bytes = _load_s3_binary(bucket, key)
        raw_model_output = _invoke_bedrock(image_bytes, key, prompt_text)
        normalized = _normalize_decisions(raw_model_output)
        annotated_image_s3_uri, crop_uris = _save_visual_artifacts(bucket, key, image_bytes, normalized)

        _write_routing_results(bucket, key, normalized, annotated_image_s3_uri, crop_uris)

        logger.info("Processed %s: %s", key, normalized["summary"])
        return {
            "bucket": bucket,
            "key": key,
            "table": ROUTING_TABLE_NAME,
            "annotated_image_s3_uri": annotated_image_s3_uri,
            **normalized["summary"],
        }
    except Exception:
        logger.exception("Downstream adjuster processing failed")
        raise
