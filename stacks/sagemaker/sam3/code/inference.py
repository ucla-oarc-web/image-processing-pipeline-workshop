import numpy as np
import torch
import json
import io
import boto3
from PIL import Image, ImageFilter
from transformers import Sam3Model, Sam3Processor

def _parse_s3_uri(s3_uri, field_name):
    """Validate an s3://bucket/key URI and return (bucket, key)"""
    if not s3_uri.startswith('s3://') or '/' not in s3_uri[5:]:
        raise ValueError(f"{field_name} must be a valid S3 URI (s3://bucket/key): {s3_uri}")
    return s3_uri[5:].split('/', 1)

def model_fn(model_dir):
    """Load the SAM3 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Sam3Model.from_pretrained(model_dir).to(device)
    processor = Sam3Processor.from_pretrained(model_dir)
    model.eval()

    return {"model": model, "processor": processor, "device": device}

def input_fn(request_body, request_content_type):
    """Parse and validate input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)

        # Validate required fields
        required_fields = ['before_image', 'after_image', 'compared_output', 'text']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Validate and load images
        s3 = boto3.client('s3')
        for image_field in ['before_image', 'after_image']:
            s3_uri = input_data[image_field]
            bucket, key = _parse_s3_uri(s3_uri, image_field)

            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                image = Image.open(io.BytesIO(response['Body'].read())).convert('RGB')
                input_data[f"{image_field}_data"] = image
            except Exception as e:
                raise ValueError(f"Failed to load {image_field} from {s3_uri}: {str(e)}")

        # Validate output S3 URI
        _parse_s3_uri(input_data['compared_output'], 'compared_output')

        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def _process_image(image, input_data, model_dict):
    """Segment one image and return (boxes, masks): a bounding box (x_min, y_min, x_max, y_max)
    and a boolean pixel mask per detected object"""
    device = model_dict["device"]
    text_prompt = input_data["text"]
    model = model_dict["model"]
    processor = model_dict["processor"]

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    boxes, masks = [], []
    for mask in results.get("masks", []):
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        if len(x_coords) == 0:
            continue
        boxes.append((int(x_coords.min()), int(y_coords.min()), int(x_coords.max()), int(y_coords.max())))
        masks.append(mask.bool().cpu().numpy())

    return boxes, masks

def _draw_outline(image, mask, color, width=3):
    """Trace the edge of a mask onto the image in the given color"""
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    edge = mask_image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(width))
    image.paste(color, mask=edge)

def _save_image_to_s3(image, s3_uri):
    """Helper to save image to S3"""
    s3 = boto3.client('s3')
    bucket, key = _parse_s3_uri(s3_uri, "output URI")

    output_buffer = io.BytesIO()
    image.save(output_buffer, format='PNG')

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=output_buffer.getvalue(),
        ContentType='image/png'
    )

def predict_fn(input_data, model_dict):
    """Run SAM3 inference and compare before/after"""

    # Get S3 URIs and pre-loaded images
    before_uri = input_data["before_image"]
    after_uri = input_data["after_image"]
    compared_output_uri = input_data["compared_output"]

    # Use pre-loaded images from input_fn
    before_image = input_data["before_image_data"]
    after_image = input_data["after_image_data"]

    # Process both images
    before_boxes, before_masks = _process_image(before_image, input_data, model_dict)
    after_boxes, _ = _process_image(after_image, input_data, model_dict)

    # The before and after photos are not always framed identically. Shift the
    # after boxes onto the before image so the same house lines up in both.
    dx, dy = _estimate_shift(before_image, after_image)
    after_boxes = [(x1 + dx, y1 + dy, x2 + dx, y2 + dy) for x1, y1, x2, y2 in after_boxes]

    # Compare boxes and outline each house's shape
    compared_image = before_image.copy()

    # Find surviving vs destroyed houses. A house "survived" if a box in the
    # after image overlaps its before box by more than 50% (IoU).
    detections = []
    for before_box, before_mask in zip(before_boxes, before_masks):
        house_survived = any(_calculate_overlap(before_box, after_box) > 0.5 for after_box in after_boxes)
        color = "green" if house_survived else "red"
        _draw_outline(compared_image, before_mask, color)
        detections.append({
            "id": len(detections) + 1,
            "bbox": list(before_box),
            "label": "survived" if house_survived else "burned",
        })

    # Save compared image to specified S3 location
    _save_image_to_s3(compared_image, compared_output_uri)

    # Save the detections next to the compared image
    image_id = _image_id(before_uri)
    width, height = before_image.size
    detection = {
        "image_id": image_id,
        "image_width": width,
        "image_height": height,
        "gsd_m": input_data.get("input", {}).get("gsd_m"),  # meters per pixel, if given in the input JSON
        "detections": detections,
    }
    detection_uri = f"{compared_output_uri.rsplit('/', 1)[0]}/{image_id}-detections.json"
    _save_json_to_s3(detection, detection_uri)

    return {
        "compare": compared_output_uri,
        "before": before_uri,
        "after": after_uri,
        "detection_file": detection_uri,
        "detection": detection,
    }


def _estimate_shift(before_image, after_image, max_shift=200):
    """Pixel offset (dx, dy) that moves the after image onto the before image.

    Uses phase correlation: the peak of the normalized cross-power spectrum
    marks the shift between two images of the same scene. Handles shifts only
    (not rotation or zoom). Returns (0, 0) if the result is implausibly large.
    """
    width = min(before_image.width, after_image.width)
    height = min(before_image.height, after_image.height)
    before = np.asarray(before_image.convert("L").crop((0, 0, width, height)), dtype=np.float32)
    after = np.asarray(after_image.convert("L").crop((0, 0, width, height)), dtype=np.float32)
    before -= before.mean()
    after -= after.mean()

    cross_power = np.fft.fft2(before) * np.conj(np.fft.fft2(after))
    correlation = np.abs(np.fft.ifft2(cross_power / (np.abs(cross_power) + 1e-9)))
    peak_y, peak_x = np.unravel_index(correlation.argmax(), correlation.shape)

    # Peaks past the halfway point are negative shifts (the FFT wraps around)
    dy = int(peak_y if peak_y < height / 2 else peak_y - height)
    dx = int(peak_x if peak_x < width / 2 else peak_x - width)

    if abs(dx) > max_shift or abs(dy) > max_shift:
        print(f"Alignment shift ({dx}, {dy}) too large; comparing without alignment")
        return 0, 0
    print(f"Aligned after image by dx={dx}, dy={dy} pixels")
    return dx, dy


def _image_id(before_uri):
    """Image ID from the before image name, e.g. s3://bucket/images/1-before.png -> 1"""
    name = before_uri.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    return name.replace("-before", "").replace("_before", "")


def _save_json_to_s3(data, s3_uri):
    """Write a dict as JSON to S3"""
    bucket, key = _parse_s3_uri(s3_uri, "output URI")
    boto3.client('s3').put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode(),
        ContentType='application/json'
    )


def _calculate_overlap(box1, box2):
    """Intersection over union (IoU) of two (x_min, y_min, x_max, y_max) boxes"""
    x1, y1, x1_max, y1_max = box1
    x2, y2, x2_max, y2_max = box2

    # Find intersection rectangle
    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1_max, x2_max)
    intersect_y2 = min(y1_max, y2_max)

    # Check if there's any intersection
    if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
        return 0

    # Calculate areas
    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    box1_area = (x1_max - x1) * (y1_max - y1)
    box2_area = (x2_max - x2) * (y2_max - y2)
    union_area = box1_area + box2_area - intersect_area

    return intersect_area / union_area if union_area > 0 else 0

def output_fn(prediction_output, content_type):
    """Format output"""
    return json.dumps(prediction_output)
