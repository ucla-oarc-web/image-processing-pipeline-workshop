import torch
import json
import io
import boto3
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor

def _parse_s3_uri(s3_uri, field_name):
    """Helper to validate and parse S3 URI"""
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"{field_name} must be a valid S3 URI")
    try:
        return s3_uri[5:].split('/', 1)
    except ValueError:
        raise ValueError(f"Invalid S3 URI format for {field_name}: {s3_uri}")

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
    """Segment one image and return a bounding box (x_min, y_min, x_max, y_max) per detected object"""
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

    boxes = []
    for mask in results.get("masks", []):
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        if len(x_coords) == 0:
            continue
        boxes.append((int(x_coords.min()), int(y_coords.min()), int(x_coords.max()), int(y_coords.max())))

    return boxes

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
    before_boxes = _process_image(before_image, input_data, model_dict)
    after_boxes = _process_image(after_image, input_data, model_dict)

    # Compare masks and draw outlines
    compared_image = before_image.copy()
    draw = ImageDraw.Draw(compared_image)

    # Find surviving vs destroyed houses
    for before_box in before_boxes:
        house_survived = any(_calculate_overlap(before_box, after_box) > 0.5 for after_box in after_boxes)
        color = "green" if house_survived else "red"
        draw.rectangle(before_box, outline=color, width=3)

    # Save compared image to specified S3 location
    _save_image_to_s3(compared_image, compared_output_uri)

    return {
        "compare": compared_output_uri,
        "before": before_uri,
        "after": after_uri,
    }


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
