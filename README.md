# OARC Image Processing Pipeline Workshop

Serverless ML pipeline for analyzing before/after satellite imagery using AWS services. Designed as a workshop for UCLA OARC demonstrating serverless ML workflows.

## What It Does

1. Accepts before/after satellite image pairs via S3 JSON inputs
2. Uses SAM3 (Segment Anything Model 3) on SageMaker for object segmentation
3. Compares images to identify changes (red = destroyed, green = survived)
4. Uses Claude Opus 5.5 via Bedrock to analyze the before, after, and comparison images
5. Produces styled markdown reports saved to S3

## Configuration

All settings are in `cdk.json`. Edit this file before deploying:

```json
{
  "context": {
    "region": "us-west-2",
    "bucket_name": "",
    "endpoint_name": "oarc-ws-sam3-endpoint",
    "bedrock_model_id": "us.anthropic.claude-opus-5-5",
    ...
  }
}
```

Leave `bucket_name` empty and the bucket is named `oarc-ws-pipeline-<account-id>-<region>`.
Bucket names must be unique across all AWS accounts, and this keeps yours unique.

Scripts in `scripts/` read from `cdk.json` and the deployed stack outputs — no other config needed.

## Quick Start

### Prerequisites

- Python 3.11+, Node.js, and Docker (CDK builds the Lambda and SageMaker images locally)
- Access to Claude Opus 5.5 in Amazon Bedrock for your account
- A Hugging Face account that has accepted the [SAM3 license](https://huggingface.co/facebook/sam3)

```bash
# Install AWS CDK CLI (if not already installed). The CLI is versioned
# separately from aws-cdk-lib; any recent version works.
npm install -g aws-cdk

# Bootstrap your AWS account for CDK (one-time setup per account/region)
cdk bootstrap aws://ACCOUNT-ID/us-west-2
```

### Download the SAM3 weights (one time)

The model weights (~3.4GB) are not in this repo. Download them next to the
config files in `stacks/sagemaker/sam3/model/`:

```bash
pip install -U huggingface_hub
hf auth login
hf download facebook/sam3 --include "*.safetensors" --local-dir stacks/sagemaker/sam3/model
```

### Deploy

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate Docker to AWS Deep Learning Containers ECR registry
# 763104351884 is AWS's official account ID for Deep Learning Container images
# Required to pull the PyTorch SageMaker base image during CDK deployment
# See: https://aws.github.io/deep-learning-containers/reference/available_images/#pytorch-inference
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# Deploy infrastructure (OarcWsStorageStack + OarcWsPipelineStack)
# This builds and deploys Lambda functions and SageMaker model container
cdk deploy --all

# Start SageMaker endpoint (costs $0.736/hr)
python scripts/deploy_endpoint.py create

# Trigger the pipeline and watch for results
python scripts/run_pipeline.py

# Check endpoint status, cost, and auto-shutdown timer
python scripts/deploy_endpoint.py status

# Clean up when done
python scripts/deploy_endpoint.py delete
cdk destroy --all
```

## Project Structure

```
stacks/
  storage_stack.py        # S3 bucket (persists across pipeline redeployments)
  pipeline_stack.py       # Lambda, Step Functions, EventBridge, monitoring
  adjuster_stack.py       # Adjuster Lambda + DynamoDB routing table
  lambda_functions/       # Processor, adjuster, and endpoint monitor Lambda code (Docker)
  sagemaker/sam3/         # SageMaker SAM3 model container (Docker, 3.4GB)
scripts/
  config.py               # Reads config from cdk.json (do not edit)
  deploy_endpoint.py      # Create/delete/status SageMaker endpoint
  run_pipeline.py         # Upload test inputs and watch for results (and failures)
  adjuster_report.py      # Show adjuster routing decisions and sync reports
files/
  images/                 # Sample before/after satellite images
  inputs/                 # Sample JSON input files (1.json, 2.json, 3.json)
  reports/                # Generated markdown reports (output)
```

## Architecture

```
S3 Upload → EventBridge → Step Functions → SageMaker (async) → Lambda → Bedrock → S3 Report
```

**Event-driven workflow:**
1. User uploads JSON to S3 `inputs/` → EventBridge detects ObjectCreated event
2. Step Functions reads JSON, creates payload, invokes SageMaker async endpoint
3. SageMaker processes images with SAM3, writes comparison to S3 `compared/`
4. SageMaker writes output to S3 `async-out/` → triggers EventBridge again
5. Step Functions invokes Lambda processor
6. Lambda reads the before, after, and comparison images, calls Bedrock Claude for analysis
7. Lambda generates styled markdown report, saves to S3 `markdown/`

## Cost Protection

The SageMaker endpoint costs $0.736/hr ($530/month if left running). Three safeguards:

1. **Auto-shutdown** after 1 hour of inactivity (CloudWatch alarm on `InvocationsProcessed`)
2. **Daily cleanup** at 2 AM UTC (EventBridge schedule)
3. **Cost warnings** displayed after every `cdk deploy`

Check status anytime:
```bash
python scripts/deploy_endpoint.py status
```

## AWS Services Used

| Service | Purpose |
|---------|---------|
| S3 | Storage for images, model, and outputs |
| Lambda | Post-processing (Bedrock + report generation) |
| Step Functions | Workflow orchestration |
| SageMaker | ML inference (SAM3 async endpoint) |
| Bedrock | LLM analysis (Claude Opus 5.5) |
| EventBridge | S3 event routing + scheduled cleanup |
| CloudWatch | Endpoint monitoring + auto-shutdown alarm |
