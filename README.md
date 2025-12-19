# OARC Image Processing Pipeline Workshop

This repository contains the code and configuration for an image processing pipeline using AWS services, including Lambda, Step Functions, S3, and SageMaker.

This project does leverage the AWS Deep Learning Image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker. This is an image provided by AWS that is able to run the model and inference code, and provides hooks that we can leverage so we don't need to create the api routes for invoke endpoint and health.

See [deep-learning-containers/available_images.md](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) for more information.

## Project Structure
- `aws/`: Contains AWS-specific configurations and code.
  - `event-bridge/`: Example cloudformation template.
  - `lambda/`: AWS Lambda function code and example SAM template.
    - `scripts/`: Helper code to create/update layers, update function code.
  - `step-functions/`: Example cloudformation template.
  - `s3/`:
    - `files/`: S3 bucket example images & json, along with the model.
    - `scripts/`: Deletes everything in the s3 bucket and re uploads model and example files.
- `model/`: Model inference code with requirements and model config/safetensors.
- `scripts/`: Utility scripts for testing and deployment.
  - `sagemaker/`: Scripts to create the model, endpoint config, and endpoint.
  - `tests/`: Some helpful test scripts to trigger the workflow.

## Setup Instructions
1. **AWS Console**: You will need to create the s3 bucket, as well as the shell for the lambda function. The cloud formation templates were created to help guide your creation of these services and not meant to create the services for you. IAM roles/policies will need to be created for all of these services.

2. **AWS Configuration**: Ensure you have AWS CLI configured with the necessary permissions to create and manage Lambda functions, Step Functions, and S3 buckets.

3. **S3 Bucket**: Create an S3 bucket to store input and output files. Update the bucket name in the scripts as needed.

4. **Deploy Lambda Function**: Deploy the Lambda function using the AWS Management Console. Consult the SAM template `aws/lambda/template.yml` and function code `aws/lambda/lambda_function.py` for reference.

5. **Configure Step Functions**: Deploy the Step Functions state machine using the AWS Management Console. See the Cloud Formation template `aws/step-functions/template.json` for reference.

6. **Upload Input Files**: Place your input JSON files in the designated S3 bucket `inputs/` folder to trigger the processing pipeline or use/alter the helper script `scripts/tests/trigger-processing.sh`.
