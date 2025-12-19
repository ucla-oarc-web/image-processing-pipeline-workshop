#!/bin/bash

BUCKET="oarc-image-processing-pipeline"
REGION="us-west-2"

echo "Clearing S3 bucket ${BUCKET}..."

# Delete all objects in the bucket
aws s3 rm s3://oarc-image-processing-pipeline --recursive --region us-west-2

echo "Uploading files to bucket root..."

# Upload images directory
aws s3 cp ../files/images/ s3://${BUCKET}/images/ --recursive --region ${REGION}

# Upload prompt.txt to root
aws s3 cp ../files/prompt.txt s3://${BUCKET}/prompt.txt --region ${REGION}

# Upload sam3-model.tar.gz to root
aws s3 cp ../files/sam3-model.tar.gz s3://${BUCKET}/sam3-model.tar.gz --region ${REGION}

echo "Bucket reset complete!"
aws s3 ls s3://${BUCKET}/ --region ${REGION}
