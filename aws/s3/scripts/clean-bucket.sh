#!/bin/bash

BUCKET="oarc-image-processing-pipeline"
REGION="us-west-2"

echo "Cleaning outputs in S3 bucket ${BUCKET}..."

aws s3 rm s3://${BUCKET}/inputs/ --recursive && \
aws s3 rm s3://${BUCKET}/async-failure/ --recursive && \
aws s3 rm s3://${BUCKET}/async-out/ --recursive && \
aws s3 rm s3://${BUCKET}/compared/ --recursive && \
aws s3 rm s3://${BUCKET}/llm_output/ --recursive && \
aws s3 rm s3://${BUCKET}/markdown/ --recursive && \
aws s3 rm s3://${BUCKET}/payload/ --recursive

echo ""
echo ""
echo "Bucket cleaning complete!"
aws s3 ls s3://${BUCKET}/ --region ${REGION}
