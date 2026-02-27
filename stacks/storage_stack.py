"""Storage stack - S3 bucket for the OARC Image Processing Pipeline.

Separated from the pipeline so the bucket (and its data) persists
across pipeline redeployments.

Deploy:   cdk deploy OarcWsStorageStack
Destroy:  cdk destroy OarcWsStorageStack
"""
import aws_cdk as cdk
from constructs import Construct
from aws_cdk import aws_s3 as s3


class StorageStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket_name = self.node.try_get_context("bucket_name")

        self.bucket = s3.Bucket(
            self,
            "PipelineBucket",
            bucket_name=bucket_name,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            # EventBridge integration required for Step Functions triggers
            event_bridge_enabled=True,
        )

        cdk.CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
