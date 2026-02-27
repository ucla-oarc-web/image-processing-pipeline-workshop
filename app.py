"""CDK app entry point for the OARC Image Processing Pipeline.

Three stacks:
  - OarcWsStorageStack:  S3 bucket (persists across pipeline redeployments)
  - OarcWsPipelineStack: Lambda, Step Functions, EventBridge, monitoring
  - OarcWsAdjusterStack: Downstream adjuster Lambda for insurance processing

Deploy all:    cdk deploy --all
Deploy one:    cdk deploy OarcWsStorageStack
Destroy all:   cdk destroy --all
"""
import aws_cdk as cdk
from stacks.storage_stack import StorageStack

app = cdk.App()

storage = StorageStack(app, "OarcWsStorageStack",
                       description="OARC Image Pipeline - S3 bucket for pipeline data")

tags: dict = app.node.try_get_context("tags") or {}
for tag_key, tag_value in tags.items():
    cdk.Tags.of(app).add(tag_key, tag_value)

app.synth()
