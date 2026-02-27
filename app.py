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
from stacks.pipeline_stack import PipelineStack
from stacks.adjuster_stack import OarcWsAdjusterStack

app = cdk.App()

storage = StorageStack(app, "OarcWsStorageStack",
                       description="OARC Image Pipeline - S3 bucket for pipeline data")

pipeline = PipelineStack(app, "OarcWsPipelineStack",
                         bucket=storage.bucket,
                         description="OARC Image Pipeline - Lambda, Step Functions, EventBridge, monitoring")

adjuster = OarcWsAdjusterStack(app, "OarcWsAdjusterStack",
                               description="OARC Image Pipeline - Downstream adjuster for insurance processing")

pipeline.add_dependency(storage)
adjuster.add_dependency(storage)

tags: dict = app.node.try_get_context("tags") or {}
for tag_key, tag_value in tags.items():
    cdk.Tags.of(app).add(tag_key, tag_value)

app.synth()
