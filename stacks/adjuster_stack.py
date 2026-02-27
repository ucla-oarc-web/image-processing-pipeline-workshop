import aws_cdk as cdk
from constructs import Construct
from aws_cdk import (
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_s3 as s3,
)

class OarcWsAdjusterStack(cdk.Stack):
    """Stack for downstream adjuster Lambda that processes compared images."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Read config from cdk.json
        bucket_name = self.node.try_get_context("bucket_name")
        model_id = self.node.try_get_context("adjuster_model_id") or self.node.try_get_context("bedrock_model_id")
        prefix = self.node.try_get_context("resource_prefix")

        # Reference existing S3 bucket
        bucket = s3.Bucket.from_bucket_name(self, "Bucket", bucket_name)

        # DynamoDB table for routing
        routing_table = dynamodb.Table(
            self,
            "RoutingTable",
            partition_key=dynamodb.Attribute(name="routing_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
        )

        # Adjuster Lambda (Docker container)
        adjuster_lambda = _lambda.DockerImageFunction(
            self,
            "AdjusterFunction",
            function_name=f"{prefix}-adjuster",
            code=_lambda.DockerImageCode.from_image_asset(
                "stacks/lambda_functions/adjuster",
                platform=cdk.aws_ecr_assets.Platform.LINUX_AMD64,
                asset_name=f"{prefix}-adjuster-lambda",
            ),
            timeout=cdk.Duration.minutes(10),
            memory_size=1024,
            retry_attempts=0,
            environment={
                "ROUTING_TABLE_NAME": routing_table.table_name,
                "BEDROCK_MODEL_ID": model_id,
            },
        )

        # Permissions
        routing_table.grant_write_data(adjuster_lambda)
        bucket.grant_read_write(adjuster_lambda)

        # Scope Bedrock to specific model
        adjuster_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    f"arn:{self.partition}:bedrock:*::foundation-model/{model_id}",
                    f"arn:{self.partition}:bedrock:*::foundation-model/{model_id.removeprefix('us.')}",
                    f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/{model_id}",
                ],
            )
        )

        # EventBridge rule for compared/ prefix (not S3 notifications)
        events.Rule(
            self,
            "ComparedObjectCreatedRule",
            rule_name=f"{prefix}-adjuster-trigger",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket_name]},
                    "object": {"key": [{"wildcard": "compared/*"}]},
                },
            ),
            targets=[events_targets.LambdaFunction(
                adjuster_lambda,
                retry_attempts=0,
            )],
        )

        # Stack outputs
        cdk.CfnOutput(self, "RoutingTableName", value=routing_table.table_name)
