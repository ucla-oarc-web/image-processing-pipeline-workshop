"""Pipeline stack - processing resources for the OARC Image Processing Pipeline.

Creates Lambda, Step Functions, EventBridge, and endpoint monitoring.
Depends on the StorageStack for the S3 bucket.

Deploy:   cdk deploy OarcWsPipelineStack
Destroy:  cdk destroy OarcWsPipelineStack
"""
import os
import aws_cdk as cdk
from constructs import Construct
from aws_cdk import (
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_s3 as s3,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
    aws_logs as logs,
    aws_sagemaker as sagemaker,
)

LAMBDA_DIR = os.path.join(os.path.dirname(__file__), "lambda_functions")


class PipelineStack(cdk.Stack):

    def __init__(self, scope: Construct, construct_id: str, *, bucket: s3.IBucket, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = self.node.try_get_context("resource_prefix")
        endpoint_name = self.node.try_get_context("endpoint_name")
        bedrock_model_id = self.node.try_get_context("bedrock_model_id")
        bucket_name = bucket.bucket_name

        # -----------------------------------------------------------
        # Processor Lambda (Bedrock analysis + report generation)
        # -----------------------------------------------------------
        processor = lambda_.DockerImageFunction(
            self, "ProcessorFunction",
            function_name=f"{prefix}-processor",
            code=lambda_.DockerImageCode.from_image_asset(
                os.path.join(LAMBDA_DIR, "processor"),
                platform=cdk.aws_ecr_assets.Platform.LINUX_AMD64,
            ),
            memory_size=512,
            timeout=cdk.Duration.minutes(3),
            environment={
                "BUCKET_NAME": bucket_name,
                "BEDROCK_MODEL_ID": bedrock_model_id,
            },
        )
        bucket.grant_read_write(processor)
        processor.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:{self.partition}:bedrock:*::foundation-model/{bedrock_model_id}",
                f"arn:{self.partition}:bedrock:*::foundation-model/{bedrock_model_id.removeprefix('us.')}",
                f"arn:{self.partition}:bedrock:{self.region}:{self.account}:inference-profile/{bedrock_model_id}",
            ],
        ))
        endpoint_arn = self.format_arn(service="sagemaker", resource="endpoint", resource_name=endpoint_name)
        # -----------------------------------------------------------
        # Step Functions state machine
        # -----------------------------------------------------------
        read_input = sfn_tasks.CallAwsService(
            self, "ReadInputJSON",
            service="s3", action="getObject",
            parameters={
                "Bucket": sfn.JsonPath.string_at("$.detail.bucket.name"),
                "Key": sfn.JsonPath.string_at("$.detail.object.key"),
            },
            iam_resources=[bucket.arn_for_objects("inputs/*")],
            result_path="$.s3Object",
            result_selector={"Body.$": "$.Body", "ParsedBody.$": "States.StringToJson($.Body)"},
        )

        parse_input = sfn.Pass(
            self, "ParseInputJSON",
            parameters={
                "bucket.$": "$.detail.bucket.name",
                "before_image.$": "States.Format('s3://{}/images/{}', $.detail.bucket.name, $.s3Object.ParsedBody.before)",
                "after_image.$": "States.Format('s3://{}/images/{}', $.detail.bucket.name, $.s3Object.ParsedBody.after)",
                "compared_output.$": "States.Format('s3://{}/compared/{}', $.detail.bucket.name, $.s3Object.ParsedBody.compared_output)",
                "text": "house, building, roof",
                "payloadKey.$": "States.Format('payload/{}.json', States.UUID())",
            },
        )

        create_payload = sfn_tasks.CallAwsService(
            self, "CreatePayload",
            service="s3", action="putObject",
            parameters={
                "Body.$": "$", "Bucket.$": "$.bucket",
                "Key.$": "$.payloadKey", "ContentType": "application/json",
            },
            iam_resources=[bucket.arn_for_objects("payload/*")],
            result_path="$.payloadFile",
        )

        call_sagemaker = sfn_tasks.CallAwsService(
            self, "CallSAM3",
            service="sagemakerruntime", action="invokeEndpointAsync",
            parameters={
                "EndpointName": endpoint_name,
                "InputLocation.$": "States.Format('s3://{}/{}', $.bucket, $.payloadKey)",
                "ContentType": "application/json",
            },
            iam_resources=[endpoint_arn],
            iam_action="sagemaker:InvokeEndpointAsync",
        )

        process_image = sfn_tasks.LambdaInvoke(
            self, "ProcessImage",
            lambda_function=processor,
            payload=sfn.TaskInput.from_object({
                "bucket": sfn.JsonPath.string_at("$.detail.bucket.name"),
                "key": sfn.JsonPath.string_at("$.detail.object.key"),
            }),
        )

        # Route based on S3 key pattern
        router = sfn.Choice(self, "CheckInputFormat")
        router.when(
            sfn.Condition.string_matches("$.detail.object.key", "inputs/*.json"),
            read_input.next(parse_input).next(create_payload).next(call_sagemaker),
        )
        router.when(
            sfn.Condition.and_(
                sfn.Condition.string_matches("$.detail.object.key", "async-out/*"),
                sfn.Condition.string_matches("$.detail.object.key", "*.out"),
            ),
            process_image,
        )
        router.otherwise(sfn.Pass(self, "UnmatchedKeyPattern"))

        state_machine = sfn.StateMachine(
            self, "ImagePipelineStateMachine",
            state_machine_name=f"{prefix}-image-pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(router),
            state_machine_type=sfn.StateMachineType.STANDARD,
            logs=sfn.LogOptions(
                destination=logs.LogGroup(self, "StateMachineLogGroup",
                    log_group_name=f"/aws/vendedlogs/states/{prefix}-image-pipeline",
                    removal_policy=cdk.RemovalPolicy.DESTROY),
                level=sfn.LogLevel.ALL, include_execution_data=False,
            ),
        )

        # EventBridge rule - triggers on S3 ObjectCreated events
        events.Rule(
            self, "S3ObjectCreatedRule",
            rule_name=f"{prefix}-s3-pipeline-trigger",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket_name]},
                    "object": {"key": [
                        {"wildcard": "inputs/*.json"},
                        {"wildcard": "async-out/*.out"},
                    ]},
                },
            ),
            targets=[events_targets.SfnStateMachine(state_machine)],
        )

        # -----------------------------------------------------------
        # SageMaker Model + EndpointConfig (no cost, created once)
        # -----------------------------------------------------------
        tags = [cdk.CfnTag(key=k, value=v)
                for k, v in (self.node.try_get_context("tags") or {}).items()]

        sam3_image = cdk.aws_ecr_assets.DockerImageAsset(
            self, "SAM3Image",
            directory=os.path.join(os.path.dirname(__file__), "sagemaker", "sam3"),
            platform=cdk.aws_ecr_assets.Platform.LINUX_AMD64,
        )

        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            role_name=f"{prefix}-sagemaker-execution-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )
        bucket.grant_read(sagemaker_role)
        bucket.grant_write(sagemaker_role, "async-out/*")
        bucket.grant_write(sagemaker_role, "async-failure/*")
        bucket.grant_write(sagemaker_role, "compared/*")
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:GetAuthorizationToken"],
            resources=["*"],
        ))
        sagemaker_role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:{self.partition}:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"],
        ))

        sm_model = sagemaker.CfnModel(
            self, "SAM3Model",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=sam3_image.image_uri,
            ),
            tags=tags,
        )
        sm_model.node.add_dependency(sagemaker_role)

        sm_endpoint_config = sagemaker.CfnEndpointConfig(
            self, "SAM3EndpointConfig",
            production_variants=[sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                variant_name="primary",
                model_name=sm_model.attr_model_name,
                initial_instance_count=1,
                instance_type="ml.g4dn.xlarge",
            )],
            async_inference_config=sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty(
                output_config=sagemaker.CfnEndpointConfig.AsyncInferenceOutputConfigProperty(
                    s3_output_path=f"s3://{bucket_name}/async-out/",
                    s3_failure_path=f"s3://{bucket_name}/async-failure/",
                ),
                client_config=sagemaker.CfnEndpointConfig.AsyncInferenceClientConfigProperty(
                    max_concurrent_invocations_per_instance=1,
                ),
            ),
            tags=tags,
        )
        sm_endpoint_config.add_dependency(sm_model)

        # -----------------------------------------------------------
        # Outputs
        # -----------------------------------------------------------
        cdk.CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        cdk.CfnOutput(self, "EndpointName", value=endpoint_name)
        cdk.CfnOutput(self, "EndpointConfigName", value=sm_endpoint_config.attr_endpoint_config_name)
        cdk.CfnOutput(self, "CostWarning",
                       value="WARNING: SageMaker endpoint costs ~$0.736/hr (~$530/month). "
                             "Auto-shutdown after 1hr inactivity. Daily cleanup at 2AM UTC.")
