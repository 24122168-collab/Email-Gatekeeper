"""
email_gatekeeper_stack.py — AWS CDK Stack for the Email Gatekeeper.

Resources created:
  - S3 bucket          : receives raw .eml files from SES
  - Lambda function    : classifies each email using the rule-based engine
  - DynamoDB table     : stores every triage result (email_id as partition key)
  - SNS topic          : fires an alert whenever a Security Breach is detected
  - SES receipt rule   : routes inbound email → S3 bucket  (requires verified domain)
  - IAM roles/policies : least-privilege access for Lambda → S3, DynamoDB, SNS

Deploy:
  cd cdk
  pip install aws-cdk-lib constructs
  cdk bootstrap          # first time only per account/region
  cdk deploy
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_s3_notifications as s3n,
    aws_ses as ses,
    aws_ses_actions as ses_actions,
    aws_iam as iam,
)
from constructs import Construct


class EmailGatekeeperStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── 1. S3 bucket — stores raw .eml files delivered by SES ─────────────
        email_bucket = s3.Bucket(
            self, "EmailBucket",
            bucket_name=f"email-gatekeeper-inbox-{self.account}",
            # Block all public access — emails are private
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            # Auto-delete raw emails after 30 days to control storage costs
            lifecycle_rules=[
                s3.LifecycleRule(expiration=Duration.days(30))
            ],
            removal_policy=RemovalPolicy.RETAIN,    # keep emails if stack is deleted
        )

        # ── 2. DynamoDB table — persists every triage decision ─────────────────
        results_table = dynamodb.Table(
            self, "EmailResultsTable",
            table_name="EmailTriageResults",
            partition_key=dynamodb.Attribute(
                name="email_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,  # serverless billing
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ── 3. SNS topic — security breach alerts ─────────────────────────────
        security_topic = sns.Topic(
            self, "SecurityAlertTopic",
            topic_name="EmailGatekeeperSecurityAlerts",
            display_name="Email Gatekeeper — Security Breach Alerts",
        )

        # Add your alert email here — replace with a real address
        security_topic.add_subscription(
            sns_subs.EmailSubscription("security-team@your-domain.com")
        )

        # ── 4. Lambda function ─────────────────────────────────────────────────
        classifier_fn = lambda_.Function(
            self, "EmailClassifierFn",
            function_name="EmailGatekeeperClassifier",
            runtime=lambda_.Runtime.PYTHON_3_12,
            # Points to the ../lambda/ directory — CDK zips it automatically
            code=lambda_.Code.from_asset("../lambda"),
            handler="handler.lambda_handler",
            timeout=Duration.seconds(30),
            memory_size=256,                        # classifier is CPU-light
            environment={
                "EMAIL_RESULTS_TABLE":      results_table.table_name,
                "SECURITY_ALERT_TOPIC_ARN": security_topic.topic_arn,
            },
        )

        # Grant Lambda least-privilege access to each resource
        email_bucket.grant_read(classifier_fn)
        results_table.grant_write_data(classifier_fn)
        security_topic.grant_publish(classifier_fn)

        # ── 5. S3 → Lambda trigger ─────────────────────────────────────────────
        # Fires whenever SES drops a new .eml into the bucket
        email_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(classifier_fn),
        )

        # ── 6. SES receipt rule — routes inbound email to S3 ──────────────────
        # IMPORTANT: your domain must be verified in SES before this works.
        # Replace "mail.your-domain.com" with your actual verified domain.
        rule_set = ses.ReceiptRuleSet(
            self, "EmailRuleSet",
            rule_set_name="EmailGatekeeperRuleSet",
        )

        rule_set.add_rule(
            "StoreInS3Rule",
            recipients=["inbox@mail.your-domain.com"],  # ← replace with your address
            actions=[
                ses_actions.S3(
                    bucket=email_bucket,
                    object_key_prefix="incoming/",      # all emails land under incoming/
                )
            ],
            scan_enabled=True,                          # enable SES spam/virus scanning
        )

        # ── 7. Allow SES to write to the S3 bucket ────────────────────────────
        email_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowSESPuts",
                principals=[iam.ServicePrincipal("ses.amazonaws.com")],
                actions=["s3:PutObject"],
                resources=[email_bucket.arn_for_objects("incoming/*")],
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account}
                },
            )
        )

        # ── 8. CloudFormation outputs — useful after deploy ────────────────────
        cdk.CfnOutput(self, "BucketName",      value=email_bucket.bucket_name)
        cdk.CfnOutput(self, "TableName",       value=results_table.table_name)
        cdk.CfnOutput(self, "LambdaArn",       value=classifier_fn.function_arn)
        cdk.CfnOutput(self, "SecurityTopicArn",value=security_topic.topic_arn)
