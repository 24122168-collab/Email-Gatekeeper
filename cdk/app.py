#!/usr/bin/env python3
"""
app.py — CDK application entry point.

Usage:
    cd cdk
    pip install aws-cdk-lib constructs
    cdk bootstrap        # first time only
    cdk synth            # preview CloudFormation template
    cdk deploy           # deploy to AWS
    cdk destroy          # tear down all resources
"""

import aws_cdk as cdk
from email_gatekeeper_stack import EmailGatekeeperStack

app = cdk.App()

EmailGatekeeperStack(
    app,
    "EmailGatekeeperStack",
    # Pin to a specific account + region to avoid environment-agnostic limitations
    # (required for SES receipt rules and S3 bucket policies).
    # Replace with your actual AWS account ID and preferred region.
    env=cdk.Environment(
        account="123456789012",     # ← replace with your AWS account ID
        region="us-east-1",         # ← SES inbound is only available in us-east-1,
    ),                              #   us-west-2, and eu-west-1
)

app.synth()
