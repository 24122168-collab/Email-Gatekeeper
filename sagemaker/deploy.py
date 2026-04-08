"""
deploy.py — SageMaker Endpoint Deployment Script
=================================================
Packages the rule-based classifier and deploys it as a real-time
SageMaker endpoint with full CloudWatch monitoring support.

Prerequisites:
    pip install boto3 sagemaker

Usage:
    python deploy.py                         # deploy only
    python deploy.py --test                  # deploy + smoke tests
    python deploy.py --monitor               # deploy + smoke tests + CW dashboard
    python deploy.py --delete                # tear down endpoint

AWS permissions required on your IAM user/role:
    sagemaker:CreateModel, CreateEndpointConfig, CreateEndpoint
    sagemaker:InvokeEndpoint, DeleteEndpoint
    s3:PutObject          (SageMaker default bucket)
    iam:PassRole          (SageMaker execution role)
    iam:PutRolePolicy     (to attach CW policy to execution role)
    cloudwatch:PutDashboard
"""

import argparse
import json
import os
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# ── Configuration — edit these ────────────────────────────────────────────────
ENDPOINT_NAME   = "email-gatekeeper-v1"
INSTANCE_TYPE   = "ml.t2.medium"        # cheapest real-time; upgrade for prod
SKLEARN_VERSION = "1.2-1"
CW_NAMESPACE    = "EmailGatekeeper/Inference"   # must match inference.py
REGION          = boto3.session.Session().region_name or "us-east-1"

_MODEL_FILES = [
    os.path.join(os.path.dirname(__file__), "inference.py"),
    os.path.join(os.path.dirname(__file__), "classifier.py"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_model_tar(s3_client, bucket: str, prefix: str) -> str:
    """Bundle inference.py + classifier.py → model.tar.gz → S3, return URI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fpath in _MODEL_FILES:
                tar.add(fpath, arcname=os.path.basename(fpath))
            # config.json lets model_fn read runtime overrides without redeploying
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"version": "1.0.0", "cw_namespace": CW_NAMESPACE}, f)
            tar.add(config_path, arcname="config.json")

        s3_key = f"{prefix}/model.tar.gz"
        s3_client.upload_file(tar_path, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"  ✅ model.tar.gz → {s3_uri}")
        return s3_uri


def _ensure_cloudwatch_policy(role_name: str) -> None:
    """
    Attach an inline IAM policy to the SageMaker execution role so the
    container can call cloudwatch:PutMetricData. Idempotent.
    Scoped to CW_NAMESPACE only — least-privilege.
    """
    iam = boto3.client("iam", region_name=REGION)
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Sid":      "EmailGatekeeperCWMetrics",
            "Effect":   "Allow",
            "Action":   ["cloudwatch:PutMetricData"],
            "Resource": "*",
            "Condition": {
                "StringEquals": {"cloudwatch:namespace": CW_NAMESPACE}
            },
        }],
    }
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="EmailGatekeeperCloudWatchMetrics",
        PolicyDocument=json.dumps(policy),
    )
    print(f"  ✅ CloudWatch IAM policy attached → role: {role_name}")


def _create_cloudwatch_dashboard() -> None:
    """
    Create (or overwrite) a 6-widget CloudWatch dashboard:
      Row 1 — ExactMatch rate        | PartialMatch rate
      Row 2 — SecurityMiss count     | WrongClassification count
      Row 3 — Avg RewardScore        | SecurityBreachFlag count
    """
    cw = boto3.client("cloudwatch", region_name=REGION)

    def _widget(title, metric, stat="Sum", color="#1f77b4"):
        return {
            "type": "metric",
            "width": 12,
            "height": 6,
            "properties": {
                "title":   title,
                "metrics": [[CW_NAMESPACE, metric,
                              "EndpointName", ENDPOINT_NAME]],
                "stat":    stat,
                "period":  300,
                "view":    "timeSeries",
                "color":   color,
                "region":  REGION,
            },
        }

    dashboard_body = {
        "widgets": [
            _widget("✅ Exact Matches (5-min)",       "ExactMatch",          color="#2ca02c"),
            _widget("🔶 Partial Matches (5-min)",     "PartialMatch",        color="#ff7f0e"),
            _widget("🚨 Security Misses (5-min)",     "SecurityMiss",        color="#d62728"),
            _widget("❌ Wrong Classifications",        "WrongClassification", color="#9467bd"),
            _widget("📈 Avg Reward Score",            "RewardScore",
                    stat="Average",                                           color="#8c564b"),
            _widget("🔒 Security Breach Flags",       "SecurityBreachFlag",  color="#e377c2"),
        ]
    }

    cw.put_dashboard(
        DashboardName="EmailGatekeeper-Inference",
        DashboardBody=json.dumps(dashboard_body),
    )
    print("  ✅ CloudWatch dashboard created: EmailGatekeeper-Inference")
    print(f"     https://{REGION}.console.aws.amazon.com/cloudwatch/home"
          f"?region={REGION}#dashboards:name=EmailGatekeeper-Inference")


def _smoke_test(sm_runtime) -> None:
    """Run 3 labelled test cases — one per urgency level — against the live endpoint."""
    test_cases = [
        {
            "name": "Security Breach",
            "payload": {
                "subject": "Your account has been hacked",
                "body":    "Unauthorized access detected. Reset your password immediately.",
                # ground_truth triggers CW metric emission during smoke test
                "ground_truth": {"urgency": 2, "routing": 1, "resolution": 2},
            },
            "expected_category": "Security Breach",
        },
        {
            "name": "Billing Dispute",
            "payload": {
                "subject": "Refund not received",
                "body":    "I requested a refund 3 weeks ago and have not received it.",
                "ground_truth": {"urgency": 1, "routing": 2, "resolution": 2},
            },
            "expected_category": "Billing",
        },
        {
            "name": "Spam",
            "payload": {
                "subject": "You won a free prize!",
                "body":    "Claim your free offer now. Win big today!",
                "ground_truth": {"urgency": 0, "routing": 0, "resolution": 0},
            },
            "expected_category": "General",
        },
    ]

    print("\n  Running smoke tests...")
    all_passed = True

    for tc in test_cases:
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(tc["payload"]),
        )
        result   = json.loads(response["Body"].read())
        category = result["triage"]["category"]
        match    = result.get("match_result", {})
        passed   = category == tc["expected_category"]
        icon     = "✅" if passed else "❌"
        all_passed = all_passed and passed

        print(f"  {icon} [{tc['name']}]  "
              f"category='{category}'  "
              f"match={match.get('status','?')}  "
              f"reward={match.get('reward','?')}")

    print(f"\n  Smoke tests: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def deploy(create_dashboard: bool = False) -> object:
    sess      = sagemaker.Session()
    bucket    = sess.default_bucket()
    role      = sagemaker.get_execution_role()
    role_name = role.split("/")[-1]
    s3_client = boto3.client("s3", region_name=REGION)

    print(f"\n{'═' * 62}")
    print("  Email Gatekeeper — SageMaker + CloudWatch Deployment")
    print(f"  Endpoint     : {ENDPOINT_NAME}")
    print(f"  Instance     : {INSTANCE_TYPE}")
    print(f"  Region       : {REGION}")
    print(f"  CW Namespace : {CW_NAMESPACE}")
    print(f"{'═' * 62}\n")

    # 1. Attach CloudWatch IAM policy to execution role
    print("  Attaching CloudWatch IAM policy...")
    _ensure_cloudwatch_policy(role_name)

    # 2. Package and upload model artifacts
    print("  Packaging model artifacts...")
    model_uri = _build_model_tar(s3_client, bucket, "email-gatekeeper/model")

    # 3. Create SageMaker SKLearn model
    #    env passes ENDPOINT_NAME into the container so model_fn can read it
    model = SKLearnModel(
        model_data=model_uri,
        role=role,
        entry_point="inference.py",
        framework_version=SKLEARN_VERSION,
        sagemaker_session=sess,
        name=f"{ENDPOINT_NAME}-model",
        env={"SAGEMAKER_ENDPOINT_NAME": ENDPOINT_NAME},
    )

    # 4. Deploy real-time endpoint
    print("  Deploying endpoint (~5 min)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"\n  ✅ Endpoint live: {ENDPOINT_NAME}")

    # 5. Optional CloudWatch dashboard
    if create_dashboard:
        print("  Creating CloudWatch dashboard...")
        _create_cloudwatch_dashboard()

    return predictor


def delete_endpoint() -> None:
    sm = boto3.client("sagemaker", region_name=REGION)
    print(f"  Deleting endpoint: {ENDPOINT_NAME}")
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print("  ✅ Endpoint deleted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete",  action="store_true", help="Delete the endpoint")
    parser.add_argument("--test",    action="store_true", help="Run smoke tests after deploy")
    parser.add_argument("--monitor", action="store_true", help="Create CloudWatch dashboard")
    args = parser.parse_args()

    if args.delete:
        delete_endpoint()
    else:
        predictor  = deploy(create_dashboard=args.monitor)
        sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
        if args.test or args.monitor:
            _smoke_test(sm_runtime)
        print(
            f"\n  Invoke example (with ground_truth for CW metrics):\n"
            f"  aws sagemaker-runtime invoke-endpoint \\\n"
            f"    --endpoint-name {ENDPOINT_NAME} \\\n"
            f"    --content-type application/json \\\n"
            f"    --body '{{\"subject\":\"hacked\",\"body\":\"unauthorized access\","
            f"\"ground_truth\":{{\"urgency\":2,\"routing\":1,\"resolution\":2}}}}' \\\n"
            f"    response.json && cat response.json\n"
        )
