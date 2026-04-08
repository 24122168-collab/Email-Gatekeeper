"""
handler.py — AWS Lambda entry point for the Email Gatekeeper.

Trigger paths:
  A) S3 Event    : SES stores raw .eml → S3 → Lambda (s3:ObjectCreated)
  B) Direct JSON : {"subject": "...", "body": "..."} for testing / API Gateway

On each invocation:
  1. Parse the email (S3 object or direct payload)
  2. Extract features  (classifier.extract_features)
  3. Classify          (classifier.classify)
  4. Persist result    → DynamoDB table  (EMAIL_RESULTS_TABLE env var)
  5. Alert on breach   → SNS topic       (SECURITY_ALERT_TOPIC_ARN env var)
  6. Return JSON result
"""

import json
import os
import email
import uuid
import logging
from datetime import datetime, timezone

import boto3

from classifier import classify, decode, extract_features

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients — initialised once at cold-start for connection reuse
_s3       = boto3.client("s3")
_dynamodb = boto3.resource("dynamodb")
_sns      = boto3.client("sns")

# Environment variables injected by CDK
_TABLE_NAME  = os.environ.get("EMAIL_RESULTS_TABLE", "")
_TOPIC_ARN   = os.environ.get("SECURITY_ALERT_TOPIC_ARN", "")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_eml(raw_bytes: bytes) -> tuple[str, str]:
    """Extract subject and plain-text body from a raw .eml byte string."""
    msg = email.message_from_bytes(raw_bytes)
    subject = msg.get("Subject", "")

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                break
    else:
        body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

    return subject, body


def _fetch_from_s3(bucket: str, key: str) -> tuple[str, str]:
    """Download a raw .eml from S3 and return (subject, body)."""
    logger.info("Fetching s3://%s/%s", bucket, key)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    return _parse_eml(raw)


def _save_to_dynamodb(record: dict) -> None:
    """Persist the triage result to DynamoDB (best-effort, non-blocking)."""
    if not _TABLE_NAME:
        return
    try:
        table = _dynamodb.Table(_TABLE_NAME)
        table.put_item(Item=record)
    except Exception as exc:
        logger.error("DynamoDB write failed: %s", exc)


def _alert_security(record: dict) -> None:
    """Publish an SNS alert when a Security Breach is detected."""
    if not _TOPIC_ARN:
        return
    try:
        _sns.publish(
            TopicArn=_TOPIC_ARN,
            Subject="🚨 Security Breach Email Detected",
            Message=json.dumps(record, indent=2),
        )
        logger.info("SNS alert published for email_id=%s", record.get("email_id"))
    except Exception as exc:
        logger.error("SNS publish failed: %s", exc)


# ── Main handler ─────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Unified entry point for S3-triggered and direct-invocation events.

    S3 event shape (from SES → S3 → Lambda notification):
      {"Records": [{"s3": {"bucket": {"name": "..."}, "object": {"key": "..."}}}]}

    Direct invocation shape (for testing or API Gateway):
      {"subject": "Your invoice is overdue", "body": "Please pay immediately."}
    """
    logger.info("Event received: %s", json.dumps(event)[:500])

    # ── Determine input source ────────────────────────────────────────────────
    records = event.get("Records", [])

    if records and records[0].get("eventSource") == "aws:s3":
        # Path A: triggered by S3 object creation (SES-delivered email)
        s3_info = records[0]["s3"]
        bucket  = s3_info["bucket"]["name"]
        key     = s3_info["object"]["key"]
        subject, body = _fetch_from_s3(bucket, key)
        source_ref = f"s3://{bucket}/{key}"
    else:
        # Path B: direct JSON invocation (testing / API Gateway)
        subject    = event.get("subject", "")
        body       = event.get("body", "")
        source_ref = "direct-invocation"

    if not subject and not body:
        return {"statusCode": 400, "body": "No email content found in event."}

    # ── Classify ──────────────────────────────────────────────────────────────
    features              = extract_features(subject, body)
    urgency, routing, res = classify(features)
    result                = decode(urgency, routing, res)

    # ── Build persistence record ──────────────────────────────────────────────
    email_id = str(uuid.uuid4())
    record = {
        "email_id":        email_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "source":          source_ref,
        "subject":         subject[:500],           # cap to avoid DDB item size issues
        "detected_keywords": features["keywords"],
        "sentiment":       features["sentiment"],
        "context":         features["context"],
        **result,                                   # urgency/routing/resolution labels + codes
    }

    logger.info("Classification result: %s", json.dumps(result))

    # ── Persist & alert ───────────────────────────────────────────────────────
    _save_to_dynamodb(record)

    if urgency == 2:                                # Security Breach
        _alert_security(record)

    return {
        "statusCode": 200,
        "body": json.dumps(record),
    }
