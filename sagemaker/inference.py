"""
inference.py  —  SageMaker Entry Point  |  Email Gatekeeper  |  Phase 1
========================================================================

Think of this file like a circuit board with 4 connectors.
SageMaker plugs into each one in order, every time a request arrives:

  [1] model_fn   → Power-on.  Runs ONCE when the server starts.
  [2] input_fn   → Input pin.  Reads the raw HTTP request bytes.
  [3] predict_fn → Logic gate. Runs your classifier, scores the result.
  [4] output_fn  → Output pin. Sends the JSON response back.

Your classifier lives in classifier.py (same folder).
No GPU, no heavy ML libraries needed — pure Python logic.
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone

# classifier.py must be in the same folder as this file
from classifier import classify, decode, extract_features

# SageMaker streams all logger.info() calls to CloudWatch Logs automatically
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Reward weights (must match environment.py exactly) ────────────────────────
# These are the scores your RL agent learned against.
# They are used here only for logging — not for routing decisions.
_REWARDS = {
    "EXACT":         1.0,   # all 3 dimensions correct
    "PARTIAL_1":     0.2,   # urgency correct, 1 other dimension wrong
    "PARTIAL_2":     0.1,   # urgency correct, both other dimensions wrong
    "SECURITY_MISS": -2.0,  # security email but urgency was NOT flagged as 2
    "WRONG":         0.0,   # urgency wrong on a non-security email
}

# ── SLA table: urgency code → response time target ───────────────────────────
_SLA = {
    0: {"priority": "P3", "respond_within_minutes": 1440},  # General  — 24 h
    1: {"priority": "P2", "respond_within_minutes": 240},   # Billing  —  4 h
    2: {"priority": "P1", "respond_within_minutes": 15},    # Security — 15 min
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Partial-match scorer
# ─────────────────────────────────────────────────────────────────────────────

def _score_match(predicted: tuple, ground_truth: dict) -> dict:
    """
    Compare the 3 predicted dimensions against the known correct answer.

    Only called when the request includes a "ground_truth" field.
    Useful for:
      - Offline evaluation / batch testing
      - Logging accuracy metrics to CloudWatch

    Returns a dict with:
      status       — one of EXACT / PARTIAL_1 / PARTIAL_2 / SECURITY_MISS / WRONG
      reward       — float score matching your RL reward function
      wrong_fields — list of dimension names that were predicted incorrectly
    """
    p_urgency, p_routing, p_resolution = predicted

    g_urgency    = int(ground_truth["urgency"])
    g_routing    = int(ground_truth["routing"])
    g_resolution = int(ground_truth["resolution"])

    # Which of the 3 dimensions are correct?
    correct = {
        "urgency":    p_urgency    == g_urgency,
        "routing":    p_routing    == g_routing,
        "resolution": p_resolution == g_resolution,
    }
    wrong = [dim for dim, ok in correct.items() if not ok]

    # ── Decision tree (same priority order as environment.py) ─────────────────
    # Rule 1: Security email that was NOT flagged as security → worst outcome
    if g_urgency == 2 and p_urgency != 2:
        status = "SECURITY_MISS"

    # Rule 2: All 3 correct → perfect
    elif not wrong:
        status = "EXACT"

    # Rule 3: Urgency correct but 1 other dimension wrong → partial credit
    elif correct["urgency"] and len(wrong) == 1:
        status = "PARTIAL_1"

    # Rule 4: Urgency correct but both other dimensions wrong → small credit
    elif correct["urgency"] and len(wrong) == 2:
        status = "PARTIAL_2"

    # Rule 5: Urgency itself wrong → no credit
    else:
        status = "WRONG"

    logger.info(
        "MATCH_EVAL | status=%s reward=%.1f wrong_fields=%s",
        status, _REWARDS[status], wrong
    )

    return {
        "status":       status,
        "reward":       _REWARDS[status],
        "correct_dims": correct,
        "wrong_fields": wrong,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [1] model_fn  —  Power-on. Runs ONCE at container start.
# ─────────────────────────────────────────────────────────────────────────────

def model_fn(model_dir: str) -> dict:
    """
    SageMaker calls this once when the container boots up.
    model_dir is the folder where SageMaker unpacks your model.tar.gz.

    For a rule-based classifier there are no weights to load.
    We just return a config dict that predict_fn will use.
    """
    logger.info("model_fn | model_dir=%s", model_dir)

    # Optional: load a config.json from your model.tar.gz to override defaults
    # at runtime without redeploying (e.g. change SLA targets).
    config_path = os.path.join(model_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        logger.info("Config loaded: %s", config)

    model = {
        "version":       config.get("version", "1.0.0"),
        "sla":           config.get("sla", _SLA),
        # SageMaker injects the endpoint name as an env var
        "endpoint_name": os.environ.get("SAGEMAKER_ENDPOINT_NAME", "local"),
    }

    logger.info("Model ready | version=%s endpoint=%s",
                model["version"], model["endpoint_name"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# [2] input_fn  —  Input pin. Deserialise the raw HTTP request.
# ─────────────────────────────────────────────────────────────────────────────

def input_fn(request_body: str | bytes, content_type: str) -> dict:
    """
    Converts the raw bytes from the HTTP POST body into a Python dict.

    Accepted request formats:

    Format A — JSON with raw email text (most common):
        {
            "subject": "Your account was hacked",
            "body":    "We detected unauthorized access..."
        }

    Format B — JSON with pre-extracted features (faster, skips NLP):
        {
            "keywords": ["hacked", "password"],
            "sentiment": "negative",
            "context":   "security"
        }

    Format C — Add ground_truth to either format above for accuracy scoring:
        {
            "subject": "...",
            "body":    "...",
            "ground_truth": {"urgency": 2, "routing": 1, "resolution": 2}
        }
    """
    logger.info("input_fn | content_type=%s", content_type)

    ct = content_type.lower().split(";")[0].strip()

    if ct == "application/json":
        if isinstance(request_body, bytes):
            request_body = request_body.decode("utf-8")
        payload = json.loads(request_body)

    elif ct == "text/plain":
        # Accept raw email text directly — treat entire body as email body
        text    = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
        payload = {"subject": "", "body": text}

    else:
        raise ValueError(
            f"Unsupported content type: '{content_type}'. "
            "Send 'application/json' or 'text/plain'."
        )

    # Must have at least something to classify
    if not any([payload.get("subject"), payload.get("body"),
                payload.get("keywords"), payload.get("context")]):
        raise ValueError(
            "Request must include 'subject', 'body', 'keywords', or 'context'."
        )

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# [3] predict_fn  —  Logic gate. Run the classifier.
# ─────────────────────────────────────────────────────────────────────────────

def predict_fn(data: dict, model: dict) -> dict:
    """
    The main classification step. Runs on every request.

    Step 1: Extract features from raw text  (or use pre-supplied features)
    Step 2: Classify → 3 integer codes      (urgency, routing, resolution)
    Step 3: Decode codes → human labels     ("Security Breach", "Escalate", ...)
    Step 4: Score against ground_truth      (only if ground_truth is in request)
    Step 5: Return everything as a dict     (output_fn will format it as JSON)
    """
    logger.info("predict_fn | keys=%s", list(data.keys()))

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    # Fast path: caller already extracted features
    if data.get("context"):
        features = {
            "keywords":  data.get("keywords", []),
            "sentiment": data.get("sentiment", "neutral"),
            "context":   data["context"],
        }
    # NLP path: extract from raw subject + body text
    else:
        features = extract_features(
            subject=data.get("subject", ""),
            body=data.get("body", ""),
        )

    # ── Step 2: Classify → 3 codes ────────────────────────────────────────────
    urgency, routing, resolution = classify(features)

    # ── Step 3: Decode to human-readable labels ───────────────────────────────
    labels = decode(urgency, routing, resolution)

    logger.info(
        "CLASSIFIED | category=%s dept=%s action=%s | context=%s keywords=%s",
        labels["urgency"], labels["routing"], labels["resolution"],
        features["context"], features["keywords"],
    )

    # ── Step 4: Score against ground_truth (optional) ─────────────────────────
    ground_truth = data.get("ground_truth")
    if ground_truth:
        match = _score_match((urgency, routing, resolution), ground_truth)
    else:
        # No ground_truth supplied — this is a live production request
        match = {"status": "UNVERIFIED", "reward": None,
                 "correct_dims": {}, "wrong_fields": []}

    # ── Step 5: Return raw prediction dict ────────────────────────────────────
    return {
        "urgency_code":    urgency,
        "routing_code":    routing,
        "resolution_code": resolution,
        "labels":          labels,
        "features":        features,
        "match":           match,
        "sla":             model["sla"][urgency],
        "endpoint":        model["endpoint_name"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# [4] output_fn  —  Output pin. Format and send the response.
# ─────────────────────────────────────────────────────────────────────────────

def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    """
    Converts the prediction dict into the final HTTP response body.

    Default response format: application/json
    Optional CSV format:     text/csv  (useful for batch jobs writing to S3)

    JSON response shape:
    {
        "request_id":   "uuid",
        "timestamp":    "2024-01-15T10:30:00Z",

        "triage": {
            "category":   "Security Breach",   ← urgency label
            "department": "Tech Support",       ← routing label
            "action":     "Escalate"            ← resolution label
        },

        "codes": {
            "urgency": 2, "routing": 1, "resolution": 2
        },

        "match_result": {
            "status":  "EXACT",     ← or PARTIAL_1 / PARTIAL_2 / SECURITY_MISS / WRONG
            "reward":  1.0,         ← RL reward score
            "wrong_fields": []      ← which dimensions were wrong
        },

        "sla": {
            "priority": "P1",
            "respond_within_minutes": 15
        }
    }
    """
    accept_type = (accept or "application/json").lower().split(";")[0].strip()

    response = {
        "request_id": str(uuid.uuid4()),
        "timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "triage": {
            "category":   prediction["labels"]["urgency"],
            "department": prediction["labels"]["routing"],
            "action":     prediction["labels"]["resolution"],
        },
        "codes": {
            "urgency":    prediction["urgency_code"],
            "routing":    prediction["routing_code"],
            "resolution": prediction["resolution_code"],
        },
        "features": {
            "keywords":  prediction["features"]["keywords"],
            "sentiment": prediction["features"]["sentiment"],
            "context":   prediction["features"]["context"],
        },
        "match_result": {
            "status":       prediction["match"]["status"],
            "reward":       prediction["match"]["reward"],
            "wrong_fields": prediction["match"]["wrong_fields"],
        },
        "sla": prediction["sla"],
    }

    # ── CSV output (for SageMaker Batch Transform jobs) ───────────────────────
    if accept_type == "text/csv":
        row = ",".join([
            response["request_id"],
            response["triage"]["category"],
            response["triage"]["department"],
            response["triage"]["action"],
            str(response["codes"]["urgency"]),
            str(response["codes"]["routing"]),
            str(response["codes"]["resolution"]),
            str(response["match_result"]["status"]),
            str(response["match_result"]["reward"] or ""),
            response["sla"]["priority"],
        ])
        return row, "text/csv"

    return json.dumps(response, ensure_ascii=False), "application/json"
