"""
classifier.py — Portable rule-based email classifier for AWS Lambda.
Extracted from inference.py with zero heavy dependencies (no numpy/gymnasium).
All logic is identical to the local rule engine.
"""

# ── Label maps (mirrors environment.py) ─────────────────────────────────────
URGENCY_LABELS    = {0: "General",       1: "Billing",      2: "Security Breach"}
ROUTING_LABELS    = {0: "AI Auto-Reply", 1: "Tech Support", 2: "Legal"}
RESOLUTION_LABELS = {0: "Archive",       1: "Draft Reply",  2: "Escalate"}

# Keywords that push a security email to Legal (ransomware / extortion level).
_LEGAL_SECURITY_KW  = {"lawsuit", "attorney", "sue", "ransomware", "extortion", "legal"}

# Only "refund" escalates a billing email to Legal — "overdue" stays routine.
_BILLING_ESCALATE_KW = {"refund"}

# Full keyword vocabulary used for feature extraction from raw email text.
KEYWORD_VOCAB = [
    "invoice", "payment", "overdue", "refund",
    "hacked", "breach", "unauthorized", "password",
    "crash", "error", "bug", "slow",
    "lawsuit", "legal", "attorney", "sue",
    "spam", "offer", "win", "free",
    "urgent", "critical", "angry", "threat",
]


def extract_features(subject: str, body: str) -> dict:
    """
    Parse raw email text into the feature dict expected by classify().
    Returns: {keywords, sentiment, context}
    """
    text = (subject + " " + body).lower()
    tokens = set(text.split())

    keywords = [kw for kw in KEYWORD_VOCAB if kw in tokens]

    # Simple sentiment: negative words outweigh positive
    neg_words = {"angry", "threat", "hacked", "breach", "lawsuit", "overdue",
                 "unauthorized", "ransomware", "critical", "urgent", "error",
                 "crash", "bug", "refund"}
    pos_words = {"win", "free", "offer", "congratulations", "prize"}

    neg_hits = len(tokens & neg_words)
    pos_hits = len(tokens & pos_words)

    if neg_hits > pos_hits:
        sentiment = "negative"
    elif pos_hits > 0:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Context: first strong signal wins
    kw_set = set(keywords)
    if kw_set & {"hacked", "breach", "unauthorized", "ransomware"}:
        context = "security"
    elif kw_set & {"lawsuit", "attorney", "sue"}:
        context = "legal"
    elif kw_set & {"invoice", "payment", "overdue", "refund"}:
        context = "billing"
    elif kw_set & {"crash", "error", "bug", "slow", "password"}:
        context = "tech"
    elif kw_set & {"spam", "offer", "win", "free"}:
        context = "spam"
    else:
        context = "general"

    return {"keywords": keywords, "sentiment": sentiment, "context": context}


def classify(email: dict) -> tuple[int, int, int]:
    """
    Rule-based classifier. Accepts a feature dict (keywords, context).
    Returns (urgency, routing, resolution) as plain ints.

    Priority order (first match wins):
      1. Legal context / legal keywords  → (2, 2, 2)
      2. Security + legal signal         → (2, 2, 2)
      2. Security account-level          → (2, 1, 2)
      3. Billing dispute (refund)        → (1, 2, 2)
      4. Billing routine                 → (1, 0, 1)
      5. Tech support                    → (0, 1, 1)
      6. Spam / default                  → (0, 0, 0)
    """
    kw      = set(email.get("keywords", []))
    context = email.get("context", "").lower()

    if context == "legal" or kw & {"lawsuit", "attorney", "sue"}:
        return (2, 2, 2)

    if context == "security":
        if kw & _LEGAL_SECURITY_KW or ("hacked" in kw and "breach" in kw):
            return (2, 2, 2)
        return (2, 1, 2)

    if context == "billing":
        if kw & _BILLING_ESCALATE_KW:
            return (1, 2, 2)
        return (1, 0, 1)

    if context == "tech" or kw & {"crash", "error", "bug", "slow"}:
        return (0, 1, 1)

    return (0, 0, 0)


def decode(urgency: int, routing: int, resolution: int) -> dict:
    """Convert integer action tuple to human-readable label dict."""
    return {
        "urgency_code":    urgency,
        "routing_code":    routing,
        "resolution_code": resolution,
        "urgency":         URGENCY_LABELS[urgency],
        "routing":         ROUTING_LABELS[routing],
        "resolution":      RESOLUTION_LABELS[resolution],
    }
