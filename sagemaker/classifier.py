"""
classifier.py — Production Rule-Based Email Classifier
=======================================================
Shared by SageMaker inference.py and Lambda handler.py.
Zero heavy dependencies — no numpy, no gymnasium.

Key fix vs lambda/classifier.py:
  "legal" removed from _LEGAL_SECURITY_KW — it is a deception keyword
  in phishing emails (TC-H-09), not a routing signal. Context field
  is the authoritative source for legal routing.
"""

# ── Label maps ────────────────────────────────────────────────────────────────
URGENCY_LABELS    = {0: "General",       1: "Billing",      2: "Security Breach"}
ROUTING_LABELS    = {0: "AI Auto-Reply", 1: "Tech Support", 2: "Legal"}
RESOLUTION_LABELS = {0: "Archive",       1: "Draft Reply",  2: "Escalate"}

# Security emails that need Legal routing (ransomware / extortion / IP theft).
# NOTE: "legal" intentionally excluded — it appears in phishing deception text.
_LEGAL_SECURITY_KW   = {"lawsuit", "attorney", "sue", "ransomware", "extortion"}

# Only "refund" escalates billing to Legal — "overdue" stays routine.
_BILLING_ESCALATE_KW = {"refund"}

# Canonical keyword vocabulary (must match environment.py KEYWORD_VOCAB)
KEYWORD_VOCAB = [
    "invoice",  "payment",      "overdue",  "refund",
    "hacked",   "breach",       "unauthorized", "password",
    "crash",    "error",        "bug",      "slow",
    "lawsuit",  "legal",        "attorney", "sue",
    "spam",     "offer",        "win",      "free",
    "urgent",   "critical",     "angry",    "threat",
]

# Words used for sentiment scoring
_NEG_WORDS = {
    "angry", "threat", "hacked", "breach", "lawsuit", "overdue",
    "unauthorized", "ransomware", "critical", "urgent", "error",
    "crash", "bug", "refund",
}
_POS_WORDS = {"win", "free", "offer", "congratulations", "prize"}


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(subject: str, body: str) -> dict:
    """
    Parse raw email text → feature dict {keywords, sentiment, context}.
    Used when the caller does not supply pre-computed features.
    """
    text   = (subject + " " + body).lower()
    tokens = set(text.split())

    keywords = [kw for kw in KEYWORD_VOCAB if kw in tokens]
    kw_set   = set(keywords)

    # Sentiment
    neg_hits = len(tokens & _NEG_WORDS)
    pos_hits = len(tokens & _POS_WORDS)
    if neg_hits > pos_hits:
        sentiment = "negative"
    elif pos_hits > 0:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Context — priority order matches the classifier decision tree
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


# ── Classifier ────────────────────────────────────────────────────────────────

def classify(email: dict) -> tuple[int, int, int]:
    """
    Deterministic rule-based classifier.
    Returns (urgency, routing, resolution) as plain ints.

    Decision tree — first match wins:
      Rule 1  legal context OR lawsuit/attorney/sue keywords → (2, 2, 2)
      Rule 2a security + ransomware/extortion/hacked+breach  → (2, 2, 2)
      Rule 2b security (account-level attack)                → (2, 1, 2)
      Rule 3  billing + refund keyword                       → (1, 2, 2)
      Rule 4  billing routine                                → (1, 0, 1)
      Rule 5  tech context or crash/error/bug/slow           → (0, 1, 1)
      Rule 6  spam / default                                 → (0, 0, 0)
    """
    kw      = set(email.get("keywords", []))
    context = email.get("context", "").lower()

    # Rule 1 — Legal
    if context == "legal" or kw & {"lawsuit", "attorney", "sue"}:
        return (2, 2, 2)

    # Rule 2 — Security
    if context == "security":
        if kw & _LEGAL_SECURITY_KW or ("hacked" in kw and "breach" in kw):
            return (2, 2, 2)   # ransomware / extortion → Legal
        return (2, 1, 2)       # account-level attack   → Tech Support

    # Rule 3 & 4 — Billing
    if context == "billing":
        return (1, 2, 2) if kw & _BILLING_ESCALATE_KW else (1, 0, 1)

    # Rule 5 — Tech
    if context == "tech" or kw & {"crash", "error", "bug", "slow"}:
        return (0, 1, 1)

    # Rule 6 — Spam / default
    return (0, 0, 0)


# ── Decoder ───────────────────────────────────────────────────────────────────

def decode(urgency: int, routing: int, resolution: int) -> dict:
    """Convert integer action codes to human-readable label dict."""
    return {
        "urgency":    URGENCY_LABELS[urgency],
        "routing":    ROUTING_LABELS[routing],
        "resolution": RESOLUTION_LABELS[resolution],
    }


# ── Batch helper ─────────────────────────────────────────────────────────────

def classify_batch(emails: list[dict]) -> list[dict]:
    """
    Classify a list of email dicts in one call.
    Each dict may contain pre-computed features OR raw subject+body.
    Returns a list of decode() dicts with codes attached.
    """
    results = []
    for email in emails:
        if not email.get("context"):
            features = extract_features(
                email.get("subject", ""),
                email.get("body", ""),
            )
        else:
            features = email

        u, r, res = classify(features)
        result    = decode(u, r, res)
        result.update({"urgency_code": u, "routing_code": r, "resolution_code": res})
        results.append(result)
    return results
