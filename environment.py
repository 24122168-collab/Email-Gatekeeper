"""
EmailTriageEnv — Intelligent Email Gatekeeper RL Environment
============================================================
Observation : flat Box vector encoding keywords, sentiment, and context.
Action      : MultiDiscrete([3, 3, 3])
              [0] Urgency   — 0=General, 1=Billing, 2=Security Breach
              [1] Routing   — 0=AI Auto-Reply, 1=Tech Support, 2=Legal
              [2] Resolution— 0=Archive, 1=Draft Reply, 2=Escalate to Human
Reward      : +0.5 fully correct | -0.4 wrong priority on crisis email
              Cumulative ideal score over one episode ≈ 1.0 (normalised).
Episode     : one email per step; done after all emails in the batch.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Vocabulary & encoding helpers
# ---------------------------------------------------------------------------

# Fixed keyword vocabulary — extend freely.
KEYWORD_VOCAB = [
    "invoice", "payment", "overdue", "refund",          # billing
    "hacked", "breach", "unauthorized", "password",     # security
    "crash", "error", "bug", "slow",                    # tech
    "lawsuit", "legal", "attorney", "sue",              # legal
    "spam", "offer", "win", "free",                     # spam
    "urgent", "critical", "angry", "threat",            # sentiment signals
]

# Sentiment classes  → index
SENTIMENT_MAP = {"positive": 0, "neutral": 1, "negative": 2}

# Context classes    → index
CONTEXT_MAP = {"spam": 0, "billing": 1, "tech": 2, "security": 3, "legal": 4}

# Observation vector length: keyword flags + one-hot sentiment + one-hot context
OBS_DIM = len(KEYWORD_VOCAB) + len(SENTIMENT_MAP) + len(CONTEXT_MAP)

# ---------------------------------------------------------------------------
# Mock email dataset
# ---------------------------------------------------------------------------
# Each entry:
#   keywords    : list[str]  — subset of KEYWORD_VOCAB
#   sentiment   : str        — key in SENTIMENT_MAP
#   context     : str        — key in CONTEXT_MAP
#   difficulty  : str        — "easy" | "medium" | "hard"
#   correct_actions: tuple   — (urgency, routing, resolution)
#   description : str        — human-readable label (for debugging)

EMAIL_DATASET = [
    # ── Easy: Spam vs Real ──────────────────────────────────────────────────
    {
        "description": "Spam promo",
        "keywords": ["spam", "offer", "win", "free"],
        "sentiment": "positive",
        "context": "spam",
        "difficulty": "easy",
        "correct_actions": (0, 0, 0),   # General | AI Auto-Reply | Archive
    },
    {
        "description": "Spam lottery",
        "keywords": ["free", "win", "offer"],
        "sentiment": "positive",
        "context": "spam",
        "difficulty": "easy",
        "correct_actions": (0, 0, 0),
    },
    {
        "description": "Routine support request",
        "keywords": ["slow", "error"],
        "sentiment": "neutral",
        "context": "tech",
        "difficulty": "easy",
        "correct_actions": (0, 1, 1),   # General | Tech Support | Draft Reply
    },
    {
        "description": "General billing inquiry",
        "keywords": ["invoice", "payment"],
        "sentiment": "neutral",
        "context": "billing",
        "difficulty": "easy",
        "correct_actions": (1, 0, 1),   # Billing | AI Auto-Reply | Draft Reply
    },
    # ── Medium: Billing / Tech context ──────────────────────────────────────
    {
        "description": "Overdue invoice complaint",
        "keywords": ["invoice", "overdue", "payment", "angry"],
        "sentiment": "negative",
        "context": "billing",
        "difficulty": "medium",
        "correct_actions": (1, 0, 1),   # Billing | AI Auto-Reply | Draft Reply
    },
    {
        "description": "Refund dispute",
        "keywords": ["refund", "payment", "angry"],
        "sentiment": "negative",
        "context": "billing",
        "difficulty": "medium",
        "correct_actions": (1, 2, 2),   # Billing | Legal | Escalate
    },
    {
        "description": "App crash report",
        "keywords": ["crash", "bug", "error"],
        "sentiment": "negative",
        "context": "tech",
        "difficulty": "medium",
        "correct_actions": (0, 1, 1),   # General | Tech Support | Draft Reply
    },
    {
        "description": "Persistent login bug",
        "keywords": ["bug", "password", "error"],
        "sentiment": "negative",
        "context": "tech",
        "difficulty": "medium",
        "correct_actions": (0, 1, 1),
    },
    # ── Medium: Passive-Aggressive Legal Threats ────────────────────────────
    {
        "id": "TC-M-01",
        "description": "Polite refund ultimatum with implied legal action",
        "keywords": ["refund", "legal", "angry", "threat"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-02",
        "description": "Overdue invoice with attorney CC warning",
        "keywords": ["invoice", "overdue", "attorney", "legal", "payment", "threat"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-03",
        "description": "Disappointed customer implying regulatory complaint",
        "keywords": ["angry", "threat", "legal"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-04",
        "description": "SLA breach complaint with legal remedy language",
        "keywords": ["breach", "legal", "threat", "angry"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-05",
        "description": "Billing dispute with small claims court mention",
        "keywords": ["refund", "payment", "angry", "lawsuit", "urgent"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-06",
        "description": "GDPR deletion request with legal consequence warning",
        "keywords": ["legal", "breach", "threat", "angry"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-07",
        "description": "Subscription cancellation with chargeback and CFPB threat",
        "keywords": ["payment", "refund", "angry", "threat", "legal"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-M-08",
        "description": "Vendor threatening IP infringement claim",
        "keywords": ["unauthorized", "legal", "attorney", "threat"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "medium",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    # ── Hard: Subtle Phishing Attempts ──────────────────────────────────────
    {
        "id": "TC-H-01",
        "description": "IT password reset disguised as routine security audit",
        "keywords": ["password", "unauthorized", "critical", "urgent", "threat"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-02",
        "description": "Fake invoice payment portal redirect — credential harvest",
        "keywords": ["invoice", "payment", "password", "unauthorized", "urgent"],
        "sentiment": "neutral",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-03",
        "description": "HR benefits enrollment with credential capture",
        "keywords": ["password", "urgent", "critical"],
        "sentiment": "neutral",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-04",
        "description": "Fake account suspension notice with login link",
        "keywords": ["unauthorized", "password", "breach", "urgent", "threat"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-05",
        "description": "Vendor onboarding BEC — admin credentials via reply",
        "keywords": ["password", "unauthorized", "urgent"],
        "sentiment": "neutral",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-06",
        "description": "Fake new sign-in alert — was this you? phish",
        "keywords": ["unauthorized", "password", "hacked", "breach", "urgent"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-07",
        "description": "Payroll migration phish — salary interruption fear",
        "keywords": ["payment", "password", "urgent", "threat"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "id": "TC-H-08",
        "description": "Software license renewal — admin credential request",
        "keywords": ["password", "critical", "urgent", "error"],
        "sentiment": "neutral",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    # ── Hard: Phishing + Legal Threat Overlay ───────────────────────────────
    {
        "id": "TC-H-09",
        "description": "Fake GDPR breach notice — credential harvest via legal fear",
        "keywords": ["breach", "hacked", "password", "legal", "threat", "urgent", "unauthorized"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
        # NOTE: legal language is the deception layer; credential harvest = Tech Support
    },
    {
        "id": "TC-H-10",
        "description": "Ransomware disguised as software compliance audit",
        "keywords": ["hacked", "breach", "unauthorized", "lawsuit", "legal", "threat", "critical", "urgent"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-H-11",
        "description": "Extortion — threatening to publish stolen data",
        "keywords": ["hacked", "breach", "unauthorized", "attorney", "threat", "critical", "urgent"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "id": "TC-H-12",
        "description": "Fake law firm letter claiming evidence of data misuse",
        "keywords": ["unauthorized", "breach", "attorney", "lawsuit", "legal", "threat"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "hard",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    # ── Hard: Crisis / Security threats ─────────────────────────────────────
    {
        "description": "Account hacked — urgent",
        "keywords": ["hacked", "unauthorized", "password", "urgent", "angry"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),   # Security | Tech Support | Escalate
    },
    {
        "description": "Data breach notification",
        "keywords": ["breach", "unauthorized", "critical", "threat"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 1, 2),
    },
    {
        "description": "Legal threat — lawsuit",
        "keywords": ["lawsuit", "legal", "attorney", "threat", "angry"],
        "sentiment": "negative",
        "context": "legal",
        "difficulty": "hard",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
    {
        "description": "Ransomware / extortion threat",
        "keywords": ["hacked", "threat", "critical", "urgent", "breach"],
        "sentiment": "negative",
        "context": "security",
        "difficulty": "hard",
        "correct_actions": (2, 2, 2),   # Security | Legal | Escalate
    },
]

# ---------------------------------------------------------------------------
# Reward weights — adjust freely
# ---------------------------------------------------------------------------
REWARD_CORRECT_FULL      =  1.0   # teeno actions bilkul sahi
REWARD_PARTIAL_ONE_WRONG =  0.2   # urgency sahi + sirf ek dimension galat
REWARD_PARTIAL_BOTH_WRONG=  0.1   # urgency sahi lekin routing aur resolution dono galat
PENALTY_MISSED_SECURITY  = -2.0   # security breach email miss kiya


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnv(gym.Env):
    """
    Single-email-per-step triage environment.

    Parameters
    ----------
    batch : list[dict] | None
        Custom email batch. Defaults to the full EMAIL_DATASET.
    shuffle : bool
        Shuffle the batch on each reset (default True).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, batch: list | None = None, shuffle: bool = True):
        super().__init__()

        self.email_batch = batch if batch is not None else EMAIL_DATASET
        self.shuffle = shuffle

        # Action space: [urgency(3), routing(3), resolution(3)]
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # Observation space: binary keyword flags + one-hot sentiment + one-hot context
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Internal state (populated by reset)
        self._queue: list[dict] = []
        self._current_email: dict = {}
        self._step_idx: int = 0

        # Normalisation constant: max possible reward per episode
        self._max_episode_reward = len(self.email_batch) * REWARD_CORRECT_FULL

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, email: dict) -> np.ndarray:
        """Convert an email dict into a flat float32 observation vector."""
        # Keyword flags (binary)
        kw_flags = np.array(
            [1.0 if kw in email["keywords"] else 0.0 for kw in KEYWORD_VOCAB],
            dtype=np.float32,
        )
        # One-hot sentiment
        sentiment_vec = np.zeros(len(SENTIMENT_MAP), dtype=np.float32)
        sentiment_vec[SENTIMENT_MAP[email["sentiment"]]] = 1.0

        # One-hot context
        context_vec = np.zeros(len(CONTEXT_MAP), dtype=np.float32)
        context_vec[CONTEXT_MAP[email["context"]]] = 1.0

        return np.concatenate([kw_flags, sentiment_vec, context_vec])

    def _compute_reward(self, action: np.ndarray, email: dict) -> float:
        """
        Strict reward rules (priority order mein check hote hain):
          -2.0  — security breach email ko urgency=2 nahi diya (sabse bada penalty)
          +1.0  — teeno actions bilkul sahi (exact match)
          +0.2  — sirf urgency sahi, routing AUR resolution dono sahi (2 out of 3)
          +0.1  — sirf urgency sahi, baaki ek galat (1 dimension wrong)
           0.0  — urgency hi galat hai (non-security email)
        """
        urgency    = int(action[0])
        routing    = int(action[1])
        resolution = int(action[2])
        correct    = email["correct_actions"]

        # Priority 1: Security breach miss — sabse bada crime
        if correct[0] == 2 and urgency != 2:
            return PENALTY_MISSED_SECURITY

        # Priority 2: Perfect match
        if (urgency, routing, resolution) == correct:
            return REWARD_CORRECT_FULL

        # Priority 3: Urgency sahi hai — partial credit
        if urgency == correct[0]:
            # Dono routing aur resolution galat hain
            routing_ok    = (routing    == correct[1])
            resolution_ok = (resolution == correct[2])
            if routing_ok and not resolution_ok:
                return REWARD_PARTIAL_ONE_WRONG   # sirf resolution galat
            if resolution_ok and not routing_ok:
                return REWARD_PARTIAL_ONE_WRONG   # sirf routing galat
            return REWARD_PARTIAL_BOTH_WRONG      # dono galat

        return 0.0

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._queue = list(self.email_batch)
        if self.shuffle:
            self.np_random.shuffle(self._queue)  # uses gymnasium's seeded RNG

        self._step_idx = 0
        self._current_email = self._queue[self._step_idx]

        obs = self._encode(self._current_email)
        info = {"description": self._current_email["description"],
                "difficulty": self._current_email["difficulty"]}
        return obs, info

    def step(self, action: np.ndarray):
        """
        Process one email triage decision.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # ── Bug Fix: correct_actions PEHLE save karo, PHIR pointer badlao ──
        # Pehle current email ka ground truth capture karo reward ke saath
        scored_email    = self._current_email
        reward          = self._compute_reward(action, scored_email)
        normalised_reward = reward / self._max_episode_reward

        # Ab pointer aage badhao
        self._step_idx += 1
        terminated = self._step_idx >= len(self._queue)

        if not terminated:
            self._current_email = self._queue[self._step_idx]
            obs = self._encode(self._current_email)
        else:
            obs = self._encode(scored_email)  # terminal step pe last obs return karo

        # info mein SCORED email ka data — agli email ka nahi
        info = {
            "raw_reward":      reward,
            "correct_actions": scored_email["correct_actions"],
            "difficulty":      scored_email["difficulty"],
            "description":     scored_email.get("description", ""),
        }
        return obs, normalised_reward, terminated, False, info

    def render(self, mode: str = "human"):
        """Print current email details to stdout."""
        e = self._current_email
        print(f"[Step {self._step_idx}] {e['description']} "
              f"| difficulty={e['difficulty']} "
              f"| sentiment={e['sentiment']} "
              f"| context={e['context']}")
