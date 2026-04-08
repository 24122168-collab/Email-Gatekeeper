"""
env.py — Email Gatekeeper RL Environment (OpenEnv Specification)
================================================================
Gymnasium environment for intelligent email triage.
Wraps the core EmailTriageEnv logic with:
  - Pydantic typed Action and Observation models
  - state() method returning current environment state
  - Three task splits: easy / medium / hard
  - Full OpenEnv-compatible interface
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel, Field
from typing import Optional

# ── Vocabulary & encoding (canonical — must not change between versions) ──────

KEYWORD_VOCAB = [
    "invoice", "payment", "overdue", "refund",
    "hacked",  "breach",  "unauthorized", "password",
    "crash",   "error",   "bug",   "slow",
    "lawsuit", "legal",   "attorney", "sue",
    "spam",    "offer",   "win",   "free",
    "urgent",  "critical","angry", "threat",
]

SENTIMENT_MAP = {"positive": 0, "neutral": 1, "negative": 2}
CONTEXT_MAP   = {"spam": 0, "billing": 1, "tech": 2, "security": 3, "legal": 4}
OBS_DIM       = len(KEYWORD_VOCAB) + len(SENTIMENT_MAP) + len(CONTEXT_MAP)  # 32

# ── Label maps ────────────────────────────────────────────────────────────────
URGENCY_LABELS    = {0: "General",       1: "Billing",      2: "Security Breach"}
ROUTING_LABELS    = {0: "AI Auto-Reply", 1: "Tech Support", 2: "Legal"}
RESOLUTION_LABELS = {0: "Archive",       1: "Draft Reply",  2: "Escalate"}

# ── Reward weights ────────────────────────────────────────────────────────────
REWARD_EXACT           =  1.0
REWARD_PARTIAL_1_WRONG =  0.2
REWARD_PARTIAL_2_WRONG =  0.1
PENALTY_SECURITY_MISS  = -2.0


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Typed Models
# ─────────────────────────────────────────────────────────────────────────────

class EmailAction(BaseModel):
    """
    The agent's triage decision for one email.
    All three dimensions must be predicted simultaneously.
    """
    urgency: int = Field(
        ..., ge=0, le=2,
        description="0=General | 1=Billing | 2=Security Breach"
    )
    routing: int = Field(
        ..., ge=0, le=2,
        description="0=AI Auto-Reply | 1=Tech Support | 2=Legal"
    )
    resolution: int = Field(
        ..., ge=0, le=2,
        description="0=Archive | 1=Draft Reply | 2=Escalate"
    )

    def to_array(self) -> np.ndarray:
        return np.array([self.urgency, self.routing, self.resolution],
                        dtype=np.int64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EmailAction":
        return cls(urgency=int(arr[0]), routing=int(arr[1]),
                   resolution=int(arr[2]))


class EmailObservation(BaseModel):
    """
    The agent's view of the current email.
    Encoded as a flat float32 vector of length 32.
    """
    keyword_flags: list[float] = Field(
        ..., description=f"Binary flags for {len(KEYWORD_VOCAB)} vocab keywords"
    )
    sentiment_onehot: list[float] = Field(
        ..., description="One-hot: [positive, neutral, negative]"
    )
    context_onehot: list[float] = Field(
        ..., description="One-hot: [spam, billing, tech, security, legal]"
    )
    # Human-readable metadata (not used by the agent, useful for logging)
    description: str = ""
    difficulty:  str = ""
    context_str: str = ""
    sentiment_str: str = ""
    keywords:    list[str] = Field(default_factory=list)

    def to_array(self) -> np.ndarray:
        return np.array(
            self.keyword_flags + self.sentiment_onehot + self.context_onehot,
            dtype=np.float32,
        )


class EnvironmentState(BaseModel):
    """Current snapshot of the environment — returned by state()."""
    step_index:    int
    total_emails:  int
    emails_remaining: int
    current_email: dict
    cumulative_reward: float
    task:          str   # "easy" | "medium" | "hard" | "all"
    terminated:    bool


class StepResult(BaseModel):
    """Typed return value from step()."""
    observation:  EmailObservation
    reward:       float
    normalised_reward: float
    terminated:   bool
    truncated:    bool
    info:         dict


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_DATASET: list[dict] = [
    # ── Easy: Spam detection ─────────────────────────────────────────────────
    {"description": "Spam promo",          "keywords": ["spam","offer","win","free"],
     "sentiment": "positive", "context": "spam",    "difficulty": "easy",
     "correct_actions": (0, 0, 0)},
    {"description": "Spam lottery",        "keywords": ["free","win","offer"],
     "sentiment": "positive", "context": "spam",    "difficulty": "easy",
     "correct_actions": (0, 0, 0)},
    {"description": "Routine support",     "keywords": ["slow","error"],
     "sentiment": "neutral",  "context": "tech",    "difficulty": "easy",
     "correct_actions": (0, 1, 1)},
    {"description": "General billing",     "keywords": ["invoice","payment"],
     "sentiment": "neutral",  "context": "billing", "difficulty": "easy",
     "correct_actions": (1, 0, 1)},
    # ── Medium: Support routing ───────────────────────────────────────────────
    {"description": "Overdue invoice",     "keywords": ["invoice","overdue","payment","angry"],
     "sentiment": "negative", "context": "billing", "difficulty": "medium",
     "correct_actions": (1, 0, 1)},
    {"description": "Refund dispute",      "keywords": ["refund","payment","angry"],
     "sentiment": "negative", "context": "billing", "difficulty": "medium",
     "correct_actions": (1, 2, 2)},
    {"description": "App crash report",    "keywords": ["crash","bug","error"],
     "sentiment": "negative", "context": "tech",    "difficulty": "medium",
     "correct_actions": (0, 1, 1)},
    {"description": "Persistent login bug","keywords": ["bug","password","error"],
     "sentiment": "negative", "context": "tech",    "difficulty": "medium",
     "correct_actions": (0, 1, 1)},
    {"description": "Polite legal ultimatum","keywords": ["refund","legal","angry","threat"],
     "sentiment": "negative", "context": "legal",   "difficulty": "medium",
     "correct_actions": (2, 2, 2)},
    {"description": "Attorney CC warning", "keywords": ["invoice","overdue","attorney","legal","payment","threat"],
     "sentiment": "negative", "context": "legal",   "difficulty": "medium",
     "correct_actions": (2, 2, 2)},
    {"description": "Regulatory complaint","keywords": ["angry","threat","legal"],
     "sentiment": "negative", "context": "legal",   "difficulty": "medium",
     "correct_actions": (2, 2, 2)},
    {"description": "SLA breach legal",    "keywords": ["breach","legal","threat","angry"],
     "sentiment": "negative", "context": "legal",   "difficulty": "medium",
     "correct_actions": (2, 2, 2)},
    # ── Hard: Phishing & security threats ────────────────────────────────────
    {"description": "IT audit phish",      "keywords": ["password","unauthorized","critical","urgent","threat"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Fake invoice portal", "keywords": ["invoice","payment","password","unauthorized","urgent"],
     "sentiment": "neutral",  "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "HR credential phish", "keywords": ["password","urgent","critical"],
     "sentiment": "neutral",  "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Fake suspension",     "keywords": ["unauthorized","password","breach","urgent","threat"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "BEC vendor reply",    "keywords": ["password","unauthorized","urgent"],
     "sentiment": "neutral",  "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Sign-in alert phish", "keywords": ["unauthorized","password","hacked","breach","urgent"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Payroll phish",       "keywords": ["payment","password","urgent","threat"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "License renewal BEC", "keywords": ["password","critical","urgent","error"],
     "sentiment": "neutral",  "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "GDPR phish",          "keywords": ["breach","hacked","password","legal","threat","urgent","unauthorized"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Ransomware audit",    "keywords": ["hacked","breach","unauthorized","lawsuit","legal","threat","critical","urgent"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 2, 2)},
    {"description": "Data extortion",      "keywords": ["hacked","breach","unauthorized","attorney","threat","critical","urgent"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 2, 2)},
    {"description": "Fake law firm",       "keywords": ["unauthorized","breach","attorney","lawsuit","legal","threat"],
     "sentiment": "negative", "context": "legal",   "difficulty": "hard",
     "correct_actions": (2, 2, 2)},
    {"description": "Account hacked",      "keywords": ["hacked","unauthorized","password","urgent","angry"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Data breach notice",  "keywords": ["breach","unauthorized","critical","threat"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 1, 2)},
    {"description": "Legal lawsuit threat","keywords": ["lawsuit","legal","attorney","threat","angry"],
     "sentiment": "negative", "context": "legal",   "difficulty": "hard",
     "correct_actions": (2, 2, 2)},
    {"description": "Ransomware threat",   "keywords": ["hacked","threat","critical","urgent","breach"],
     "sentiment": "negative", "context": "security","difficulty": "hard",
     "correct_actions": (2, 2, 2)},
]

# Task splits — used by inference.py for per-task scoring
TASK_SPLITS: dict[str, list[dict]] = {
    "easy":   [e for e in EMAIL_DATASET if e["difficulty"] == "easy"],
    "medium": [e for e in EMAIL_DATASET if e["difficulty"] == "medium"],
    "hard":   [e for e in EMAIL_DATASET if e["difficulty"] == "hard"],
    "all":    EMAIL_DATASET,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Environment
# ─────────────────────────────────────────────────────────────────────────────

class EmailTriageEnv(gym.Env):
    """
    OpenEnv-compliant Gymnasium environment for email triage.

    The agent receives one email per step as a 32-dim observation vector
    and must output three simultaneous discrete decisions.

    Parameters
    ----------
    task : str
        "easy" | "medium" | "hard" | "all"  — which email subset to use.
    shuffle : bool
        Shuffle emails on each reset (default True).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, task: str = "all", shuffle: bool = True):
        super().__init__()

        if task not in TASK_SPLITS:
            raise ValueError(f"task must be one of {list(TASK_SPLITS)}. Got '{task}'.")

        self.task          = task
        self.shuffle       = shuffle
        self.email_batch   = TASK_SPLITS[task]

        # Gymnasium spaces
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Internal state
        self._queue:          list[dict] = []
        self._current_email:  dict       = {}
        self._step_idx:       int        = 0
        self._cumulative_reward: float   = 0.0
        self._max_episode_reward: float  = len(self.email_batch) * REWARD_EXACT

    # ── Encoding helpers ──────────────────────────────────────────────────────

    def _encode_to_obs(self, email: dict) -> EmailObservation:
        """Convert an email dict into a typed EmailObservation."""
        kw_flags = [1.0 if kw in email["keywords"] else 0.0
                    for kw in KEYWORD_VOCAB]

        sentiment_vec = [0.0] * len(SENTIMENT_MAP)
        sentiment_vec[SENTIMENT_MAP[email["sentiment"]]] = 1.0

        context_vec = [0.0] * len(CONTEXT_MAP)
        context_vec[CONTEXT_MAP[email["context"]]] = 1.0

        return EmailObservation(
            keyword_flags=kw_flags,
            sentiment_onehot=sentiment_vec,
            context_onehot=context_vec,
            description=email.get("description", ""),
            difficulty=email.get("difficulty", ""),
            context_str=email["context"],
            sentiment_str=email["sentiment"],
            keywords=email["keywords"],
        )

    def _compute_reward(self, action: np.ndarray, email: dict) -> float:
        """
        Reward function — same logic as environment.py, priority order:
          1. Security miss  → -2.0  (correct urgency=2, predicted otherwise)
          2. Exact match    → +1.0
          3. Partial-1      → +0.2  (urgency correct, 1 other wrong)
          4. Partial-2      → +0.1  (urgency correct, both others wrong)
          5. Wrong          →  0.0
        """
        u, r, res   = int(action[0]), int(action[1]), int(action[2])
        c           = email["correct_actions"]

        if c[0] == 2 and u != 2:
            return PENALTY_SECURITY_MISS
        if (u, r, res) == c:
            return REWARD_EXACT
        if u == c[0]:
            wrong = sum([r != c[1], res != c[2]])
            return REWARD_PARTIAL_1_WRONG if wrong == 1 else REWARD_PARTIAL_2_WRONG
        return 0.0

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._queue = list(self.email_batch)
        if self.shuffle:
            self.np_random.shuffle(self._queue)

        self._step_idx          = 0
        self._cumulative_reward = 0.0
        self._current_email     = self._queue[0]

        obs = self._encode_to_obs(self._current_email)
        info = {
            "description": self._current_email["description"],
            "difficulty":  self._current_email["difficulty"],
            "task":        self.task,
            "total_steps": len(self._queue),
        }
        return obs.to_array(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Capture current email BEFORE advancing pointer
        scored_email = self._current_email
        raw_reward   = self._compute_reward(action, scored_email)
        norm_reward  = raw_reward / self._max_episode_reward

        self._cumulative_reward += norm_reward
        self._step_idx          += 1
        terminated               = self._step_idx >= len(self._queue)

        if not terminated:
            self._current_email = self._queue[self._step_idx]
            obs = self._encode_to_obs(self._current_email)
        else:
            obs = self._encode_to_obs(scored_email)

        # Decode action for info dict
        typed_action = EmailAction.from_array(action)
        correct      = scored_email["correct_actions"]

        info = {
            "raw_reward":       raw_reward,
            "correct_actions":  correct,
            "predicted":        (typed_action.urgency,
                                 typed_action.routing,
                                 typed_action.resolution),
            "difficulty":       scored_email["difficulty"],
            "description":      scored_email.get("description", ""),
            "urgency_label":    URGENCY_LABELS[typed_action.urgency],
            "routing_label":    ROUTING_LABELS[typed_action.routing],
            "resolution_label": RESOLUTION_LABELS[typed_action.resolution],
            "cumulative_score": self._cumulative_reward,
        }
        return obs.to_array(), norm_reward, terminated, False, info

    def state(self) -> EnvironmentState:
        """
        Return a typed snapshot of the current environment state.
        Required by the OpenEnv specification.
        """
        return EnvironmentState(
            step_index=self._step_idx,
            total_emails=len(self._queue),
            emails_remaining=max(0, len(self._queue) - self._step_idx),
            current_email=self._current_email,
            cumulative_reward=self._cumulative_reward,
            task=self.task,
            terminated=self._step_idx >= len(self._queue),
        )

    def render(self, mode: str = "human") -> None:
        e = self._current_email
        print(
            f"[{self.task.upper()} | Step {self._step_idx}/{len(self._queue)}] "
            f"{e['description']} | {e['difficulty']} | "
            f"sentiment={e['sentiment']} context={e['context']}"
        )
