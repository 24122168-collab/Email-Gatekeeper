import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Vocabulary & Encoding
# ---------------------------------------------------------------------------
KEYWORD_VOCAB = [
    "invoice", "payment", "overdue", "refund",
    "hacked", "breach", "unauthorized", "password",
    "crash", "error", "bug", "slow",
    "lawsuit", "legal", "attorney", "sue",
    "spam", "offer", "win", "free",
    "urgent", "critical", "angry", "threat",
]

SENTIMENT_MAP = {"positive": 0, "neutral": 1, "negative": 2}
CONTEXT_MAP = {"spam": 0, "billing": 1, "tech": 2, "security": 3, "legal": 4}
OBS_DIM = len(KEYWORD_VOCAB) + len(SENTIMENT_MAP) + len(CONTEXT_MAP)

# Labels for UI
URGENCY_LABELS = ["General", "Billing", "Security Breach"]
ROUTING_LABELS = ["AI Auto-Reply", "Tech Support", "Legal"]
RESOLUTION_LABELS = ["Archive", "Draft Reply", "Escalate to Human"]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
EMAIL_DATASET = [
    {"description": "Spam promo", "keywords": ["spam", "offer"], "sentiment": "positive", "context": "spam", "difficulty": "easy", "correct_actions": (0, 0, 0)},
    {"description": "Routine support", "keywords": ["slow", "error"], "sentiment": "neutral", "context": "tech", "difficulty": "easy", "correct_actions": (0, 1, 1)},
    {"description": "Billing inquiry", "keywords": ["invoice", "payment"], "sentiment": "neutral", "context": "billing", "difficulty": "easy", "correct_actions": (1, 0, 1)},
    {"description": "Overdue invoice", "keywords": ["invoice", "overdue"], "sentiment": "negative", "context": "billing", "difficulty": "medium", "correct_actions": (1, 0, 1)},
    {"description": "Refund dispute", "keywords": ["refund", "angry"], "sentiment": "negative", "context": "billing", "difficulty": "medium", "correct_actions": (1, 2, 2)},
    {"description": "Legal threat", "keywords": ["lawsuit", "attorney"], "sentiment": "negative", "context": "legal", "difficulty": "hard", "correct_actions": (2, 2, 2)},
    {"description": "Account hacked", "keywords": ["hacked", "password"], "sentiment": "negative", "context": "security", "difficulty": "hard", "correct_actions": (2, 1, 2)},
    {"description": "Data breach", "keywords": ["breach", "unauthorized"], "sentiment": "negative", "context": "security", "difficulty": "hard", "correct_actions": (2, 1, 2)},
]

# ---------------------------------------------------------------------------
# Environment Class
# ---------------------------------------------------------------------------
class EmailTriageEnv(gym.Env):
    def __init__(self, batch: list | None = None, shuffle: bool = True, task: str = "easy"):
        super().__init__()
        
        # Meta Grader logic: filter by task if no batch provided
        if batch is None:
            self.email_batch = [e for e in EMAIL_DATASET if e["difficulty"] == task]
            if not self.email_batch: # Fallback agar task match na ho
                self.email_batch = EMAIL_DATASET[:3]
        else:
            self.email_batch = batch

        self.shuffle = shuffle
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        
        self._max_episode_reward = len(self.email_batch) * 1.0

    def _encode(self, email):
        kw_flags = np.array([1.0 if kw in email["keywords"] else 0.0 for kw in KEYWORD_VOCAB], dtype=np.float32)
        sent_vec = np.zeros(len(SENTIMENT_MAP), dtype=np.float32)
        sent_vec[SENTIMENT_MAP[email["sentiment"]]] = 1.0
        ctx_vec = np.zeros(len(CONTEXT_MAP), dtype=np.float32)
        ctx_vec[CONTEXT_MAP[email["context"]]] = 1.0
        return np.concatenate([kw_flags, sent_vec, ctx_vec])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._queue = list(self.email_batch)
        if self.shuffle:
            self.np_random.shuffle(self._queue)
        self._step_idx = 0
        self._current_email = self._queue[self._step_idx]
        return self._encode(self._current_email), {"difficulty": self._current_email["difficulty"]}

    def step(self, action):
        scored_email = self._current_email
        correct = scored_email["correct_actions"]
        
        # Reward Logic
        raw_reward = 0.0
        if tuple(action) == correct:
            raw_reward = 1.0
        elif action[0] == correct[0]:
            raw_reward = 0.2 # Partial
            
        # Security Penalty
        if correct[0] == 2 and action[0] != 2:
            raw_reward = -2.0

        self._step_idx += 1
        terminated = self._step_idx >= len(self._queue)
        
        if not terminated:
            self._current_email = self._queue[self._step_idx]
            obs = self._encode(self._current_email)
        else:
            obs = self._encode(scored_email)

        # Normalize reward
        norm_reward = raw_reward / self._max_episode_reward
        
        info = {
            "raw_reward": raw_reward,
            "correct_actions": correct,
            "difficulty": scored_email["difficulty"]
        }
        return obs, norm_reward, terminated, False, info
