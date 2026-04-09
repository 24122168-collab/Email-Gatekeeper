import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- UI LABELS (Ye labels app.py mang raha hai) ---
URGENCY_LABELS = ["General", "Billing", "Security Breach"]
ROUTING_LABELS = ["AI Auto-Reply", "Tech Support", "Legal"]
RESOLUTION_LABELS = ["Archive", "Draft Reply", "Escalate to Human"]

# --- Vocabulary & Encoding Configuration ---
KEYWORD_VOCAB = [
    "invoice", "payment", "overdue", "refund",          # billing
    "hacked", "breach", "unauthorized", "password",     # security
    "crash", "error", "bug", "slow",                    # tech
    "lawsuit", "legal", "attorney", "sue",              # legal
    "spam", "offer", "win", "free",                     # spam
    "urgent", "critical", "angry", "threat",            # sentiment signals
]

SENTIMENT_MAP = {"positive": 0, "neutral": 1, "negative": 2}
CONTEXT_MAP = {"spam": 0, "billing": 1, "tech": 2, "security": 3, "legal": 4}
OBS_DIM = len(KEYWORD_VOCAB) + len(SENTIMENT_MAP) + len(CONTEXT_MAP)

# --- Environment Class ---
class EmailTriageEnv(gym.Env):
    def __init__(self, task="all", batch=None, shuffle=True):
        super().__init__()
        
        # Dataset ko import karna (app.py se load hoga)
        try:
            from app import EMAIL_DATASET
            dataset_to_use = EMAIL_DATASET
        except ImportError:
            dataset_to_use = [] # Fallback agar dataset na mile

        if batch is not None:
            self.email_batch = batch
        elif task != "all":
            self.email_batch = [e for e in dataset_to_use if e.get("difficulty") == task]
        else:
            self.email_batch = dataset_to_use
            
        self.shuffle = shuffle
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._step_idx = 0

    def _encode(self, email):
        kw_flags = np.array([1.0 if kw in email.get("keywords", []) else 0.0 for kw in KEYWORD_VOCAB])
        sent_idx = SENTIMENT_MAP.get(email.get("sentiment", "neutral"), 1)
        sentiment_vec = np.zeros(len(SENTIMENT_MAP)); sentiment_vec[sent_idx] = 1.0
        
        ctx_idx = CONTEXT_MAP.get(email.get("context", "spam"), 0)
        context_vec = np.zeros(len(CONTEXT_MAP)); context_vec[ctx_idx] = 1.0
        
        return np.concatenate([kw_flags, sentiment_vec, context_vec]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        if not self.email_batch:
            return np.zeros(OBS_DIM, dtype=np.float32), {}
        
        obs = self._encode(self.email_batch[0])
        return obs, {"description": self.email_batch[0].get("description", "")}

    def step(self, action):
        email = self.email_batch[self._step_idx]
        correct = email["correct_actions"]
        
        # Reward Logic (Score sudharne ke liye)
        reward = 0.0
        if correct[0] == 2 and action[0] != 2:
            reward = -2.0  # Security missed penalty
        elif tuple(action) == correct:
            reward = 1.0
        elif action[0] == correct[0]:
            reward = 0.2

        self._step_idx += 1
        terminated = self._step_idx >= len(self.email_batch)
        
        # Next observation
        if not terminated:
            next_email = self.email_batch[self._step_idx]
            obs = self._encode(next_email)
        else:
            obs = self._encode(email)
        
        info = {
            "description": email.get("description", ""),
            "correct_actions": correct,
            "raw_reward": reward
        }
        
        return obs, float(reward), terminated, False, info
