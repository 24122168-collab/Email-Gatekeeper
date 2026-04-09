import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- UI LABELS (Zaroori hain app.py ke liye) ---
URGENCY_LABELS = ["General", "Billing", "Security Breach"]
ROUTING_LABELS = ["AI Auto-Reply", "Tech Support", "Legal"]
RESOLUTION_LABELS = ["Archive", "Draft Reply", "Escalate to Human"]

# --- Vocabulary & Encoding Configuration ---
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

class EmailTriageEnv(gym.Env):
    def __init__(self, task="all", batch=None, shuffle=True):
        super().__init__()
        
        # Dataset load logic
        try:
            from app import EMAIL_DATASET
            dataset_to_use = EMAIL_DATASET
        except ImportError:
            dataset_to_use = [] 

        # App.py ko '_queue' attribute hi chahiye interface ke liye
        if batch is not None:
            self._queue = batch
        elif task != "all":
            self._queue = [e for e in dataset_to_use if e.get("difficulty") == task]
        else:
            self._queue = dataset_to_use
            
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
        if not self._queue: return np.zeros(OBS_DIM, dtype=np.float32), {}
        obs = self._encode(self._queue[0])
        return obs, {"description": self._queue[0].get("description", "")}

    def step(self, action):
        email = self._queue[self._step_idx]
        correct = email["correct_actions"]
        
        # Reward Logic: Security miss par bhari penalty
        reward = 0.0
        if correct[0] == 2 and action[0] != 2:
            reward = -2.0  
        elif tuple(action) == correct:
            reward = 1.0
        elif action[0] == correct[0]:
            reward = 0.2

        self._step_idx += 1
        terminated = self._step_idx >= len(self._queue)
        obs = self._encode(email) # current obs
        
        info = {
            "description": email.get("description", ""),
            "correct_actions": correct,
            "raw_reward": reward
        }
        return obs, float(reward), terminated, False, info
