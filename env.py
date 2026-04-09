import numpy as np
import gymnasium as gym
from gymnasium import spaces

URGENCY_LABELS = ["General", "Billing", "Security Breach"]
ROUTING_LABELS = ["AI Auto-Reply", "Tech Support", "Legal"]
RESOLUTION_LABELS = ["Archive", "Draft Reply", "Escalate to Human"]

class EmailTriageEnv(gym.Env):
    def __init__(self, task="all"):
        super().__init__()
        self.full_dataset = [
            {"difficulty": "easy", "description": "Spam promo", "correct_actions": (0, 0, 0)},
            {"difficulty": "easy", "description": "Routine support", "correct_actions": (0, 1, 1)},
            {"difficulty": "medium", "description": "Billing dispute", "correct_actions": (1, 2, 2)},
            {"difficulty": "medium", "description": "Refund request", "correct_actions": (1, 2, 2)},
            {"difficulty": "hard", "description": "IT password reset phish", "correct_actions": (2, 1, 2)},
            {"difficulty": "hard", "description": "Ransomware threat", "correct_actions": (2, 2, 2)}
        ]
        
        if task != "all":
            self._queue = [e for e in self.full_dataset if e.get("difficulty") == task]
        else:
            self._queue = self.full_dataset
            
        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self._step_idx = 0

    def reset(self, seed=None, options=None):
        self._step_idx = 0
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        if self._step_idx >= len(self._queue):
            return np.zeros(10), 0.0, True, False, {}
            
        email = self._queue[self._step_idx]
        correct = email["correct_actions"]
        
        reward = 1.0 if tuple(action) == tuple(correct) else 0.0
        # Penalty for missing security threats
        if correct[0] == 2 and action[0] != 2:
            reward = -2.0

        self._step_idx += 1
        done = self._step_idx >= len(self._queue)
        return np.zeros(10), float(reward), done, False, {"raw_reward": reward}
