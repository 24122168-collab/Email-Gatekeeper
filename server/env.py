from typing import List, Dict, Any, Tuple
import random

URGENCY_LABELS = ["low", "medium", "high"]
ROUTING_LABELS = ["general", "support", "security"]
RESOLUTION_LABELS = ["ignore", "respond", "escalate"]


class EmailTriageEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self._queue: List[Dict] = []
        self._index = 0
        self._done = False

    # ✅ TASK-WISE DATA (required for grader)
    def _generate_emails(self) -> List[Dict]:
        task_data = {
            "easy": [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow and buggy", "label": [0, 1, 1]},
            ],
            "medium": [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow and buggy", "label": [0, 1, 1]},
                {"description": "Possible phishing attempt detected", "label": [2, 2, 2]},
                {"description": "Invoice mismatch and payment issue", "label": [1, 2, 2]},
            ],
            "hard": [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow and buggy", "label": [0, 1, 1]},
                {"description": "Possible phishing attempt detected", "label": [2, 2, 2]},
                {"description": "Invoice mismatch and payment issue", "label": [1, 2, 2]},
                {"description": "Ransomware attack suspected on system", "label": [2, 2, 2]},
                {"description": "User reports data breach and performance issues", "label": [2, 2, 2]},
            ],
        }

        emails = task_data.get(self.task, task_data["easy"])
        random.shuffle(emails)
        return emails

    # ✅ RESET
    def reset(self) -> Dict[str, Any]:
        self._queue = self._generate_emails()
        self._index = 0
        self._done = False
        return self.state()

    # ✅ STATE
    def state(self) -> Dict[str, Any]:
        if self._done:
            return {"done": True}

        current = self._queue[self._index]
        return {
            "description": current["description"],
            "step": self._index,
            "remaining": len(self._queue) - self._index,
            "done": False
        }

    # ✅ STEP (GRADER LOGIC)
    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict, Dict]:
        if self._done:
            return self.state(), 0.0, True, {}, {}

        correct = self._queue[self._index]["label"]

        # 🎯 PARTIAL REWARD (important)
        matches = sum(1 for a, b in zip(action, correct) if a == b)
        reward = matches / 3.0  # normalized [0,1]

        # 🔥 BONUS for perfect prediction
        if matches == 3:
            reward = 1.0

        self._index += 1

        if self._index >= len(self._queue):
            self._done = True

        return self.state(), reward, self._done, {}, {}
