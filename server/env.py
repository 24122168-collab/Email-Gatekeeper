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

    def _generate_emails(self) -> List[Dict]:
        emails = [
            {"description": "Password reset not working", "label": [2, 1, 2]},
            {"description": "Billing refund request", "label": [1, 2, 2]},
            {"description": "App is slow and buggy", "label": [0, 1, 1]},
        ]

        if self.task in ["medium", "hard"]:
            emails += [
                {"description": "Possible phishing attempt detected", "label": [2, 2, 2]},
                {"description": "Invoice mismatch and payment issue", "label": [1, 2, 2]},
            ]

        if self.task == "hard":
            emails += [
                {"description": "Ransomware attack suspected on system", "label": [2, 2, 2]},
                {"description": "User reports data breach and performance issues", "label": [2, 2, 2]},
            ]

        random.shuffle(emails)
        return emails

    def reset(self) -> Dict[str, Any]:
        self._queue = self._generate_emails()
        self._index = 0
        self._done = False
        return self.state()

    def state(self) -> Dict[str, Any]:
        if self._done:
            return {"done": True}

        current = self._queue[self._index]
        return {
            "description": current["description"],
            "step": self._index,
            "remaining": len(self._queue) - self._index,
        }

    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict, Dict]:
        if self._done:
            return self.state(), 0.0, True, {}, {}

        correct = self._queue[self._index]["label"]

        reward = sum(
            1.0 if a == b else 0.0
            for a, b in zip(action, correct)
        ) / 3.0

        self._index += 1

        if self._index >= len(self._queue):
            self._done = True

        return self.state(), reward, self._done, {}, {}
