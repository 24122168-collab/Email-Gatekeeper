from typing import List, Dict, Any, Tuple

URGENCY_LABELS = ["low", "medium", "high"]
ROUTING_LABELS = ["general", "support", "security"]
RESOLUTION_LABELS = ["ignore", "respond", "escalate"]


class EmailTriageEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self._queue: List[Dict] = []
        self._index = 0
        self._done = False

    # ✅ EXPLICIT TASK DATA (NO RANDOMNESS)
    def _generate_emails(self) -> List[Dict]:
        if self.task == "easy":
            return [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow", "label": [0, 1, 1]},
            ]

        elif self.task == "medium":
            return [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow", "label": [0, 1, 1]},
                {"description": "Possible phishing attempt detected", "label": [2, 2, 2]},
                {"description": "Invoice mismatch issue", "label": [1, 2, 2]},
            ]

        elif self.task == "hard":
            return [
                {"description": "Password reset not working", "label": [2, 1, 2]},
                {"description": "Billing refund request", "label": [1, 2, 2]},
                {"description": "App is slow", "label": [0, 1, 1]},
                {"description": "Possible phishing attempt detected", "label": [2, 2, 2]},
                {"description": "Invoice mismatch issue", "label": [1, 2, 2]},
                {"description": "Ransomware attack suspected", "label": [2, 2, 2]},
                {"description": "Data breach reported", "label": [2, 2, 2]},
            ]

        else:
            return []

    # ✅ RESET (DETERMINISTIC)
    def reset(self) -> Dict[str, Any]:
        self._queue = self._generate_emails()
        self._index = 0
        self._done = False

        return {
            "description": self._queue[self._index]["description"],
            "step": 0,
            "remaining": len(self._queue),
            "done": False
        }

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

    # ✅ STEP (CLEAR GRADER)
    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict, Dict]:
        if self._done:
            return self.state(), 0.0, True, {}, {}

        correct = self._queue[self._index]["label"]

        # 🎯 GRADER (CLEAR + NORMALIZED)
        matches = sum(1 for a, b in zip(action, correct) if a == b)
        reward = matches / 3.0  # normalized [0,1]

        self._index += 1

        if self._index >= len(self._queue):
            self._done = True

        return self.state(), reward, self._done, {}, {}
