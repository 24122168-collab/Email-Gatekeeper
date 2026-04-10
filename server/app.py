import os
from typing import List, Tuple

from env import EmailTriageEnv


def smart_agent_logic(desc: str) -> List[int]:
    desc = desc.lower()

    if any(x in desc for x in ["password", "hacked", "breach", "phish", "threat", "ransomware"]):
        return [2, 2, 2] if "threat" in desc or "ransomware" in desc else [2, 1, 2]

    if any(x in desc for x in ["billing", "refund", "invoice", "payment"]):
        return [1, 2, 2]

    if any(x in desc for x in ["slow", "error", "bug", "support"]):
        return [0, 1, 1]

    return [0, 0, 0]


def run_episode(task: str) -> Tuple[float, List[float], int]:
    env = EmailTriageEnv(task=task)

    state = env.reset()

    rewards: List[float] = []
    steps = 0
    total_reward = 0.0

    while True:
        if state.get("done"):
            break

        desc = state["description"]

        action = smart_agent_logic(desc)

        state, reward, done, _, _ = env.step(action)

        rewards.append(reward)
        total_reward += reward
        steps += 1

        if done:
            break

    score = total_reward / len(rewards) if rewards else 0.0

    return score, rewards, steps


def main():
    task = os.getenv("MY_ENV_V4_TASK", "easy")

    score, rewards, steps = run_episode(task)

    print(f"Task={task} | Steps={steps} | Score={score:.3f}")


if __name__ == "__main__":
    main()
