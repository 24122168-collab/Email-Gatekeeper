import os
from typing import List, Optional

from openai import OpenAI

from env import EmailTriageEnv
from app import smart_agent_logic


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy")
BENCHMARK = "email_triage_env"

MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def main():
    # ✅ SAFE OpenAI initialization (FIX)
    client = None
    try:
        if API_KEY:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        else:
            print("[DEBUG] No API key found, running without OpenAI client", flush=True)
    except Exception as e:
        print(f"[DEBUG] OpenAI init failed: {e}", flush=True)

    env = EmailTriageEnv(task=TASK_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        state = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if state.get("done"):
                break

            try:
                desc = state["description"]

                action_list = smart_agent_logic(desc)

                state, reward, done, _, _ = env.step(action_list)

                rewards.append(reward)
                steps_taken = step

                log_step(step, str(action_list), reward, done, None)

                if done:
                    break

            except Exception as step_error:
                log_step(step, "error", 0.0, True, str(step_error))
                break

        if rewards:
            score = sum(rewards) / len(rewards)
            score = max(0.0, min(score, 1.0))

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
