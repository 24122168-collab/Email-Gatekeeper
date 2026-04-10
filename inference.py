import os
from typing import List, Optional

from openai import OpenAI

from env import EmailTriageEnv
from app import smart_agent_logic


# ✅ MUST use these EXACT env vars
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

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
    # ✅ REQUIRED: Initialize OpenAI client with provided proxy
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[DEBUG] OpenAI init failed: {e}", flush=True)
        client = None

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

                # ✅ 🔥 LLM CALL (MANDATORY)
                action_list = None

                if client:
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Classify email into 3 integers: urgency (0-2), routing (0-2), resolution (0-2). Return only numbers like: 2 1 2"
                                },
                                {
                                    "role": "user",
                                    "content": desc
                                }
                            ],
                            max_tokens=20,
                            temperature=0,
                        )

                        text = response.choices[0].message.content.strip()

                        # Parse response
                        action_list = [int(x) for x in text.replace(",", " ").split()[:3]]

                        if len(action_list) != 3:
                            raise ValueError("Invalid LLM output")

                    except Exception as llm_error:
                        print(f"[DEBUG] LLM failed: {llm_error}", flush=True)

                # ✅ fallback if LLM fails
                if not action_list:
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
