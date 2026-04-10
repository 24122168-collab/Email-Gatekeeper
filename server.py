from fastapi import FastAPI
import gradio as gr

from env import EmailTriageEnv
from app import smart_agent_logic

app = FastAPI()

# -------------------------
# REQUIRED API ENDPOINTS (DO NOT REMOVE)
# -------------------------
@app.post("/reset")
async def reset():
    return {"status": "ok"}

@app.get("/status")
async def status():
    return {"status": "online"}


# -------------------------
# GRADIO FUNCTION (THIS SHOWS REWARD + SCORE)
# -------------------------
def demo_fn(task):
    env = EmailTriageEnv(task=task)
    state = env.reset()

    results = []
    total_reward = 0.0
    steps = 0

    while True:
        if state.get("done"):
            break

        desc = state["description"]
        action = smart_agent_logic(desc)

        state, reward, done, _, _ = env.step(action)

        total_reward += reward
        steps += 1

        results.append(
            f"### Step {steps}\n"
            f"- 📧 Email: {desc}\n"
            f"- 🤖 Action: {action}\n"
            f"- ⭐ Reward: {reward:.2f}\n"
            f"---\n"
        )

        if done:
            break

    score = total_reward / steps if steps > 0 else 0.0

    return f"""
# 📊 Email Triage Results

{''.join(results)}

## 🏁 Final Score: **{score:.3f} / 1.000**
"""


# -------------------------
# GRADIO UI
# -------------------------
demo = gr.Interface(
    fn=demo_fn,
    inputs=gr.Dropdown(["easy", "medium", "hard"], label="Select Difficulty"),
    outputs=gr.Markdown(label="Results"),
    title="📧 Email Gatekeeper",
    description="AI Agent for Email Triage using Reinforcement Learning"
)

# -------------------------
# MOUNT UI
# -------------------------
app = gr.mount_gradio_app(app, demo, path="/")
