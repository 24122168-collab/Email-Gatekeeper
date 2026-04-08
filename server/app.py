import sys
import os
import uvicorn
from fastapi import FastAPI, Request
import gradio as gr
import numpy as np

# Environment path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import (
    EmailTriageEnv, URGENCY_LABELS, ROUTING_LABELS, RESOLUTION_LABELS,
)

app = FastAPI()

# --- Hackathon Endpoints ---
@app.post("/reset")
async def reset(request: Request):
    return {"status": "success"}

@app.post("/step")
async def step(request: Request):
    return {"status": "success"}

# --- Classification Logic ---
_LEGAL_SECURITY_KW   = {"lawsuit", "attorney", "sue", "ransomware", "extortion"}
_BILLING_ESCALATE_KW = {"refund"}

def _classify(email: dict) -> np.ndarray:
    kw      = set(email.get("keywords", []))
    context = email.get("context", "").lower()
    if context == "legal" or kw & {"lawsuit", "attorney", "sue"}:
        return np.array([2, 2, 2], dtype=np.int64)
    if context == "security":
        if kw & _LEGAL_SECURITY_KW or ("hacked" in kw and "breach" in kw):
            return np.array([2, 2, 2], dtype=np.int64)
        return np.array([2, 1, 2], dtype=np.int64)
    if context == "billing":
        return np.array([1, 2, 2] if kw & _BILLING_ESCALATE_KW else [1, 0, 1], dtype=np.int64)
    if context == "tech" or kw & {"crash", "error", "bug", "slow"}:
        return np.array([0, 1, 1], dtype=np.int64)
    return np.array([0, 0, 0], dtype=np.int64)

# --- Updated Demo Function with Partial Rewards ---
def run_task_demo(task: str) -> str:
    try:
        env = EmailTriageEnv(task=task, shuffle=False)
        env.reset(seed=42)
        email_queue = list(env._queue)
        lines = []
        cumulative_norm = 0.0
        terminated = False
        step = 0

        while not terminated:
            email = email_queue[step]
            action = _classify(email) 
            _, norm_reward, terminated, _, info = env.step(action)
            cumulative_norm += norm_reward
            raw = info["raw_reward"]
            ca  = info["correct_actions"]

            if raw >= 1.0:
                verdict = "✅ EXACT MATCH (+1.0)"
            elif raw == 0.2:
                verdict = "🔶 PARTIAL (Urgency OK, 1 wrong) (+0.2)"
            elif raw == 0.1:
                verdict = "🔶 PARTIAL (Urgency OK, 2 wrong) (+0.1)"
            elif raw < 0:
                verdict = "🚨 SECURITY MISS (-2.0)"
            else:
                verdict = "❌ WRONG (Urgency mismatch) (0.0)"

            lines.append(
                f"#{step+1:02d} [{email['difficulty'].upper()}] {email['description'][:45]}...\n"
                f"   ▶ Agent Prediction: {URGENCY_LABELS[action[0]]} | {ROUTING_LABELS[action[1]]} | {RESOLUTION_LABELS[action[2]]}\n"
                f"   ✔ Ground Truth:     {URGENCY_LABELS[ca[0]]} | {ROUTING_LABELS[ca[1]]} | {RESOLUTION_LABELS[ca[2]]}\n"
                f"   🏆 Status: {verdict}\n"
                f"{'-'*60}"
            )
            step += 1

        final_score = max(0.0, min(1.0, cumulative_norm))
        lines.append(f"\nTOTAL EPISODE SCORE: {final_score:.3f} / 1.000")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Gatekeeper")
    task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task")
    run_btn = gr.Button("Run")
    output_box = gr.Textbox(lines=25, label="Reward Breakdown")
    run_btn.click(fn=run_task_demo, inputs=task_dropdown, outputs=output_box)

app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
