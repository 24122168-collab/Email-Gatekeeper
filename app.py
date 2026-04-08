"""
app.py — Gradio Web Interface for Hugging Face Spaces
=====================================================
Provides an interactive demo of the Email Gatekeeper RL environment.
Hugging Face Spaces serves this on port 7860 automatically.
"""

import gradio as gr
import numpy as np
from env import (
    EmailTriageEnv, TASK_SPLITS,
    URGENCY_LABELS, ROUTING_LABELS, RESOLUTION_LABELS,
)

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
        return np.array([1, 2, 2] if kw & _BILLING_ESCALATE_KW
                        else [1, 0, 1], dtype=np.int64)
    if context == "tech" or kw & {"crash", "error", "bug", "slow"}:
        return np.array([0, 1, 1], dtype=np.int64)
    return np.array([0, 0, 0], dtype=np.int64)


def run_task_demo(task: str) -> str:
    env = EmailTriageEnv(task=task, shuffle=False)
    env.reset(seed=42)
    email_queue = list(env._queue)

    lines        = []
    cumulative   = 0.0
    terminated   = False
    step         = 0

    while not terminated:
        email  = email_queue[step]
        action = _classify(email)
        _, norm_reward, terminated, _, info = env.step(action)
        cumulative += norm_reward

        raw = info["raw_reward"]
        ca  = info["correct_actions"]

        verdict = ("✅ EXACT" if raw >= 1.0 else
                   "🔶 PARTIAL" if raw > 0 else
                   "🚨 SECURITY MISS" if raw < 0 else "❌ WRONG")

        lines.append(
            f"#{step+1:02d} [{email['difficulty'].upper()}] "
            f"{email['description'][:40]}\n"
            f"     Predicted : {URGENCY_LABELS[action[0]]} | "
            f"{ROUTING_LABELS[action[1]]} | {RESOLUTION_LABELS[action[2]]}\n"
            f"     Correct   : {URGENCY_LABELS[ca[0]]} | "
            f"{ROUTING_LABELS[ca[1]]} | {RESOLUTION_LABELS[ca[2]]}\n"
            f"     Reward    : {raw:+.1f}  {verdict}\n"
        )
        step += 1

    final = max(0.0, min(1.0, cumulative))
    lines.append(f"\n{'─'*50}")
    lines.append(f"Final Score : {final:.3f} / 1.0")
    return "\n".join(lines)


with gr.Blocks(title="Email Gatekeeper RL") as demo:
    gr.Markdown("""
# 📧 Email Gatekeeper — RL Environment Demo
**Meta x PyTorch Hackathon** | Gymnasium-based email triage agent

The agent classifies each email across **3 simultaneous dimensions**:
`Urgency` × `Department` × `Resolution Action`
""")

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Select Task",
        )
        run_btn = gr.Button("▶ Run Episode", variant="primary")

    output_box = gr.Textbox(
        label="Episode Results",
        lines=30,
        max_lines=50,
    )

    run_btn.click(fn=run_task_demo, inputs=task_dropdown, outputs=output_box)

    gr.Markdown("""
### Reward Function
| Result | Reward |
|---|---|
| ✅ Exact Match (all 3 correct) | +1.0 |
| 🔶 Partial (urgency correct, 1 wrong) | +0.2 |
| 🔶 Partial (urgency correct, 2 wrong) | +0.1 |
| 🚨 Security Miss | **-2.0** |
| ❌ Wrong urgency | 0.0 |
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
