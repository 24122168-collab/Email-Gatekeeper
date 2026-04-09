import sys
import os
import uvicorn
import numpy as np
import gradio as gr
from fastapi import FastAPI

# Path fix taaki env.py mil jaye
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import EmailTriageEnv, URGENCY_LABELS, ROUTING_LABELS, RESOLUTION_LABELS

app = FastAPI()

def smart_agent_logic(desc):
    desc = desc.lower()
    if any(x in desc for x in ["password", "hacked", "breach", "phish", "threat", "ransomware"]):
        return [2, 2, 2] if "threat" in desc or "ransomware" in desc else [2, 1, 2]
    if any(x in desc for x in ["billing", "refund", "dispute", "invoice", "payment"]):
        return [1, 2, 2]
    if any(x in desc for x in ["support", "routine", "slow", "error"]):
        return [0, 1, 1]
    return [0, 0, 0]

def run_demo(task):
    try:
        env = EmailTriageEnv(task=task)
        env.reset()
        results = []; total_reward = 0
        print(f"[START] Task: {task}")
        for i, email in enumerate(env._queue):
            action = smart_agent_logic(email['description'])
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
            print(f"[STEP] Index: {i} | Action: {action} | Reward: {reward}")
            status = "✅ MATCH" if reward >= 1.0 else "❌ MISMATCH"
            results.append(f"#{i+1} [{task.upper()}] {email['description'][:30]}... | {status}")
        score = total_reward / len(env._queue) if env._queue else 0
        print(f"[END] Final Score: {score}")
        return "\n".join(results) + f"\n\n--- FINAL SCORE: {score:.3f} ---"
    except Exception as e:
        return f"Error: {str(e)}"

# --- REQUIRED BY VALIDATOR ---
def main():
    print("--- 🚀 STARTING MULTI-MODE DEPLOYMENT TEST ---")
    for level in ["easy", "medium", "hard"]:
        print(run_demo(level))

@app.post("/reset")
async def reset_endpoint():
    return {"status": "success"}

# Gradio Setup
with gr.Blocks() as demo:
    gr.Markdown("# 📧 Email Gatekeeper AI")
    diff = gr.Dropdown(["easy", "medium", "hard"], value="easy", label="Difficulty")
    btn = gr.Button("Analyze Emails")
    out = gr.Textbox(label="Logs", lines=10)
    btn.click(run_demo, inputs=diff, outputs=out)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        main()
    else:
        uvicorn.run(app, host="0.0.0.0", port=7860)
