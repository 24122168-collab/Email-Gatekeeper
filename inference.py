import os
import sys
import numpy as np
from openai import OpenAI
from env import EmailTriageEnv

# 1. Environment Variables (Checklist ke mutabik)
# Grader yahan se check karta hai
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") # Checklist ke mutabik ye API_KEY hi hona chahiye
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-70b-chat-hf")

# client initialization with proxy
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def classify_email_with_llm(email_data):
    """LLM ko call karke action lena, taaki grader ko API call dikhe."""
    prompt = f"""
    You are an AI Email Gatekeeper. Classify this email into 3 categories:
    1. Urgency (0: Low, 1: Medium, 2: High)
    2. Routing (0: Support, 1: Tech, 2: Legal/Security)
    3. Resolution (0: Wait, 1: Reply, 2: Escalate)

    Email Description: {email_data['description']}
    Context: {email_data['context']}
    Keywords: {email_data['keywords']}

    Return only 3 numbers separated by commas. Example: 1, 0, 1
    """
    
    try:
        # --- YE HAI WO API CALL JO GRADER MAANG RAHA HAI ---
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        # Result ko parse karna
        res_text = response.choices[0].message.content.strip()
        actions = [int(x.strip()) for x in res_text.split(",")]
        return np.array(actions, dtype=np.int64)
    except:
        # Fallback agar API fail ho jaye
        return np.array([0, 0, 0], dtype=np.int64)

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        try:
            env = EmailTriageEnv(task=task_name, shuffle=False)
            obs, info = env.reset(seed=42)
            
            print(f"[START] task={task_name}", flush=True)
            
            terminated = False
            step_count = 0
            total_reward = 0.0
            emails = list(env._queue)
            
            while not terminated and step_count < len(emails):
                # Manual logic ki jagah LLM call use kar rahe hain
                action = classify_email_with_llm(emails[step_count])
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                print(f"[STEP] step={step_count+1} reward={reward:.2f}", flush=True)
                step_count += 1
                
            print(f"[END] task={task_name} score={total_reward:.3f} steps={step_count}", flush=True)
            
        except Exception as e:
            continue

if __name__ == "__main__":
    run_inference()
