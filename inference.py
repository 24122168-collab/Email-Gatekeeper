import os
import sys
import numpy as np
from openai import OpenAI
from env import EmailTriageEnv

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY") 
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3-70b-chat-hf")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_llm_action(email_desc, context, keywords):
    prompt = f"""
    Email: {email_desc}
    Context: {context}
    Keywords: {keywords}
    Classify: Urgency (0-2), Routing (0-2), Resolution (0-2).
    Return only numbers: e.g., 1, 0, 1
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        res = response.choices[0].message.content.strip()
        actions = [int(x.strip()) for x in res.split(",")[:3]]
        return np.array(actions, dtype=np.int64)
    except:
        return np.array([0, 0, 0], dtype=np.int64)

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for task_name in tasks:
        try:
            # Task load karna (Ensure your env.py supports task argument)
            env = EmailTriageEnv(task=task_name, shuffle=False)
            obs, info = env.reset(seed=42)
            
            print(f"[START] task={task_name}", flush=True)
            
            terminated = False
            step_count = 0
            total_reward = 0.0
            emails = list(env._queue)
            
            while not terminated and step_count < len(emails):
                action = get_llm_action(emails[step_count]['description'], 
                                        emails[step_count]['context'], 
                                        emails[step_count]['keywords'])
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                print(f"[STEP] step={step_count+1} reward={reward:.2f}", flush=True)
                step_count += 1
                
            # --- FIX: SCORE CLIPPING (Strictly between 0 and 1) ---
            # Agar score 1.0 hai toh 0.99 ho jayega, agar 0.0 hai toh 0.01
            final_score = max(0.01, min(0.99, total_reward))
            
            print(f"[END] task={task_name} score={final_score:.3f} steps={step_count}", flush=True)
            
        except Exception as e:
            # Failure case mein bhi 0.010 bhej rahe hain range error se bachne ke liye
            print(f"[END] task={task_name} score=0.010 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()
