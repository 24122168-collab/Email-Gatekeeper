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
    prompt = f"Classify email: {email_desc}. Return 3 numbers (0-2) separated by commas."
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
            # Environment Setup (Ensure env.py supports 'task' parameter)
            env = EmailTriageEnv(task=task_name, shuffle=False)
            obs, info = env.reset(seed=42)
            
            # [START] tag for Output Parsing
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
                
                # [STEP] tag
                print(f"[STEP] step={step_count+1} reward={reward:.2f}", flush=True)
                step_count += 1
                
            
            final_score = max(0.01, min(0.99, total_reward))
            
     
            print(f"[END] task={task_name} score={final_score:.3f} steps={step_count}", flush=True) 
            
        except Exception as e:
         
            print(f"[END] task={task_name} score=0.010 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()
