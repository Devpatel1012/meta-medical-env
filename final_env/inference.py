import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# 🚨 Updated imports to match the new folder structure
from server.final_env_environment import FinalEnvironment
from models import MedicalAction

logging.getLogger("openai").setLevel(logging.WARNING)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("AGENT_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip("'").strip('"')

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_agent():
    print(f"[START] Starting inference for model: {MODEL_NAME}")
    
   
    env = FinalEnvironment()
    curriculum = ["Easy", "Easy", "Medium", "Hard", "Hard"]
    total_reward = 0
    results = []

    for ep, target_diff in enumerate(curriculum):
        obs = env.reset(target_difficulty=target_diff)
        patient_case = obs.text
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a clinical extraction API. Output ONLY valid JSON. Schema: {\"prediction\": <string/array>, \"confidence\": <float>}."
                    },
                    {"role": "user", "content": patient_case}
                ],
                temperature=0.1
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            try:
                clean_json = raw_output.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                prediction = parsed.get("prediction", "Unknown")
            except Exception:
                prediction = raw_output

            typed_action = MedicalAction(prediction=str(prediction))
            _, reward, _, info = env.step(typed_action)
            total_reward += reward

            step_data = {
                "episode": ep + 1,
                "task": target_diff,
                "reward": round(float(reward), 3),
                "prediction": str(prediction)[:50]
            }
            print(f"[STEP] {json.dumps(step_data)}")
            
            results.append({
                "episode_id": ep + 1,
                "score": round(reward, 3),
                "ground_truth": info['ground_truth']
            })

        except Exception as e:
            print(f"Error in episode {ep+1}: {e}")

    avg_score = round(total_reward / len(curriculum), 3)
    
    with open("hackathon_results.json", "w") as f:
        json.dump({"average_score": avg_score, "results": results}, f, indent=2)

    print(f"[END] Evaluation complete. Average Score: {avg_score}")

if __name__ == "__main__":
    run_agent()