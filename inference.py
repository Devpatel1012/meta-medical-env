import os
import json
import logging
import traceback
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Silence network logs
logging.getLogger("httpx").setLevel(logging.WARNING)

from env import MedicalEnv

load_dotenv()

# --- Docker-Proofing the Token ---
raw_token = os.getenv("HF_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip("'").strip('"')
if not HF_TOKEN:
    print("[Warning] No HF_TOKEN found. API calls may fail or be throttled.")
    
# --- Docker-Proofing the Agent Models ---
agent_models_str = os.getenv("AGENT_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
agent_models_str = agent_models_str.strip('"').strip("'")
agent_models = [m.strip() for m in agent_models_str.split(",")]

client = InferenceClient(
    token=HF_TOKEN,
    timeout=120
)
def run_agent():
    print("\n--- Starting Medical Triage Evaluation ---")
    
    env = MedicalEnv()
    curriculum = ["Easy", "Easy", "Medium", "Hard", "Hard"]
    
    report = {
        "metadata": {
            "primary_agent_model": agent_models[0],
            "total_episodes": len(curriculum)
        },
        "results": [],
        "average_score": 0.0
    }
    
    total_reward = 0

    for ep, target_diff in enumerate(curriculum):
        print(f"\n[Episode {ep+1}/{len(curriculum)}] Target: {target_diff}")
        
        patient_case = env.reset(target_difficulty=target_diff)
        print(f"Input:  {patient_case[:100]}...")
        
        raw_output = "{}"
        prediction = "Unknown"
        confidence = 0.0
        active_model = "None"

        for model in agent_models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a clinical extraction API. You MUST output ONLY valid JSON. "
                                "Schema: {\"prediction\": <string or array of strings>, \"confidence\": <float between 0.0 and 1.0>}. "
                                "Do NOT use markdown code blocks. Do not explain."
                            )
                        },
                        {"role": "user", "content": "Patient presents with headache, fever, and chills. Extract symptoms."},
                        {"role": "assistant", "content": "{\"prediction\": [\"headache\", \"fever\", \"chills\"], \"confidence\": 0.95}"},
                        {"role": "user", "content": "Patient has suspected gallstones. What imaging?"},
                        {"role": "assistant", "content": "{\"prediction\": \"Abdominal Ultrasound\", \"confidence\": 0.90}"},
                        {"role": "user", "content": patient_case}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                raw_output = response.choices[0].message.content.strip()
                active_model = model
                
                try:
                    clean_json_str = raw_output.replace("```json", "").replace("```", "").strip()
                    parsed_response = json.loads(clean_json_str)
                    prediction = parsed_response.get("prediction", "Unknown")
                    confidence = parsed_response.get("confidence", 0.0)
                except json.JSONDecodeError:
                    prediction = raw_output 
                    confidence = 0.0
                
                break 
                
            except Exception as e:
                print(f"\n[ERROR] Agent {model} failed")
                print("Reason:", str(e))
                traceback.print_exc()
                continue

        _, reward, _, info = env.step(prediction)
        total_reward += reward

        print(f"Output: {prediction} (Confidence: {confidence})")
        print(f"Truth:  {info['ground_truth']}")
        print(f"Score:  {round(reward, 3)}")

        report["results"].append({
            "episode_id": ep + 1,
            "task_difficulty": target_diff,
            "ground_truth": info["ground_truth"],
            "prediction": prediction,
            "confidence": confidence,
            "score": round(reward, 3),
            "model_used": active_model
        })

    final_avg = round(total_reward / len(curriculum), 3)
    report["average_score"] = final_avg
    
    print("\n--- Evaluation Complete ---")
    print(f"Average Score: {final_avg}")

    output_filename = "hackathon_results.json"
    with open(output_filename, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[success] Full JSON report saved to {output_filename}")

if __name__ == "__main__":
    run_agent()