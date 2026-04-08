import os
import json
import logging
import sys
import ast
from openai import OpenAI
from dotenv import load_dotenv
from server.final_env_environment import FinalEnvironment
from models import MedicalAction

logging.getLogger("openai").setLevel(logging.WARNING)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("AGENT_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip("'").strip('"')

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def safe_parse_prediction(raw_pred):
    if not isinstance(raw_pred, str):
        return raw_pred
    if raw_pred.startswith('[') and raw_pred.endswith(']'):
        try:
            return ast.literal_eval(raw_pred)
        except (ValueError, SyntaxError):
            return raw_pred
    return raw_pred

def align_type_with_ground_truth(prediction, ground_truth):
    if isinstance(ground_truth, list) and isinstance(prediction, str):
        return [item.strip() for item in prediction.split(',')]
    if isinstance(ground_truth, (str, bytes)) and isinstance(prediction, list):
        return ", ".join(map(str, prediction))
    return prediction

def run_agent():
    print(f"[START] Starting inference for model: {MODEL_NAME}")
    
    env = FinalEnvironment()
    curriculum = ["Easy", "Easy", "Medium", "Hard", "Hard"]
    total_reward = 0.0
    results = []

    for ep, target_diff in enumerate(curriculum):
        try:
            obs = env.reset(target_difficulty=target_diff)
            patient_case = getattr(obs, 'observation', "")
            
            if not patient_case:
                print(f"Warning: Episode {ep+1} received no observation.")
                continue

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
                parsed_data = json.loads(clean_json)
                prediction = safe_parse_prediction(parsed_data.get("prediction", "Unknown"))
            except Exception:
                prediction = safe_parse_prediction(raw_output)

            typed_action = MedicalAction(prediction=str(prediction))
            obs_result = env.step(typed_action)
            
            reward_val = float(getattr(obs_result, 'reward', 0.0))
            info = getattr(obs_result, 'info', {})
            ground_truth = info.get('ground_truth', '')

            final_prediction = align_type_with_ground_truth(prediction, ground_truth)
            total_reward += reward_val

            step_data = {
                "episode": ep + 1,
                "reward": round(reward_val, 3),
                "type_match": type(final_prediction) == type(ground_truth),
                "prediction": final_prediction,
                "ground_truth": ground_truth
            }
            print(f"[STEP SUCCESS] {json.dumps(step_data)}")
            
            results.append({
                "episode_id": ep + 1,
                "score": reward_val,
                "prediction": final_prediction,
                "ground_truth": ground_truth
            })

        except Exception as e:
            print(f"Critical Error in episode {ep+1}: {str(e)}")

    avg_score = round(total_reward / len(curriculum), 3) if curriculum else 0.0
    
    try:
        with open("hackathon_results.json", "w") as f:
            json.dump({"average_score": avg_score, "results": results}, f, indent=2)
    except Exception as e:
        print(f"Failed to write results file: {e}")

    print(f"[END] Evaluation complete. Average Score: {avg_score}")
    sys.exit(0)

if __name__ == "__main__":
    run_agent()