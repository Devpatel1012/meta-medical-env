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

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("AGENT_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip("'").strip('"')

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def normalize_reward(r):
    """Forces reward to be strictly between 0 and 1 (non-inclusive)."""
    try:
        val = float(r)
        if val <= 0.0:
            return 0.01
        if val >= 1.0:
            return 0.99
        return val
    except (ValueError, TypeError):
        return 0.01

def safe_parse_prediction(raw_pred):
    if not isinstance(raw_pred, str):
        return raw_pred
    if raw_pred.startswith('[') and raw_pred.endswith(']'):
        try:
            return ast.literal_eval(raw_pred)
        except (ValueError, SyntaxError):
            return raw_pred
    return raw_pred

def format_action_for_stdout(action_data):
    """Ensures the action string has absolutely NO newlines, as required by the grader."""
    raw_str = str(action_data)
    return raw_str.replace('\n', ' ').replace('\r', '').strip()

def run_agent():
    env = FinalEnvironment()
    
    curriculum = ["Easy", "Medium", "Hard"]
    results = []

    for ep, target_diff in enumerate(curriculum):
       
        task_name = f"Medical_Task_{target_diff}"
        print(f"[START] task={task_name} env=ClinicalDiagnosisEnv model={MODEL_NAME}", flush=True)
        
        reward_val = 0.01
        action_str = "None"
        error_msg = "null"
        success = False

        try:
            obs = env.reset(target_difficulty=target_diff)
            patient_case = getattr(obs, 'observation', "")
            
            if not patient_case:
                raise ValueError("Empty patient case")

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

            action_str = format_action_for_stdout(prediction)
            typed_action = MedicalAction(prediction=str(prediction))
            
            obs_result = env.step(typed_action)
            
            raw_reward = getattr(obs_result, 'reward', 0.0)
            reward_val = normalize_reward(raw_reward)
            success = reward_val > 0.5 
            print(f"[STEP] step=1 action={action_str} reward={reward_val:.2f} done=true error=null", flush=True)

        except Exception as e:
            error_msg = str(e).replace('\n', ' ').replace('\r', '')
            print(f"[STEP] step=1 action={action_str} reward=0.01 done=true error={error_msg}", flush=True)
            reward_val = 0.01

        
        print(f"[END] success={str(success).lower()} steps=1 score={reward_val:.3f} rewards={reward_val:.2f}", flush=True)

    sys.exit(0)

if __name__ == "__main__":
    run_agent()