import os
import traceback
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from env import MedicalEnv

# 1. Load API from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
AGENT_MODEL = os.getenv("AGENT_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

client = InferenceClient(api_key=HF_TOKEN)

def run_agent():
    print("\n--- Starting Medical Triage Evaluation ---\n")
    
    # Initialize the robust Environment
    env = MedicalEnv()
    
    episodes = 5
    total_reward = 0

    for ep in range(episodes):
        print(f"[Episode {ep+1}/{episodes}]")
        
        # Load a random patient from the JSON dataset
        patient_case = env.reset()
        print("Patient:", patient_case[:120], "...")

        try:
            response = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict medical data extraction AI. "
                            "Do not explain. Do not list multiple options. "
                            "Output exactly ONE test, ONE diagnosis, or ONE list of symptoms."
                        )
                    },
                    # --- Example 1: Easy (Symptom Extraction) ---
                    {
                        "role": "user", 
                        "content": "Patient presents with a severe headache, high fever, and sudden chills. Identify the main symptoms."
                    },
                    {
                        "role": "assistant", 
                        "content": "severe headache, high fever, sudden chills"
                    },
                    
                    # --- Example 2: Medium (Test Recommendation) ---
                    {
                        "role": "user", 
                        "content": "A 45-year-old female presents with right upper quadrant abdominal pain after eating fatty foods. What imaging is recommended?"
                    },
                    {
                        "role": "assistant", 
                        "content": "Abdominal Ultrasound"
                    },
                    
                    # --- Example 3: Hard (Clinical Diagnosis) ---
                    {
                        "role": "user", 
                        "content": "A 22-year-old male athlete presents with progressive gait ataxia, slurred speech, and choreiform movements. He has a family history of similar symptoms. Provide the diagnosis."
                    },
                    {
                        "role": "assistant", 
                        "content": "Huntington's Disease"
                    },
                    
                    # --- The Actual Patient Case ---
                    {
                        "role": "user",
                        "content": patient_case
                    }
                ],
                temperature=0.1,
                max_tokens=20 # Reduced tokens to force it to stop talking!
            )
            action = response.choices[0].message.content.strip()
            
        except Exception as e:
            print("Agent API failed")
            traceback.print_exc()
            action = "Unknown"

        print("Agent Answer:", action)

        # Pass the Agent's answer to the Environment for grading
        _, reward, _, info = env.step(action)

        print("Ground Truth:", info["ground_truth"])
        print("Difficulty:", info["difficulty"])
        print("Reward:", round(reward, 2))
        print()

        total_reward += reward

    avg_score = total_reward / episodes
    print("--- Evaluation Complete ---")
    print("Average Score:", round(avg_score, 2))

if __name__ == "__main__":
    run_agent()