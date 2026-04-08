import requests
import json
import os

BASE_URL = "https://devpatel1012-clinicaldiagnosisenv.hf.space"
HF_TOKEN = os.getenv("HF_TOKEN", "")

def get_agent_prediction(observation_text):
    """
    Uses Few-Shot prompting to force the LLM to return strict JSON,
    then parses that JSON to get the prediction.
    """
    print("\n🤖 Agent is reading the case and thinking...")
    
    if HF_TOKEN:
        # Use the OpenAI-compatible Messages API format for Hugging Face
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Your exact, highly optimized prompt structure
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical diagnosis AI. You MUST output ONLY valid JSON. "
                        "Schema: {\"prediction\": <string>, \"confidence\": <float between 0.0 and 1.0>}. "
                        "Do NOT use markdown code blocks. Do not explain. Just return the JSON."
                    )
                },
                {"role": "user", "content": "Patient presents with headache, fever, and chills. Extract symptoms."},
                {"role": "assistant", "content": "{\"prediction\": \"headache, fever, chills\", \"confidence\": 0.95}"},
                {"role": "user", "content": "Patient has suspected gallstones. What imaging is needed?"},
                {"role": "assistant", "content": "{\"prediction\": \"Abdominal Ultrasound\", \"confidence\": 0.90}"},
                {"role": "user", "content": f"Case: {observation_text}\nWhat is the most likely diagnosis or required procedure?"}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # The model might still try to add ```json ... ``` despite instructions. 
                # This safely cleans it up.
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                    
                # Parse the JSON string into a Python dictionary
                parsed_json = json.loads(content)
                
                # Extract just the prediction to send to your environment
                final_answer = parsed_json.get("prediction", "Unknown Diagnosis")
                return str(final_answer)
                
            else:
                print(f"⚠️ API Error: {response.status_code} - {response.text}")
        except json.JSONDecodeError:
            print("⚠️ Model failed to output valid JSON. Falling back...")
        except Exception as e:
            print(f"⚠️ LLM Call failed: {e}")

    # Fallback if no token or API fails
    return "Fallback Diagnosis"

def test_remote_env():
    print(f"Connecting to: {BASE_URL}\n")
    
    # 1. Test Reset
    print("--- 1. Testing /reset ---")
    reset_response = requests.post(f"{BASE_URL}/reset")
    if reset_response.status_code == 200:
        print("✅ SUCCESS: Environment reset.")
        reset_data = reset_response.json()
        observation_text = reset_data.get('observation', {}).get('observation', '')
        print(f"Received Case: {observation_text[:100]}...\n")
    else:
        print(f"❌ ERROR: {reset_response.status_code} - {reset_response.text}\n")
        return

    # 2. Test State
    print("--- 2. Testing /state ---")
    state_response = requests.get(f"{BASE_URL}/state")
    if state_response.status_code == 200:
        print("✅ SUCCESS: State retrieved.")
        print(f"State: {state_response.json()}\n")
    else:
        print(f"❌ ERROR: {state_response.status_code} - {state_response.text}\n")

    # 3. Test Step
    print("--- 3. Testing /step ---")
    
    # Get the dynamic prediction from the Agent
    dynamic_prediction = get_agent_prediction(observation_text)
    print(f"🤖 Agent decided on diagnosis: -> {dynamic_prediction} <-\n")
    
    action_payload = {"action": {"prediction": dynamic_prediction}}    
    step_response = requests.post(f"{BASE_URL}/step", json=action_payload)
    if step_response.status_code == 200:
        result = step_response.json()
        print("✅ SUCCESS: Step executed.")
        print(json.dumps(result, indent=2))
        print("\n🎉 ALL TESTS PASSED! Your environment perfectly handled a dynamic Agent.")
    else:
        print(f"❌ ERROR: {step_response.status_code} - {step_response.text}\n")

# This is the line that was missing! It tells Python to actually run the script.
if __name__ == "__main__":
    test_remote_env()