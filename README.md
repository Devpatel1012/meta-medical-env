# Clinical Diagnosis Env

A medical triage OpenEnv environment evaluating an AI agent's ability to extract symptoms, recommend imaging, and provide differential diagnoses.

## Spaces
- **Observation Space**: `MedicalObservation(text: str)` - Contains the patient case text.
- **Action Space**: `MedicalAction(prediction: str)` - The agent's predicted diagnosis or recommended action.

## Setup
1. Define `.env` with `HF_TOKEN`, `API_BASE_URL`, and `AGENT_MODELS`.
2. Run `pip install -r requirements.txt`.
3. Run `python inference.py` to evaluate the baseline agent.