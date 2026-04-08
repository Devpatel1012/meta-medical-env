import json
import random
import os
import re
import string
import warnings
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openenv.core.env_server import Environment
from dotenv import load_dotenv
from argparse import Namespace

from models import MedicalObservation, MedicalAction
from pydantic import BaseModel



os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 
warnings.filterwarnings("ignore")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

load_dotenv()

def normalize_text(text):
    if isinstance(text, list):
        text = ", ".join([str(s) for s in text])
    text = str(text).lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))


class EnvState(BaseModel):
    episode_id: str
    step_count: int
    current_difficulty: str
    

class FinalEnvironment(Environment):
    def __init__(self):
        dataset_path = os.path.join(os.path.dirname(__file__), "medical_hackathon_dataset.json")
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        token = os.getenv("HF_TOKEN", "").strip('"').strip("'")
        self.judge_client = OpenAI(base_url=api_base, api_key=token)
        
        judge_models_str = os.getenv("JUDGE_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.judge_models = [m.strip() for m in judge_models_str.split(",")]
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.current_case = None
        self.step_count = 0
        self.episode_id = "0"

    def reset(self, target_difficulty=None) -> MedicalObservation:
        self.step_count = 0
        if target_difficulty:
            filtered = [c for c in self.dataset if c.get("difficulty_level") == target_difficulty]
            self.current_case = random.choice(filtered) if filtered else random.choice(self.dataset)
        else:
            self.current_case = random.choice(self.dataset)
            
        q_text = self.current_case.get("question", "")
        keys_to_add = ["distractors", "distractor_1", "distractor_2", "symptoms", "irrelevant_detail_1", "irrelevant_detail_2"]
        
        for k in keys_to_add:
            if k in self.current_case:
                val = self.current_case[k]
                q_text += " " + (" ".join(val) if isinstance(val, list) else str(val))
                    
        self.current_case["full_question"] = q_text.strip()
        return MedicalObservation(observation=self.current_case["full_question"])

    def step(self, action: MedicalAction):
        if self.current_case is None:
            self.reset()
        
        self.step_count += 1
        difficulty = self.current_case.get("difficulty_level", "Unknown")
        ground_truth = self.current_case.get("answer", "")
        question = self.current_case.get("full_question", "")
        
        prediction_str = action.prediction
        
        print(f"\n[EVALUATION] Ground Truth Answer: {ground_truth}")
        print(f"[EVALUATION] Agent Generated Answer: {prediction_str}\n")
        
        reward = 0.0
        if difficulty == "Easy":
            reward = self._grade_easy_semantic(prediction_str, ground_truth)
        elif difficulty == "Medium":
            reward = self._grade_medium_semantic(prediction_str, ground_truth)
        elif difficulty == "Hard":
            reward = self._grade_hard(prediction_str, ground_truth, question)
        
        info = {
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "score": reward
        }
        
        return MedicalObservation(
            observation="Episode finished", 
            reward=reward, 
            done=True,
            info=info
        )

    @property
    def state(self):
        """Returns the current state of the environment as a Pydantic model."""
        return EnvState(
            episode_id=str(getattr(self, "episode_id", "0")),
            step_count=int(getattr(self, "step_count", 0)),
            current_difficulty=str(self.current_case.get("difficulty_level", "None") if self.current_case else "None")
        )

    def _grade_easy_semantic(self, action, ground_truth):
        clean_action = normalize_text(action)
        clean_truth = normalize_text(ground_truth)
        embeddings = self.embedding_model.encode([clean_action, clean_truth], show_progress_bar=False)
        sim_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        if sim_score > 0.95: return 1.0
        return sim_score if sim_score >= 0.4 else 0.0

    def _grade_medium_semantic(self, action, ground_truth):
        clean_action = normalize_text(action)
        clean_truth = normalize_text(ground_truth)
        embeddings = self.embedding_model.encode([clean_action, clean_truth], show_progress_bar=False)
        sim_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        if sim_score > 0.95: return 1.0
        return sim_score if sim_score >= 0.4 else 0.0

    def _grade_hard(self, action_str, ground_truth_str, question):
        prompt = (
            f"Evaluate the Agent's diagnosis.\n"
            f"Case: {question}\n"
            f"True Diagnosis: {ground_truth_str}\n"
            f"Agent Diagnosis: {action_str}\n\n"
            f"Does it match? Output ONLY a float between 0.0 and 1.0."
        )
        for model in self.judge_models:
            try:
                response = self.judge_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10
                )
                score_str = response.choices[0].message.content.strip()
                match = re.search(r"0\.\d+|1\.0|0|1", score_str)
                if match:
                    return float(match.group())
                return 0.0
            except Exception:
                continue
        return 0.0