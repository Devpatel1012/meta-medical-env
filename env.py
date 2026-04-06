import json
import random
import os
import re
import string
import warnings
import logging

# 1. Brutally silence all background library noise for a clean terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 
warnings.filterwarnings("ignore")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

def normalize_text(text):
    """Lowercases and removes punctuation for fairer similarity scoring."""
    if isinstance(text, list):
        text = ", ".join([str(s) for s in text])
    text = str(text).lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))

class MedicalEnv:
    def __init__(self):
        dataset_path = os.path.join(os.path.dirname(__file__), "medical_hackathon_dataset.json")
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        
        # --- Docker-Proofing the Token ---
        raw_token = os.getenv("HF_TOKEN", "")
        clean_token = raw_token.strip('"').strip("'")
        self.client = InferenceClient(token=clean_token)
        
        # --- Docker-Proofing the Judge Models ---
        judge_models_str = os.getenv("JUDGE_MODELS", "meta-llama/Meta-Llama-3-8B-Instruct")
        judge_models_str = judge_models_str.strip('"').strip("'")
        self.judge_models = [m.strip() for m in judge_models_str.split(",")]
        
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.current_case = None

    def reset(self, target_difficulty=None):
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
        return self.current_case["full_question"]

    def step(self, action):
        difficulty = self.current_case.get("difficulty_level", "Unknown")
        ground_truth = self.current_case.get("answer", "")
        question = self.current_case.get("full_question", "")
        
        reward = 0.0
        if difficulty == "Easy":
            reward = self._grade_easy_semantic(action, ground_truth)
        elif difficulty == "Medium":
            reward = self._grade_medium_semantic(action, ground_truth)
        elif difficulty == "Hard":
            reward = self._grade_hard(action, ground_truth, question)
        
        info = {
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "score": reward
        }
        return None, reward, True, info

    def _grade_easy_semantic(self, action, ground_truth):
        clean_action = normalize_text(action)
        clean_truth = normalize_text(ground_truth)
        
        # Pass show_progress_bar=False to stop the terminal spam
        embeddings = self.embedding_model.encode([clean_action, clean_truth], show_progress_bar=False)
        sim_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        
        if sim_score > 0.95: return 1.0
        return sim_score if sim_score >= 0.4 else 0.0

    def _grade_medium_semantic(self, action, ground_truth):
        clean_action = normalize_text(action)
        clean_truth = normalize_text(ground_truth)
        
        # Pass show_progress_bar=False to stop the terminal spam
        embeddings = self.embedding_model.encode([clean_action, clean_truth], show_progress_bar=False)
        sim_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        
        if sim_score > 0.95: return 1.0
        return sim_score if sim_score >= 0.4 else 0.0

    def _grade_hard(self, action, ground_truth_str, question):
        prompt = (
            f"Evaluate the Agent's diagnosis.\n"
            f"Case: {question}\n"
            f"True Diagnosis: {ground_truth_str}\n"
            f"Agent Diagnosis: {action}\n\n"
            f"Does it match? Output ONLY a float between 0.0 and 1.0."
        )
        
        for model in self.judge_models:
            try:
                response = self.client.chat.completions.create(
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
            except Exception as e:
                print(f"[Judge Error] {model} failed:", str(e))
                continue
                
        return 0.0