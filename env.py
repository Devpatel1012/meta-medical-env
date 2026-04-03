import json
import random
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class MedicalEnv:
    def __init__(self):
        dataset_path = os.path.join(os.path.dirname(__file__), "medical_hackathon_dataset.json")
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        
        hf_token = os.getenv("HF_TOKEN")
        self.judge_model = os.getenv("JUDGE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.client = InferenceClient(api_key=hf_token)
        
        # --- NEW: Load the Local Embedding Model for Semantic Grading ---
        print("Loading local embedding model (this takes a few seconds the first time)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.current_case = None

    def reset(self):
        self.current_case = random.choice(self.dataset)
        question_text = self.current_case.get("question", "")
        for key in ["distractors", "distractor_1", "distractor_2", "symptoms", "irrelevant_detail_1", "irrelevant_detail_2"]:
            if key in self.current_case:
                val = self.current_case[key]
                if isinstance(val, list):
                    question_text += " " + " ".join(val)
                else:
                    question_text += " " + str(val)
                    
        self.current_case["full_question"] = question_text.strip()
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
        
        done = True 
        info = {
            "difficulty": difficulty,
            "ground_truth": ground_truth,
            "score": reward
        }
        return None, reward, done, info

    # --- THE NEW SEMANTIC GRADERS ---

    def _grade_easy_semantic(self, action, ground_truth_list):
        """Grades symptom extraction using vector similarity."""
        if not isinstance(ground_truth_list, list):
            ground_truth_list = [ground_truth_list]
            
        # Combine lists into single strings for semantic comparison
        truth_str = ", ".join([str(s) for s in ground_truth_list])
        
        # Calculate Cosine Similarity
        embeddings = self.embedding_model.encode([action, truth_str])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Apply Threshold (If it's 80% similar semantically, it's a pass)
        return 1.0 if sim_score >= 0.80 else 0.0

    def _grade_medium_semantic(self, action, ground_truth_str):
        """Grades medical tests using vector similarity."""
        embeddings = self.embedding_model.encode([str(action), str(ground_truth_str)])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Print the math so you can debug your thresholds!
        print(f"   -> [Vector Debug] Similarity Score: {sim_score:.3f}")
        
        # Lowered threshold to 0.75 to be slightly more forgiving
        return 1.0 if sim_score >= 0.75 else 0.0

    def _grade_hard(self, action, ground_truth_str, question):
        """Grades differential diagnosis using the Hugging Face API as a judge."""
        prompt = (
            f"You are a medical grading system. Evaluate the Agent's diagnosis.\n"
            f"Patient Case: {question}\n"
            f"Gold Standard Diagnosis: {ground_truth_str}\n"
            f"Agent's Diagnosis: {action}\n\n"
            f"Does the Agent's diagnosis substantially match the clinical intent of the Gold Standard? "
            f"Output ONLY a single float between 0.0 and 1.0. Output nothing else."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_str = response.choices[0].message.content.strip()
            match = re.search(r"0\.\d+|1\.0|0|1", score_str)
            return float(match.group()) if match else 0.0
        except Exception as e:
            print(f"Judge API failed: {e}")
            return 0.0