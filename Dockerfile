# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies
# We include --no-cache-dir to keep the Docker image size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files (inference.py, env.py, dataset, etc.)
COPY . .

# 6. Pre-download the Sentence Transformer model during the build phase!
# This ensures the model is baked into the image, so it doesn't have to 
# download every time the judges run your container.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 7. Define the command to run your evaluation script when the container starts
CMD ["python", "inference.py"]