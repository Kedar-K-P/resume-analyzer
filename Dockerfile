FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some sklearn builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project (flat structure — all py files in /app)
COPY . .

# FIX: Create required directories
# All scripts, dataset, model, and evaluation live flat under /app
RUN mkdir -p /app/data /app/evaluation

# Step 1: Generate dataset into /app/data/
RUN python dataset_generator.py

# Step 2: Train NER model (saves ner_model.pkl to /app/, eval to /app/evaluation/)
RUN python ner_model_trainer.py

EXPOSE 8000

# FIX: Run uvicorn from /app so all relative imports resolve correctly
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
