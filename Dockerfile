FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU only PyTorch first (same version as requirements.txt): the CUDA build adds GBs no container uses,
# since Ollama does the GPU work and the embedding and speech models run fine on CPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "$(grep '^torch==' requirements.txt)" \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Only allowlisted app files reach the image (see .dockerignore)
COPY . .

# Empty folders the app expects; documents, voice files and caches live in volumes
RUN mkdir -p data/raw models /cache

EXPOSE 8000 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
