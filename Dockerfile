FROM python:3.12-slim

WORKDIR /app

# tesseract-ocr: the layoutlmv3 table reader OCRs flagged pages that have no text layer (spec 0004)
# ffmpeg: Gradio converts each streamed spoken sentence to AAC for the browser (spec 0006)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential tesseract-ocr ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU only PyTorch first (same versions as requirements.txt): the CUDA build adds GBs no container uses,
# since Ollama does the GPU work and the embedding and speech models run fine on CPU. torchvision must come
# from the same CPU index: the PyPI build is compiled for CUDA torch and breaks `import timm` (spec 0004).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "$(grep '^torch==' requirements.txt)" "$(grep '^torchvision==' requirements.txt | cut -d' ' -f1)" \
    && pip install --no-cache-dir -r requirements.txt

# Only allowlisted app files reach the image (see .dockerignore)
COPY . .

# Empty folders the app expects; documents, voice files and caches live in volumes
RUN mkdir -p data/raw models /cache

EXPOSE 8000 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
