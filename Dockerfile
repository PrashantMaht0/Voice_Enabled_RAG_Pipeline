# Two stages: the compilers live only in the builder; the final image runs as a non root user.

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# CPU only PyTorch first (same versions as requirements.txt): the CUDA build adds GBs no container uses,
# since Ollama does the GPU work and the embedding and speech models run fine on CPU. torchvision must come
# from the same CPU index: the PyPI build is compiled for CUDA torch and breaks `import timm`.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "$(grep '^torch==' requirements.txt)" "$(grep '^torchvision==' requirements.txt | cut -d' ' -f1)" \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim

# tesseract-ocr: the layoutlmv3 table reader OCRs flagged pages that have no text layer
# ffmpeg: Gradio converts each streamed spoken sentence to AAC for the browser
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 10001 --create-home --home-dir /home/vrag vrag

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/vrag \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MPLCONFIGDIR=/cache/matplotlib \
    FLASHRANK_CACHE_DIR=/cache/flashrank

WORKDIR /app
# Only allowlisted app files reach the image (see .dockerignore)
COPY --chown=vrag:vrag . .

# Folders the app writes, owned by the app user. A fresh named volume copies this ownership on first mount;
# volumes made by an older root image need the one time chown in the README.
RUN mkdir -p data/raw vectordb models /cache/huggingface /cache/torch /cache/matplotlib /cache/flashrank \
    && chown -R vrag:vrag /app/data /app/vectordb /app/models /cache

USER vrag

EXPOSE 8000 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers"]
