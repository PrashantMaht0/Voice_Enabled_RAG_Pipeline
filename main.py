"""The API server: `uvicorn main:app --host 127.0.0.1 --port 8000` (or `python main.py`).

Every model is loaded at startup (see routes.warmup), so the first question is as fast as any other."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.routes import router as api_router, warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warms every model and checks the index before the server accepts requests."""
    warmup()
    yield


app = FastAPI(title="Vocal RAG API", lifespan=lifespan)
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    # Local only: the API has no login. Docker binds 0.0.0.0 inside the container and publishes on 127.0.0.1.
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
