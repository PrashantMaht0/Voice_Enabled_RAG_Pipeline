"""The API server: `uvicorn main:app --host 127.0.0.1 --port 8000` (or `python main.py`).

Every model is loaded at startup (see routes.warmup), so the first question is as fast as any other."""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.guards import install_guards
from src.api.routes import router as api_router, start_index, stop_index, warmup
from src.utils.config import get


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warms every model and checks the index before the server accepts requests, then starts the index worker
    (spec 0008); on shutdown, stops a build in progress."""
    warmup()
    start_index()
    yield
    stop_index()


# /docs, /redoc and /openapi.json only when api.docs_enabled (turn off for a public deployment, spec 0008)
docs = bool(get("api.docs_enabled", True))
app = FastAPI(title="Vocal RAG API", lifespan=lifespan, docs_url="/docs" if docs else None,
              redoc_url="/redoc" if docs else None, openapi_url="/openapi.json" if docs else None)
app.include_router(api_router, prefix="/api")
install_guards(app)

if __name__ == "__main__":
    # Local only: the API has no login. Docker binds 0.0.0.0 inside the container and publishes on 127.0.0.1.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, proxy_headers=False)
