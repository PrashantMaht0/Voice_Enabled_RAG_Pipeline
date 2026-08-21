from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.routes import router as api_router, warmup

@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()
    yield

app = FastAPI(title="Vocal RAG API", lifespan=lifespan)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
