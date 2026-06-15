"""
FastAPI application entry point.
AI Financial Intelligence System backend.

On startup, downloads the latest model weights from Supabase Storage
so the deployed server always uses the most recently trained checkpoint.
"""

import os
import urllib.request
import requests
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import market, news, predict, analyze, research

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
_MODEL_PATH = Path(__file__).resolve().parent.parent / "best_model.pth"

# Supabase public bucket URL for trained weights
SUPABASE_MODEL_URL = os.getenv(
    "SUPABASE_MODEL_URL",
    "https://oqyqnndynmgthsmmzhpy.supabase.co/storage/v1/object/public/models/best_model.pth"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print(" AI Financial Intelligence System starting...")
    print("   Loading services...")

    # Download latest weights from Supabase if no local file exists
    if not _MODEL_PATH.exists():
        print(f"   ⬇ Downloading model weights from Supabase...")
        try:
            urllib.request.urlretrieve(SUPABASE_MODEL_URL, str(_MODEL_PATH))
            print(f"    Weights downloaded to {_MODEL_PATH}")
        except Exception as e:
            print(f"    Could not download weights: {e}")
            print(f"   The server will still start but /predict may fail.")
    else:
        print(f" Local model weights found at {_MODEL_PATH}")

    yield
    print(" Shutting down...")


app = FastAPI(
    title="AI Financial Intelligence System",
    description="Transformer-based stock prediction with AI-powered reasoning via Llama 3",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(market.router)
app.include_router(news.router)
app.include_router(predict.router)
app.include_router(analyze.router)
app.include_router(research.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from src.config import config
    ollama_status = "unknown"
    try:
        resp = requests.get(f"{config.llm.base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            ollama_status = f"running (models: {', '.join(models) if models else 'none'})"
        else:
            ollama_status = "running (no models)"
    except Exception:
        ollama_status = "not reachable"

    return {
        "status": "healthy",
        "service": "AI Financial Intelligence System",
        "model_loaded": _MODEL_PATH.exists(),
        "ollama": ollama_status,
    }


# Serve frontend static files
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
