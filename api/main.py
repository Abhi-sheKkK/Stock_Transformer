"""
FastAPI application entry point.
AI Financial Intelligence System backend.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import market, news, predict, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 AI Financial Intelligence System starting...")
    print("   Loading services...")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="AI Financial Intelligence System",
    description="Transformer-based stock prediction with AI-powered reasoning via Llama 3",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import requests
    ollama_status = "unknown"
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
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
        "ollama": ollama_status,
    }

