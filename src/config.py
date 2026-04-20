"""
Centralized configuration for the AI Financial Intelligence System.
Loads settings from .env file and provides typed access to all config values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Project root calculation (one level up from src/)
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent

# Load .env from project root
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    """Ollama / Llama 3 configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3:8b")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))


@dataclass
class TruthEngineConfig:
    """Weights and parameters for the 4-pillar news scoring."""
    w_consistency: float = 0.4
    w_credibility: float = 0.3
    w_temporal: float = 0.3
    p_contradiction: float = 0.8  # Penalty factor
    
    # Pillar 2 Weights
    weight_yfinance: float = 1.0
    weight_finnhub: float = 0.9
    weight_newsapi: float = 0.7


@dataclass
class NewsConfig:
    """News and data ingestion configuration."""
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    cache_ttl_minutes: int = int(os.getenv("NEWS_CACHE_TTL", "15"))
    max_articles: int = int(os.getenv("NEWS_MAX_ARTICLES", "20"))
    truth_engine: TruthEngineConfig = field(default_factory=TruthEngineConfig)


@dataclass
class ModelConfig:
    """Transformer model hyperparameters."""
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    hidden_dim: int = 128
    dropout_rate: float = 0.3
    seq_length: int = 100
    out_seq_len: int = 5


@dataclass
class FeatureFlags:
    """Toggle system capabilities on/off."""
    enable_news: bool = os.getenv("ENABLE_NEWS", "true").lower() == "true"
    enable_sentiment: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    enable_ai_reasoning: bool = os.getenv("ENABLE_AI_REASONING", "true").lower() == "true"
    enable_rag: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
    enable_explainability: bool = os.getenv("ENABLE_EXPLAINABILITY", "true").lower() == "true"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flags: FeatureFlags = field(default_factory=FeatureFlags)

    # Paths
    project_root: Path = _PROJECT_ROOT
    models_dir: Path = _PROJECT_ROOT / "models"
    data_cache_dir: Path = _PROJECT_ROOT / ".cache"

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    def __post_init__(self):
        self.models_dir.mkdir(exist_ok=True)
        self.data_cache_dir.mkdir(exist_ok=True)


# Singleton
config = AppConfig()
