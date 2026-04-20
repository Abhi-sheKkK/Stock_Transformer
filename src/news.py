"""
Financial news aggregation service.
Fetches stock-relevant news from multiple sources with disk caching.
Sources: yfinance (always free), NewsAPI (optional), Finnhub (optional).
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import yfinance as yf

from .config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """A single news article."""
    title: str
    summary: str
    source: str
    published_date: str
    url: str
    ticker: str
    relevance: str = "direct"
    confidence: float = 0.0
    source_count: int = 1
    verified_sources: list = field(default_factory=list)
    truth_score: float = 0.0
    pillars: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsFeed:
    """Collection of news articles for a ticker."""
    ticker: str
    articles: list = field(default_factory=list)
    fetched_at: str = ""
    source_breakdown: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "fetched_at": self.fetched_at,
            "total_articles": len(self.articles),
            "source_breakdown": self.source_breakdown,
            "articles": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.articles],
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_DIR_RAW = config.data_cache_dir / "news" / "raw"
_CACHE_DIR_INTEL = config.data_cache_dir / "news" / "intelligence"
_CACHE_DIR_RAW.mkdir(parents=True, exist_ok=True)
_CACHE_DIR_INTEL.mkdir(parents=True, exist_ok=True)


def _cache_key(ticker: str, type: str = "intel") -> Path:
    h = hashlib.md5(ticker.upper().encode()).hexdigest()[:10]
    dir = _CACHE_DIR_INTEL if type == "intel" else _CACHE_DIR_RAW
    return dir / f"{ticker.upper().replace('.', '_')}_{h}.json"


def _read_cache(ticker: str) -> Optional[NewsFeed]:
    """Read from the Intelligence cache."""
    path = _cache_key(ticker, type="intel")
    # Also check legacy root path for old cache transition
    legacy_path = config.data_cache_dir / "news" / path.name
    
    selected_path = path if path.exists() else (legacy_path if legacy_path.exists() else None)
    
    if not selected_path:
        return None
        
    try:
        data = json.loads(selected_path.read_text())
        fetched = datetime.fromisoformat(data["fetched_at"])
        if datetime.now() - fetched > timedelta(minutes=config.news.cache_ttl_minutes):
            return None
        
        # Handle migration of cached data without confidence fields
        articles = []
        for a in data["articles"]:
            if "confidence" not in a: a["confidence"] = 0.33
            if "source_count" not in a: a["source_count"] = 1
            if "verified_sources" not in a: a["verified_sources"] = [a.get("source", "Unknown")]
            articles.append(NewsArticle(**a))
            
        return NewsFeed(
            ticker=data["ticker"], articles=articles,
            fetched_at=data["fetched_at"],
            source_breakdown=data.get("source_breakdown", {}),
        )
    except Exception:
        return None


def _write_cache(feed: NewsFeed, raw_articles: list):
    """Save both Intelligence (Top 10) and Raw (All) caches."""
    # 1. Save Intelligence Cache
    intel_path = _cache_key(feed.ticker, type="intel")
    intel_path.write_text(json.dumps(feed.to_dict(), indent=2, default=str))
    
    # 2. Save Raw Cache
    raw_path = _cache_key(feed.ticker, type="raw")
    raw_data = {
        "ticker": feed.ticker,
        "fetched_at": feed.fetched_at,
        "total_raw_articles": len(raw_articles),
        "articles": [a.to_dict() if hasattr(a, "to_dict") else a for a in raw_articles]
    }
    raw_path.write_text(json.dumps(raw_data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _fetch_yfinance_news(ticker: str) -> list:
    """Fetch news from yfinance (always available, no key needed)."""
    articles = []
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
        for item in news_items[:config.news.max_articles]:
            # Support both old flat structure and new nested 'content' structure
            content = item.get("content", item)
            
            title = content.get("title", "")
            # New format uses clickThroughUrl, old format uses link
            link = content.get("clickThroughUrl", {}).get("url", content.get("link", ""))
            # New format uses provider.displayName
            publisher = content.get("provider", {}).get("displayName", content.get("publisher", "Unknown"))
            # New format uses pubDate
            pub_time = content.get("providerPublishTime", 0)
            if not pub_time and "pubDate" in content:
                # pubDate is ISO string like 2024-04-20T11:38:47Z
                pub_date_str = content["pubDate"]
                articles.append(NewsArticle(
                    title=title, summary=content.get("summary", title), source=publisher,
                    published_date=pub_date_str, url=link, ticker=ticker,
                ))
                continue
            
            pub_date = datetime.fromtimestamp(pub_time).isoformat() if pub_time else ""
            summary = content.get("summary", title)

            articles.append(NewsArticle(
                title=title, summary=summary, source=publisher,
                published_date=pub_date, url=link, ticker=ticker,
            ))
    except Exception as e:
        logger.warning(f"yfinance news failed for {ticker}: {e}")
    return articles


def _fetch_newsapi(ticker: str, company_name: str = "") -> list:
    """Fetch from NewsAPI.org (requires free API key)."""
    if not config.news.news_api_key:
        return []
    articles = []
    query = company_name or ticker.split(".")[0]
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": f"{query} stock",
                "apiKey": config.news.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": config.news.max_articles,
            },
            timeout=10,
        )
        resp.raise_for_status()
        for item in resp.json().get("articles", []):
            articles.append(NewsArticle(
                title=item.get("title", ""),
                summary=item.get("description", ""),
                source=item.get("source", {}).get("name", "Unknown"),
                published_date=item.get("publishedAt", ""),
                url=item.get("url", ""), ticker=ticker,
            ))
    except Exception as e:
        logger.warning(f"NewsAPI failed for {ticker}: {e}")
    return articles


def _fetch_finnhub(ticker: str) -> list:
    """Fetch from Finnhub (requires free API key)."""
    if not config.news.finnhub_api_key:
        return []
    articles = []
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        # Finnhub uses base ticker without exchange suffix
        symbol = ticker.split(".")[0]
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": symbol, "from": week_ago, "to": today, "token": config.news.finnhub_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        for item in resp.json()[:config.news.max_articles]:
            articles.append(NewsArticle(
                title=item.get("headline", ""),
                summary=item.get("summary", ""),
                source=item.get("source", "Unknown"),
                published_date=datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                url=item.get("url", ""), ticker=ticker,
            ))
    except Exception as e:
        logger.warning(f"Finnhub failed for {ticker}: {e}")
    return articles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for cross-source matching (lowercase, alphanumeric only)."""
    import re
    if not text: return ""
    # Lowercase and keep only alphanumeric chars
    normalized = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized[:80]


# ---------------------------------------------------------------------------
# Truth Engine (The Analytical Heart)
# ---------------------------------------------------------------------------

class TruthEngine:
    """
    Analyzes news clusters across 4 pillars:
    1. Content Consistency (Semantic Similarity)
    2. Source Credibility (Weighting)
    3. Temporal Convergence (Breaking News Detection)
    4. Contradiction Check (Negation Detection)
    """
    
    @staticmethod
    def _calculate_consistency(texts: list) -> float:
        """Pillar 1: Cosine Similarity between titles/summaries."""
        if len(texts) < 2: return 0.2 # Lower baseline for single source
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf)
            # Average similarity between all pairs
            n = len(texts)
            avg_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            return float(avg_sim)
        except Exception:
            return 0.5

    @staticmethod
    def _calculate_credibility(sources: list) -> float:
        """Pillar 2: Weighted source reliability."""
        weights = config.news.truth_engine
        source_map = {
            "yahoofinance.com": weights.weight_yfinance,
            "yahoo finance": weights.weight_yfinance,
            "yfinance": weights.weight_yfinance,
            "finnhub": weights.weight_finnhub,
            "newsapi": weights.weight_newsapi,
        }
        
        score = 0.0
        for s in sources:
            score = max(score, source_map.get(s.lower(), 0.5))
        
        # Bonus for diversity (multi-source validation)
        if len(sources) > 1:
            score = min(score + 0.1 * (len(sources) - 1), 1.0)
        return score

    @staticmethod
    def _calculate_temporal(dates: list) -> float:
        """Pillar 3: Convergence in time (Break within 15 min window)."""
        if len(dates) < 2: return 0.5
        try:
            parsed_dates = []
            for d in dates:
                if not d: continue
                # Handle ISO format
                parsed_dates.append(datetime.fromisoformat(d.replace("Z", "")))
            
            if not parsed_dates: return 0.5
            
            # Find spread in minutes
            spread = (max(parsed_dates) - min(parsed_dates)).total_seconds() / 60
            
            if spread <= 15: return 1.0
            if spread <= 60: return 0.8
            if spread <= 1440: return 0.5 # Same day
            return 0.3
        except Exception:
            return 0.5

    @staticmethod
    def _detect_contradictions(texts: list) -> float:
        """Pillar 4: Explicit negation search (Confirmed vs Denied)."""
        NEGATIONS = ["deny", "denied", "reject", "rejected", "false", "refute", "incorrect", "cancel", "fail", "no"]
        CONFIRMATIONS = ["confirm", "confirmed", "true", "verify", "yes", "deal", "merge", "win"]
        
        text_blob = " ".join(texts).lower()
        has_neg = any(word in text_blob for word in NEGATIONS)
        has_conf = any(word in text_blob for word in CONFIRMATIONS)
        
        # If both patterns are present in the same cluster, it highlights a contradiction
        if has_neg and has_conf:
            return config.news.truth_engine.p_contradiction
        return 0.0

    @classmethod
    def score_cluster(cls, cluster: list) -> tuple:
        """
        Calculates the final Truth Score (S) for a cluster of news articles.
        Returns (final_score, pillar_breakdown)
        """
        texts = [f"{a.title} {a.summary}" for a in cluster]
        sources = [a.source for a in cluster]
        dates = [a.published_date for a in cluster]
        
        # Calculate individual pillars
        c_score = cls._calculate_consistency(texts)
        w_score = cls._calculate_credibility(sources)
        t_score = cls._calculate_temporal(dates)
        penalty = cls._detect_contradictions(texts)
        
        # Formula: S = (w1*C) + (w2*W) + (w3*T) - Penalty
        eng = config.news.truth_engine
        s = (eng.w_consistency * c_score) + \
            (eng.w_credibility * w_score) + \
            (eng.w_temporal * t_score) - penalty
        
        final_score = max(min(s, 1.0), 0.0)
        
        breakdown = {
            "consistency": round(c_score, 2),
            "credibility": round(w_score, 2),
            "temporal": round(t_score, 2),
            "contradiction_penalty": round(penalty, 2)
        }
        
        return final_score, breakdown


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(ticker: str, company_name: str = "", bypass_cache: bool = False) -> NewsFeed:
    """
    Fetch financial news for a ticker from all available sources.
    Uses cross-source consensus to calculate confidence scores.
    """
    if not bypass_cache:
        cached = _read_cache(ticker)
        if cached:
            logger.info(f"Cache hit for {ticker} ({len(cached.articles)} articles)")
            return cached

    active_sources = 1 # yfinance is always active
    if config.news.news_api_key: active_sources += 1
    if config.news.finnhub_api_key: active_sources += 1

    source_counts = {}
    raw_articles = []

    # 1. Fetch from all sources
    yf_articles = _fetch_yfinance_news(ticker)
    raw_articles.extend(yf_articles)
    if yf_articles: source_counts["yfinance"] = len(yf_articles)

    newsapi_articles = _fetch_newsapi(ticker, company_name)
    raw_articles.extend(newsapi_articles)
    if newsapi_articles: source_counts["newsapi"] = len(newsapi_articles)

    finnhub_articles = _fetch_finnhub(ticker)
    raw_articles.extend(finnhub_articles)
    if finnhub_articles: source_counts["finnhub"] = len(finnhub_articles)

    # 2. Cluster by similarity (normalized titles)
    clusters = {} # key: normalized_title -> list of NewsArticle
    for art in raw_articles:
        key = _normalize_text(art.title)
        if not key: continue
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(art)

    # 3. Calculate Confidence & Consensus using Truth Engine
    consolidated = []
    for key, cluster in clusters.items():
        truth_score, breakdown = TruthEngine.score_cluster(cluster)
        
        # Pick the representative article (one with longest summary preferably)
        representative = max(cluster, key=lambda a: len(a.summary) if a.summary else 0)
        
        unique_sources = list(set(a.source for a in cluster))
        representative.confidence = round(truth_score, 2)
        representative.truth_score = round(truth_score, 2)
        representative.source_count = len(unique_sources)
        representative.verified_sources = unique_sources
        representative.pillars = breakdown
        
        consolidated.append(representative)

    # 4. Rank by Truth Score (Primary) and Date (Secondary)
    consolidated.sort(key=lambda a: (a.truth_score, a.published_date), reverse=True)
    
    # Top 10 as requested
    final_articles = consolidated[:10]

    feed = NewsFeed(ticker=ticker.upper(), articles=final_articles, source_breakdown=source_counts)
    _write_cache(feed, raw_articles=raw_articles)
    return feed


def get_headlines(ticker: str, max_count: int = 10) -> list:
    """Return just headline strings for sentiment scoring."""
    feed = fetch_news(ticker)
    return [a.title for a in feed.articles[:max_count]]


def get_news_summary_text(ticker: str, max_count: int = 5) -> str:
    """Formatted text summary of recent news for LLM consumption."""
    feed = fetch_news(ticker)
    if not feed.articles:
        return f"No recent news found for {ticker}."

    lines = [f"Recent news for {ticker} (as of {feed.fetched_at[:10]}):\n"]
    for i, article in enumerate(feed.articles[:max_count], 1):
        date_str = article.published_date[:10] if article.published_date else "N/A"
        lines.append(f"{i}. [{date_str}] {article.title}")
        if article.summary and article.summary != article.title:
            lines.append(f"   {article.summary[:200]}")
        lines.append(f"   Source: {article.source}\n")

    return "\n".join(lines)
