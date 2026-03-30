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

from config import config

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

_CACHE_DIR = config.data_cache_dir / "news"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(ticker: str) -> Path:
    h = hashlib.md5(ticker.upper().encode()).hexdigest()[:10]
    return _CACHE_DIR / f"{ticker.upper().replace('.', '_')}_{h}.json"


def _read_cache(ticker: str) -> Optional[NewsFeed]:
    path = _cache_key(ticker)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        fetched = datetime.fromisoformat(data["fetched_at"])
        if datetime.now() - fetched > timedelta(minutes=config.news.cache_ttl_minutes):
            return None
        articles = [NewsArticle(**a) for a in data["articles"]]
        return NewsFeed(
            ticker=data["ticker"], articles=articles,
            fetched_at=data["fetched_at"],
            source_breakdown=data.get("source_breakdown", {}),
        )
    except Exception:
        return None


def _write_cache(feed: NewsFeed):
    path = _cache_key(feed.ticker)
    path.write_text(json.dumps(feed.to_dict(), indent=2, default=str))


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
            title = item.get("title", "")
            link = item.get("link", "")
            publisher = item.get("publisher", "Unknown")
            pub_time = item.get("providerPublishTime", 0)
            pub_date = datetime.fromtimestamp(pub_time).isoformat() if pub_time else ""
            summary = item.get("summary", title)

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
# Public API
# ---------------------------------------------------------------------------

def fetch_news(ticker: str, company_name: str = "", bypass_cache: bool = False) -> NewsFeed:
    """
    Fetch financial news for a ticker from all available sources.
    Results cached for NEWS_CACHE_TTL minutes.
    """
    if not bypass_cache:
        cached = _read_cache(ticker)
        if cached:
            logger.info(f"Cache hit for {ticker} ({len(cached.articles)} articles)")
            return cached

    source_counts = {}
    all_articles = []

    yf_articles = _fetch_yfinance_news(ticker)
    all_articles.extend(yf_articles)
    if yf_articles:
        source_counts["yfinance"] = len(yf_articles)

    newsapi_articles = _fetch_newsapi(ticker, company_name)
    all_articles.extend(newsapi_articles)
    if newsapi_articles:
        source_counts["newsapi"] = len(newsapi_articles)

    finnhub_articles = _fetch_finnhub(ticker)
    all_articles.extend(finnhub_articles)
    if finnhub_articles:
        source_counts["finnhub"] = len(finnhub_articles)

    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        key = a.title.lower().strip()[:60]
        if key and key not in seen:
            seen.add(key)
            unique.append(a)

    unique.sort(key=lambda a: a.published_date, reverse=True)
    unique = unique[:config.news.max_articles]

    feed = NewsFeed(ticker=ticker.upper(), articles=unique, source_breakdown=source_counts)
    _write_cache(feed)
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
