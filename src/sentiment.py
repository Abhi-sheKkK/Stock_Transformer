"""
Sentiment analysis service.
Primary: FinBERT (ProsusAI/finbert) for financial-domain sentiment.
Fallback: Lightweight keyword-based scoring.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SentimentScore:
    """Sentiment for a single text."""
    text: str
    label: str          # bullish | bearish | neutral
    confidence: float   # 0.0 - 1.0
    score: float        # -1.0 (bearish) to +1.0 (bullish)


@dataclass
class SentimentReport:
    """Aggregated sentiment report for a ticker."""
    ticker: str
    overall_label: str = "neutral"
    overall_score: float = 0.0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    total_analyzed: int = 0
    details: list = field(default_factory=list)
    method: str = "keyword"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "overall_label": self.overall_label,
            "overall_score": round(self.overall_score, 3),
            "distribution": {
                "bullish": self.bullish_count,
                "bearish": self.bearish_count,
                "neutral": self.neutral_count,
            },
            "total_analyzed": self.total_analyzed,
            "method": self.method,
            "details": [asdict(d) if hasattr(d, "__dataclass_fields__") else d for d in self.details],
        }

    @property
    def summary_text(self) -> str:
        pct_bull = (self.bullish_count / max(self.total_analyzed, 1)) * 100
        pct_bear = (self.bearish_count / max(self.total_analyzed, 1)) * 100
        return (
            f"Sentiment for {self.ticker}: {self.overall_label.upper()} "
            f"(score: {self.overall_score:+.2f})\n"
            f"  Bullish: {self.bullish_count}/{self.total_analyzed} ({pct_bull:.0f}%)\n"
            f"  Bearish: {self.bearish_count}/{self.total_analyzed} ({pct_bear:.0f}%)\n"
            f"  Neutral: {self.neutral_count}/{self.total_analyzed}\n"
            f"  Method: {self.method}"
        )


# ---------------------------------------------------------------------------
# FinBERT (lazy-loaded)
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully")
        return _finbert_pipeline
    except Exception as e:
        logger.warning(f"FinBERT unavailable, using keyword fallback: {e}")
        return None


def _score_with_finbert(texts: list) -> list:
    pipe = _load_finbert()
    if pipe is None:
        return _score_with_keywords(texts)

    results = []
    for text in texts:
        try:
            output = pipe(text[:512])[0]
            scores_map = {item["label"].lower(): item["score"] for item in output}
            pos = scores_map.get("positive", 0)
            neg = scores_map.get("negative", 0)
            neu = scores_map.get("neutral", 0)
            composite = pos - neg

            if pos > neg and pos > neu:
                label = "bullish"
            elif neg > pos and neg > neu:
                label = "bearish"
            else:
                label = "neutral"

            results.append(SentimentScore(
                text=text[:100], label=label,
                confidence=round(max(pos, neg, neu), 3),
                score=round(composite, 3),
            ))
        except Exception as e:
            logger.warning(f"FinBERT scoring error: {e}")
            results.append(SentimentScore(text=text[:100], label="neutral", confidence=0.5, score=0.0))
    return results


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

_BULLISH = {
    "surge", "rally", "gain", "rise", "jump", "soar", "beat", "exceed",
    "bullish", "upgrade", "outperform", "growth", "profit", "record",
    "breakout", "momentum", "buy", "strong", "optimistic", "positive",
    "upside", "dividend", "recover", "boom", "expansion",
}

_BEARISH = {
    "drop", "fall", "decline", "crash", "plunge", "loss", "miss", "below",
    "bearish", "downgrade", "underperform", "risk", "warning", "layoff",
    "recession", "sell", "weak", "negative", "downside", "cut", "lawsuit",
    "investigation", "scandal", "debt", "default", "bankruptcy",
}


def _score_with_keywords(texts: list) -> list:
    results = []
    for text in texts:
        words = set(text.lower().split())
        bull = len(words & _BULLISH)
        bear = len(words & _BEARISH)
        total = bull + bear

        if total == 0:
            label, score, conf = "neutral", 0.0, 0.5
        elif bull > bear:
            label, score = "bullish", min(bull / total, 1.0)
            conf = score
        else:
            label, score = "bearish", -min(bear / total, 1.0)
            conf = abs(score)

        results.append(SentimentScore(
            text=text[:100], label=label,
            confidence=round(conf, 3), score=round(score, 3),
        ))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_headlines(headlines: list, use_finbert: bool = True) -> list:
    """Score headlines for financial sentiment."""
    if not headlines:
        return []
    if use_finbert:
        return _score_with_finbert(headlines)
    return _score_with_keywords(headlines)


def get_sentiment_report(ticker: str, headlines: Optional[list] = None) -> SentimentReport:
    """
    Generate a full sentiment report for a ticker.
    Fetches headlines from news service if not provided.
    """
    if headlines is None:
        from .news import get_headlines
        headlines = get_headlines(ticker, max_count=15)

    if not headlines:
        return SentimentReport(ticker=ticker)

    scores = score_headlines(headlines)

    bull = sum(1 for s in scores if s.label == "bullish")
    bear = sum(1 for s in scores if s.label == "bearish")
    neut = sum(1 for s in scores if s.label == "neutral")
    avg_score = sum(s.score for s in scores) / len(scores)

    if avg_score > 0.15:
        overall = "bullish"
    elif avg_score < -0.15:
        overall = "bearish"
    else:
        overall = "neutral"

    method = "finbert" if _finbert_pipeline is not None else "keyword"

    return SentimentReport(
        ticker=ticker, overall_label=overall, overall_score=avg_score,
        bullish_count=bull, bearish_count=bear, neutral_count=neut,
        total_analyzed=len(scores), details=scores, method=method,
    )
