"""
Analysis API routes.
Full AI-powered analysis combining prediction + reasoning + strategy.
(Fully implemented in Phase 2 with Ollama/Llama 3)
"""

from fastapi import APIRouter, HTTPException

from src.features import get_market_snapshot
from src.news import fetch_news, get_news_summary_text
from src.sentiment import get_sentiment_report

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post("/{ticker}")
async def full_analysis(ticker: str):
    """
    Generate a comprehensive AI analysis for a stock.
    Combines: market snapshot + news + sentiment + prediction + AI reasoning.

    Example: POST /analyze/RELIANCE.NS
    """
    try:
        # Gather all data
        snapshot = get_market_snapshot(ticker)
        if "error" in snapshot:
            raise HTTPException(status_code=404, detail=snapshot["error"])

        news_feed = fetch_news(ticker)
        sentiment = get_sentiment_report(ticker)

        response = {
            "ticker": ticker.upper(),
            "market_snapshot": snapshot,
            "news": {
                "total_articles": len(news_feed.articles),
                "headlines": [a.title for a in news_feed.articles[:5]],
                "news_summary": get_news_summary_text(ticker),
            },
            "sentiment": sentiment.to_dict(),
            "ai_reasoning": {
                "status": "available_after_phase2",
                "message": "Full AI reasoning (prediction rationale, strategy, news impact) will be powered by Llama 3 via Ollama in Phase 2.",
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
