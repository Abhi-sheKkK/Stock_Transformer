"""
News and sentiment API routes.
"""

from fastapi import APIRouter, HTTPException, Query

from api.services.news import fetch_news, get_news_summary_text
from api.services.sentiment import get_sentiment_report

router = APIRouter(prefix="/news", tags=["News & Sentiment"])


@router.get("/{ticker}")
async def get_news(
    ticker: str,
    max_articles: int = Query(default=10, ge=1, le=50),
    include_sentiment: bool = Query(default=True),
):
    """
    Get recent news for a ticker with optional sentiment analysis.
    
    Example: GET /news/TCS.NS?max_articles=5&include_sentiment=true
    """
    try:
        feed = fetch_news(ticker)
        response = feed.to_dict()
        
        # Trim articles
        response["articles"] = response["articles"][:max_articles]
        response["total_articles"] = len(response["articles"])

        if include_sentiment and feed.articles:
            headlines = [a.title for a in feed.articles[:max_articles]]
            sentiment = get_sentiment_report(ticker, headlines=headlines)
            response["sentiment"] = sentiment.to_dict()
            response["sentiment"]["summary"] = sentiment.summary_text
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {str(e)}")


@router.get("/{ticker}/summary")
async def get_news_text_summary(ticker: str):
    """
    Get a plain text summary of recent news (useful for LLM consumption).
    
    Example: GET /news/RELIANCE.NS/summary
    """
    try:
        return {"ticker": ticker, "summary": get_news_summary_text(ticker)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/sentiment")
async def get_sentiment(ticker: str):
    """
    Get sentiment analysis for a ticker's recent news.
    
    Example: GET /news/INFY.NS/sentiment
    """
    try:
        report = get_sentiment_report(ticker)
        result = report.to_dict()
        result["summary"] = report.summary_text
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
