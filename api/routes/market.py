"""
Market snapshot API routes.
"""

import traceback
from fastapi import APIRouter, HTTPException, Query

from src.features import get_market_snapshot

router = APIRouter(prefix="/market", tags=["Market"])

# Supported tickers with company names for search suggestions
TICKERS = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', 'NVDA': 'NVIDIA Corp.',
    'AVGO': 'Broadcom Inc.', 'ORCL': 'Oracle Corp.', 'AMD': 'AMD Inc.',
    'CRM': 'Salesforce Inc.', 'GOOGL': 'Alphabet Inc.', 'META': 'Meta Platforms',
    'NFLX': 'Netflix Inc.', 'DIS': 'Walt Disney Co.', 'TMUS': 'T-Mobile US',
    'CMCSA': 'Comcast Corp.', 'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America',
    'MS': 'Morgan Stanley', 'GS': 'Goldman Sachs', 'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.', 'AXP': 'American Express', 'LLY': 'Eli Lilly',
    'UNH': 'UnitedHealth Group', 'JNJ': 'Johnson & Johnson', 'MRK': 'Merck & Co.',
    'ABBV': 'AbbVie Inc.', 'TMO': 'Thermo Fisher', 'ISRG': 'Intuitive Surgical',
    'AMZN': 'Amazon.com Inc.', 'TSLA': 'Tesla Inc.', 'WMT': 'Walmart Inc.',
    'COST': 'Costco Wholesale', 'HD': 'Home Depot', 'NKE': 'Nike Inc.',
    'KO': 'Coca-Cola Co.', 'XOM': 'Exxon Mobil', 'CVX': 'Chevron Corp.',
    'CAT': 'Caterpillar Inc.', 'GE': 'GE Aerospace', 'UNP': 'Union Pacific',
    'HON': 'Honeywell Intl.', 'ETN': 'Eaton Corp.',
}


@router.get("/tickers/search")
async def search_tickers(q: str = Query(default="", description="Search query")):
    """Search supported tickers by symbol or company name."""
    query = q.strip().upper()
    if not query:
        return [{"symbol": k, "name": v} for k, v in TICKERS.items()]

    results = []
    for symbol, name in TICKERS.items():
        if query in symbol or query.lower() in name.lower():
            results.append({"symbol": symbol, "name": name})
    return results


@router.get("/{ticker}")
async def market_snapshot(ticker: str):
    """
    Get live market snapshot with technical indicators and signal interpretation.
    
    Example: GET /market/RELIANCE.NS
    """
    try:
        snapshot = get_market_snapshot(ticker)
        if "error" in snapshot:
            raise HTTPException(status_code=404, detail=snapshot["error"])
        return snapshot
    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback to Render logs for debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")
