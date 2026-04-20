"""
Market snapshot API routes.
"""

from fastapi import APIRouter, HTTPException

from src.features import get_market_snapshot

router = APIRouter(prefix="/market", tags=["Market"])


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
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")
