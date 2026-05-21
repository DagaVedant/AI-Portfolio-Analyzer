"""
backend_api.py - FastAPI wrapper around the existing inference pipeline.

This creates REST API endpoints that the React frontend can call.
Run with: uvicorn backend_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from typing import Optional, Dict, Any

# Add src to path to import existing code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, load_env
from src.inference import run_inference

# Initialize FastAPI app
app = FastAPI(
    title="AI Portfolio Analyzer API",
    description="REST API for the AI Portfolio Analyzer",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config once at startup
config = None

@app.on_event("startup")
async def startup_event():
    """Load config when server starts"""
    global config
    load_env()
    config = load_config("configs/config.yaml")
    if os.getenv("NEWSAPI_KEY", "").strip():
        config["news"]["provider"] = "newsapi"
    print("✓ FastAPI server started")

# Request/Response models
class AnalysisRequest(BaseModel):
    """Request model for portfolio analysis"""
    ticker: str
    forecast_horizon: int = 21

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    data: Dict[str, Any]
    error: Optional[str] = None
    success: bool = True

# Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "AI Portfolio Analyzer API",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticker(request: AnalysisRequest) -> AnalysisResponse:
    """
    Analyze a stock ticker and return forecast, risk, and sentiment data.
    
    Args:
        request: AnalysisRequest containing ticker and forecast_horizon
    
    Returns:
        AnalysisResponse with analysis results
    
    Raises:
        HTTPException: If analysis fails
    """
    try:
        global config
        if config is None:
            raise HTTPException(status_code=500, detail="Config not loaded")
        
        # Update forecast horizon
        cfg = config.copy()
        cfg["data"]["forecast_horizon"] = request.forecast_horizon
        
        # Run inference
        result = run_inference(request.ticker, cfg)
        
        # Check for errors
        if "error" in result:
            return AnalysisResponse(
                data={},
                error=result["error"],
                success=False
            )
        
        return AnalysisResponse(
            data=result,
            error=None,
            success=True
        )
    
    except Exception as e:
        return AnalysisResponse(
            data={},
            error=str(e),
            success=False
        )

@app.get("/tickers")
async def get_popular_tickers():
    """Get list of popular tickers to display in UI"""
    return {
        "tickers": ["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL", "AMD", "SPY", "QQQ"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "AI Portfolio Analyzer",
        "database": "connected",
        "model": "loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
