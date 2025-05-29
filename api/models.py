from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

class AnalysisType(str, Enum):
    """Supported analysis types"""
    COMPREHENSIVE = "comprehensive"
    PORTFOLIO = "portfolio_analysis"
    EARNINGS = "earnings_analysis" 
    MARKET_SENTIMENT = "market_sentiment"
    RISK_ASSESSMENT = "risk_assessment"
    COMPANY_INSIGHTS = "company_insights"
    PEER_COMPARISON = "peer_comparison"
    SECTOR_ANALYSIS = "sector_analysis"

class QueryRequest(BaseModel):
    """Request model for user queries"""
    query: str = Field(..., description="User's natural language query")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
class ProcessedQuery(BaseModel):
    """Processed query format for orchestrator"""
    query: str = Field(..., description="Processed and enhanced query text")
    symbols: List[str] = Field(default_factory=list, description="Extracted ticker symbols")
    context: Dict[str, Any] = Field(default_factory=dict, description="Query context and parameters")

class QueryContext(BaseModel):
    """Context structure for queries"""
    tickers: List[str] = Field(default_factory=list, description="Ticker symbols")
    analysis_type: AnalysisType = Field(AnalysisType.COMPREHENSIVE, description="Type of analysis")
    include_historical: bool = Field(True, description="Include historical data")
    include_peer_analysis: bool = Field(False, description="Include peer comparison")
    include_sector_trends: bool = Field(False, description="Include sector analysis")
    time_window_hours: int = Field(168, description="Time window in hours (default: 7 days)")
    confidence_threshold: float = Field(0.7, description="Minimum confidence for results")
    max_results: int = Field(10, description="Maximum number of results")

class MarketDataPoint(BaseModel):
    """Individual market data point"""
    symbol: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    timestamp: datetime

class CompanyInsight(BaseModel):
    """Company-specific insights"""
    symbol: str
    company_name: str
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    recent_news: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = None

class AnalysisResult(BaseModel):
    """Structured analysis results"""
    executive_summary: str
    market_analysis: Dict[str, Any] = Field(default_factory=dict)
    company_insights: List[CompanyInsight] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float
    data_points_used: int
    portfolio_focused: bool = False

class QueryResponse(BaseModel):
    """Complete response model"""
    success: bool
    query_id: str
    processed_query: ProcessedQuery
    analysis_result: Optional[AnalysisResult] = None
    language_brief: Optional[str] = None 
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: Optional[float] = None

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    healthy: bool
    services: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)