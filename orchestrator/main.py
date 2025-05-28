"""
FastAPI Microservice Wrapper for Multi-Agent Finance Assistant Orchestrator
Provides HTTP-based API endpoints for the orchestrator functionality
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import your orchestrator (assuming it's in a separate module)
from agent_orchestrator import FinanceOrchestratorAgent, create_orchestrator, QueryType, AnalysisContext
# For this example, we'll assume the classes are available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for API

class QueryTypeEnum(str):
    """Enum for query types"""
    MORNING_BRIEF = "morning_brief"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_ANALYSIS = "market_analysis"
    EARNINGS_ANALYSIS = "earnings_analysis"
    RISK_ANALYSIS = "risk_analysis"
    NEWS_SENTIMENT = "news_sentiment"

class AnalysisRequest(BaseModel):
    """Request model for analysis queries"""
    query: str = Field(..., description="The user query to analyze", min_length=1, max_length=1000)
    query_type: Optional[QueryTypeEnum] = Field(None, description="Optional query type classification")
    portfolio_tickers: Optional[List[str]] = Field(None, description="List of portfolio tickers to analyze")
    time_horizon: str = Field("1d", description="Time horizon for analysis", regex=r"^(1d|1w|1m|3m|1y)$")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold", ge=0.0, le=1.0)
    max_sources: int = Field(10, description="Maximum number of sources to use", ge=1, le=50)
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")
    
    @validator('portfolio_tickers')
    def validate_tickers(cls, v):
        if v is not None:
            # Basic ticker validation
            for ticker in v:
                if not ticker.isupper() or len(ticker) > 10:
                    raise ValueError(f"Invalid ticker format: {ticker}")
        return v

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    request_id: str = Field(..., description="Unique request identifier")
    response: str = Field(..., description="Generated analysis response")
    confidence: float = Field(..., description="Confidence score for the analysis")
    metadata: Dict[str, Any] = Field(..., description="Additional response metadata")
    sources_used: int = Field(..., description="Number of data sources used")
    fallback_triggered: bool = Field(..., description="Whether fallback logic was triggered")
    errors: List[str] = Field(..., description="List of any errors encountered")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    orchestrator_status: Dict[str, Any] = Field(..., description="Orchestrator agent status")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    queries: List[AnalysisRequest] = Field(..., description="List of analysis requests", min_items=1, max_items=10)
    parallel_processing: bool = Field(True, description="Whether to process queries in parallel")

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis"""
    batch_id: str = Field(..., description="Unique batch identifier")
    results: List[AnalysisResponse] = Field(..., description="List of analysis results")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")

class AsyncTaskStatus(BaseModel):
    """Status model for async tasks"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    created_at: str = Field(..., description="Task creation timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")
    result: Optional[AnalysisResponse] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")

# Global variables for service state
service_start_time = time.time()
orchestrator_instance = None
active_tasks: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global orchestrator_instance
    
    # Startup
    logger.info("Starting Finance Orchestrator API Service...")
    
    # Initialize orchestrator
    orchestrator_config = {
        "confidence_threshold": 0.7,
        "max_sources_per_agent": 15,
        "timeout_seconds": 30,
        "enable_fallbacks": True,
        "log_level": "INFO"
    }
    
    orchestrator_instance = create_orchestrator(orchestrator_config)
    logger.info("Orchestrator initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Finance Orchestrator API Service...")
    # Clean up any resources if needed

# Create FastAPI app
app = FastAPI(
    title="Finance Orchestrator API",
    description="Multi-Agent Finance Assistant Orchestrator API Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependency to get orchestrator instance
async def get_orchestrator() -> FinanceOrchestratorAgent:
    """Dependency to get orchestrator instance"""
    if orchestrator_instance is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator_instance

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check(orchestrator: FinanceOrchestratorAgent = Depends(get_orchestrator)):
    """Health check endpoint"""
    uptime = time.time() - service_start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime,
        orchestrator_status=orchestrator.get_workflow_status()
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_query(
    request: AnalysisRequest,
    orchestrator: FinanceOrchestratorAgent = Depends(get_orchestrator)
):
    """
    Analyze a financial query using the orchestrator
    
    This endpoint processes a single financial query and returns comprehensive analysis
    using multiple data sources and AI agents.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing query request {request_id}: {request.query}")
        
        # Build analysis context
        context = AnalysisContext(
            query_type=QueryType(request.query_type) if request.query_type else None,
            user_query=request.query,
            portfolio_tickers=request.portfolio_tickers,
            time_horizon=request.time_horizon,
            confidence_threshold=request.confidence_threshold,
            max_sources=request.max_sources,
            preferences=request.preferences or {}
        )
        
        # Process query
        result = await orchestrator.process_query(request.query, context)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return AnalysisResponse(
            request_id=request_id,
            response=result["response"],
            confidence=result["confidence"],
            metadata=result["metadata"],
            sources_used=result["sources_used"],
            fallback_triggered=result["fallback_triggered"],
            errors=result["errors"],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            request_id=request_id,
            response="I apologize, but I encountered an error processing your request.",
            confidence=0.0,
            metadata={"error": str(e)},
            sources_used=0,
            fallback_triggered=True,
            errors=[str(e)],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )

@app.post("/analyze/async")
async def analyze_query_async(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    orchestrator: FinanceOrchestratorAgent = Depends(get_orchestrator)
):
    """
    Submit a query for asynchronous processing
    
    Returns a task ID that can be used to check status and retrieve results.
    """
    task_id = str(uuid.uuid4())
    
    # Store task info
    active_tasks[task_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "result": None,
        "error": None
    }
    
    # Add background task
    background_tasks.add_task(process_async_query, task_id, request, orchestrator)
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "message": "Query submitted for processing",
        "status_url": f"/tasks/{task_id}/status"
    }

@app.get("/tasks/{task_id}/status", response_model=AsyncTaskStatus)
async def get_task_status(task_id: str = Path(..., description="Task ID")):
    """Get the status of an async task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    return AsyncTaskStatus(
        task_id=task_id,
        status=task_info["status"],
        created_at=task_info["created_at"],
        completed_at=task_info.get("completed_at"),
        result=task_info.get("result"),
        error=task_info.get("error")
    )

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    orchestrator: FinanceOrchestratorAgent = Depends(get_orchestrator)
):
    """
    Process multiple queries in batch
    
    Can process queries either in parallel or sequentially based on the request.
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    logger.info(f"Processing batch request {batch_id} with {len(request.queries)} queries")
    
    results = []
    successful_queries = 0
    failed_queries = 0
    
    if request.parallel_processing:
        # Process in parallel
        tasks = []
        for query_request in request.queries:
            task = process_single_query(query_request, orchestrator)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                failed_queries += 1
                # Create error response
                error_response = AnalysisResponse(
                    request_id=str(uuid.uuid4()),
                    response="Error processing query",
                    confidence=0.0,
                    metadata={"error": str(result)},
                    sources_used=0,
                    fallback_triggered=True,
                    errors=[str(result)],
                    processing_time_ms=0,
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_response)
            else:
                successful_queries += 1
                results.append(result)
    else:
        # Process sequentially
        for query_request in request.queries:
            try:
                result = await process_single_query(query_request, orchestrator)
                results.append(result)
                successful_queries += 1
            except Exception as e:
                failed_queries += 1
                error_response = AnalysisResponse(
                    request_id=str(uuid.uuid4()),
                    response="Error processing query",
                    confidence=0.0,
                    metadata={"error": str(e)},
                    sources_used=0,
                    fallback_triggered=True,
                    errors=[str(e)],
                    processing_time_ms=0,
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_response)
    
    total_processing_time = (time.time() - start_time) * 1000
    
    return BatchAnalysisResponse(
        batch_id=batch_id,
        results=results,
        total_processing_time_ms=total_processing_time,
        successful_queries=successful_queries,
        failed_queries=failed_queries
    )

@app.get("/query-types")
async def get_query_types():
    """Get available query types"""
    return {
        "query_types": [
            {"value": "morning_brief", "description": "Daily morning market brief"},
            {"value": "portfolio_analysis", "description": "Portfolio performance analysis"},
            {"value": "market_analysis", "description": "General market analysis"},
            {"value": "earnings_analysis", "description": "Earnings reports analysis"},
            {"value": "risk_analysis", "description": "Risk assessment analysis"},
            {"value": "news_sentiment", "description": "News sentiment analysis"}
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    uptime = time.time() - service_start_time
    
    return {
        "uptime_seconds": uptime,
        "active_tasks": len(active_tasks),
        "completed_tasks": len([t for t in active_tasks.values() if t["status"] == "completed"]),
        "failed_tasks": len([t for t in active_tasks.values() if t["status"] == "failed"]),
        "memory_usage": "Not implemented",  # Could add psutil for memory metrics
        "service_version": "1.0.0"
    }

@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str = Path(..., description="Task ID")):
    """Cancel an async task"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    if task_info["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Task already completed")
    
    # Mark as cancelled
    task_info["status"] = "cancelled"
    task_info["completed_at"] = datetime.now().isoformat()
    
    return {"message": "Task cancelled successfully"}

@app.post("/orchestrator/reload")
async def reload_orchestrator():
    """Reload the orchestrator configuration"""
    global orchestrator_instance
    
    try:
        # Reinitialize orchestrator
        orchestrator_config = {
            "confidence_threshold": 0.7,
            "max_sources_per_agent": 15,
            "timeout_seconds": 30,
            "enable_fallbacks": True,
            "log_level": "INFO"
        }
        
        orchestrator_instance = create_orchestrator(orchestrator_config)
        
        return {"message": "Orchestrator reloaded successfully"}
    
    except Exception as e:
        logger.error(f"Error reloading orchestrator: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload orchestrator: {str(e)}")

# Helper Functions

async def process_async_query(
    task_id: str,
    request: AnalysisRequest,
    orchestrator: FinanceOrchestratorAgent
):
    """Process a query asynchronously"""
    try:
        active_tasks[task_id]["status"] = "processing"
        
        result = await process_single_query(request, orchestrator)
        
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        active_tasks[task_id]["result"] = result
        
    except Exception as e:
        logger.error(f"Error in async task {task_id}: {str(e)}")
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        active_tasks[task_id]["error"] = str(e)

async def process_single_query(
    request: AnalysisRequest,
    orchestrator: FinanceOrchestratorAgent
) -> AnalysisResponse:
    """Process a single query and return response"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Build analysis context
    context = AnalysisContext(
        query_type=QueryType(request.query_type) if request.query_type else None,
        user_query=request.query,
        portfolio_tickers=request.portfolio_tickers,
        time_horizon=request.time_horizon,
        confidence_threshold=request.confidence_threshold,
        max_sources=request.max_sources,
        preferences=request.preferences or {}
    )
    
    # Process query
    result = await orchestrator.process_query(request.query, context)
    
    processing_time = (time.time() - start_time) * 1000
    
    return AnalysisResponse(
        request_id=request_id,
        response=result["response"],
        confidence=result["confidence"],
        metadata=result["metadata"],
        sources_used=result["sources_used"],
        fallback_triggered=result["fallback_triggered"],
        errors=result["errors"],
        processing_time_ms=processing_time,
        timestamp=datetime.now().isoformat()
    )

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )