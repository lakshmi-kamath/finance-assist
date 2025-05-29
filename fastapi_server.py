import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import QueryRequest, QueryResponse, HealthCheckResponse, ErrorResponse
from api.query_processor import create_query_processor
from utils.response_formatter import create_response_formatter
from orchestrator.agent_orchestrator import create_orchestrator_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fastapi_server.log')
    ]
)

logger = logging.getLogger(__name__)

# Global components
orchestrator_agent = None
query_processor = None
response_formatter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    logger.info("Starting FastAPI server...")
    
    global orchestrator_agent, query_processor, response_formatter
    
    try:
        # Initialize components
        logger.info("Initializing orchestrator agent...")
        orchestrator_agent = create_orchestrator_agent()
        
        logger.info("Initializing query processor...")
        query_processor = create_query_processor()
        
        logger.info("Initializing response formatter...")
        response_formatter = create_response_formatter()
        
        # Health check
        health = await orchestrator_agent.async_health_check()
        if health.get('healthy'):
            logger.info("All components initialized successfully")
        else:
            logger.warning(f"Some components may have issues: {health}")
        
        yield  # Server runs
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down FastAPI server...")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Finance Assistant API",
    description="AI-powered financial analysis and market intelligence API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Finance Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if orchestrator_agent is None:
            raise HTTPException(status_code=503, detail="Orchestrator agent not initialized")
        
        # Perform async health check
        health_result = await orchestrator_agent.async_health_check()
        
        return HealthCheckResponse(
            status="healthy" if health_result.get('healthy') else "unhealthy",
            healthy=health_result.get('healthy', False),
            services=health_result.get('sub_agents', {})
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            healthy=False,
            services={"error": {"healthy": False, "error": str(e)}}
        )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Main query processing endpoint"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing query {query_id}: {request.query[:100]}...")
        
        # Validate components
        if not all([orchestrator_agent, query_processor, response_formatter]):
            raise HTTPException(status_code=503, detail="Service components not fully initialized")
        
        # Process query
        processed_query = await query_processor.process_query(request.query)
        logger.info(f"Query {query_id} processed: {len(processed_query.symbols)} symbols identified")
        
        # Create orchestrator task
        orchestrator_task = {
            'type': 'process_user_query',
            'parameters': {
                'query': processed_query.query,
                'symbols': processed_query.symbols,
                'context': processed_query.context
            }
        }
        
        # Execute orchestrator task
        from agents.base_agent import Task, TaskPriority
        task = Task(
            id=query_id,
            type=orchestrator_task['type'],
            parameters=orchestrator_task['parameters'],
            priority=TaskPriority.HIGH
        )
        
        logger.info(f"Executing orchestrator task for query {query_id}")
        orchestrator_result = await orchestrator_agent.execute_task(task)
        
        if not orchestrator_result.success:
            raise HTTPException(
                status_code=500, 
                detail=f"Orchestrator execution failed: {orchestrator_result.error}"
            )
        
        # Format response
        processing_time = time.time() - start_time
        formatted_result = response_formatter.format_orchestrator_result(
            orchestrator_result.data, 
            processing_time
        )
        
        # *** IMPROVED FIX: Extract language brief from orchestrator result ***
        language_brief = None
        
        # Try multiple locations where the language brief might be stored
        if orchestrator_result.data:
            # Check if orchestrator_result.data has language_brief directly
            if isinstance(orchestrator_result.data, dict):
                # First try direct access
                brief_candidate = orchestrator_result.data.get('language_brief')
                
                # If it's a dict, extract the 'brief' key
                if isinstance(brief_candidate, dict):
                    language_brief = brief_candidate.get('brief')
                    # If no 'brief' key, try other common keys
                    if not language_brief:
                        language_brief = brief_candidate.get('result') or brief_candidate.get('content') or brief_candidate.get('analysis')
                elif isinstance(brief_candidate, str):
                    language_brief = brief_candidate
                
                # If still not found, check in agent_results
                if not language_brief:
                    agent_results = orchestrator_result.data.get('agent_results', {})
                    if isinstance(agent_results, dict):
                        # Look for language agent result
                        for agent_name, agent_data in agent_results.items():
                            if isinstance(agent_data, dict):
                                # Check agents with relevant names
                                if any(keyword in agent_name.lower() for keyword in ['language', 'brief', 'summary', 'natural']):
                                    # Try multiple extraction approaches
                                    brief_candidate = (
                                        agent_data.get('language_brief') or 
                                        agent_data.get('brief') or 
                                        agent_data.get('result') or 
                                        agent_data.get('content') or
                                        agent_data.get('analysis')
                                    )
                                    
                                    # Handle dict responses
                                    if isinstance(brief_candidate, dict):
                                        language_brief = (
                                            brief_candidate.get('brief') or 
                                            brief_candidate.get('result') or 
                                            brief_candidate.get('content') or
                                            brief_candidate.get('analysis')
                                        )
                                    elif isinstance(brief_candidate, str):
                                        language_brief = brief_candidate
                                    
                                    if language_brief:
                                        break
                                
                                # Also check any agent that has a language_brief field
                                if not language_brief:
                                    brief_candidate = agent_data.get('language_brief')
                                    if isinstance(brief_candidate, dict):
                                        language_brief = brief_candidate.get('brief')
                                    elif isinstance(brief_candidate, str):
                                        language_brief = brief_candidate
                                    
                                    if language_brief:
                                        break
        
        # Ensure language_brief is a string
        if language_brief and not isinstance(language_brief, str):
            # If it's still not a string, try to convert or extract
            if isinstance(language_brief, dict):
                # Try common keys
                language_brief = (
                    language_brief.get('brief') or 
                    language_brief.get('result') or 
                    language_brief.get('content') or
                    language_brief.get('analysis') or
                    str(language_brief)  # Last resort: stringify
                )
            else:
                language_brief = str(language_brief)
        
        # Log for debugging
        if language_brief:
            logger.info(f"Language brief found for query {query_id}, length: {len(language_brief)}")
        else:
            logger.warning(f"Language brief not found for query {query_id}")
            logger.debug(f"Orchestrator result structure: {list(orchestrator_result.data.keys()) if orchestrator_result.data else 'None'}")
            # Log the actual structure for debugging
            if orchestrator_result.data:
                logger.debug(f"Agent results keys: {list(orchestrator_result.data.get('agent_results', {}).keys())}")
        
        # Ensure formatted_result includes the language brief
        if not formatted_result:
            formatted_result = {}
        
        if not formatted_result.get('analysis_result'):
            formatted_result['analysis_result'] = {}
        
        # Add language brief to the analysis result
        if language_brief:
            formatted_result['analysis_result']['language_brief'] = language_brief
        
        # Create response with explicit language brief inclusion
        response_data = {
            "success": True,
            "query_id": query_id,
            "processed_query": processed_query,
            "analysis_result": formatted_result.get('analysis_result', {}),
            "metadata": formatted_result.get('metadata', {}),
            "processing_time_seconds": processing_time
        }
        
        # Ensure language brief is accessible at multiple levels for frontend compatibility
        if language_brief:
            # Add at top level for easy access
            response_data["language_brief"] = language_brief
            # Ensure it's in analysis_result
            if not response_data["analysis_result"].get("language_brief"):
                response_data["analysis_result"]["language_brief"] = language_brief
        
        response = QueryResponse(**response_data)
        
        logger.info(f"Query {query_id} completed successfully in {processing_time:.2f}s")
        
        # Schedule background cleanup if needed
        background_tasks.add_task(cleanup_query_resources, query_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Query {query_id} failed after {processing_time:.2f}s: {e}")
        
        # Return error response
        return QueryResponse(
            success=False,
            query_id=query_id,
            processed_query=processed_query if 'processed_query' in locals() else None,
            error=str(e),
            processing_time_seconds=processing_time)
    
@app.post("/query/analyze", response_model=QueryResponse)
async def analyze_symbols(symbols: list[str], analysis_type: str = "comprehensive"):
    """Direct symbol analysis endpoint"""
    try:
        # Create query from symbols
        query_text = f"Provide {analysis_type} analysis for {', '.join(symbols)}"
        
        request = QueryRequest(query=query_text)
        return await process_query(request, BackgroundTasks())
        
    except Exception as e:
        logger.error(f"Symbol analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{query_id}/status")
async def get_query_status(query_id: str):
    """Get status of a specific query (for future async processing)"""
    # Placeholder for future async query status tracking
    return {
        "query_id": query_id,
        "status": "completed",  # For now, all queries are synchronous
        "message": "Query processing completed"
    }

@app.get("/symbols/supported")
async def get_supported_symbols():
    """Get list of supported symbols and their mappings"""
    if query_processor is None:
        raise HTTPException(status_code=503, detail="Query processor not initialized")
    
    return {
        "supported_mappings": query_processor.symbol_mappings,
        "total_count": len(query_processor.symbol_mappings),
        "categories": {
            "us_tech": ["apple", "microsoft", "google", "amazon", "meta", "tesla"],
            "asian_tech": ["samsung", "tsmc", "alibaba", "tencent", "sony", "nintendo"],
            "us_listed_asian": ["tsm", "baba", "sony", "ntdoy"]
        }
    }

@app.get("/analysis/types")
async def get_analysis_types():
    """Get supported analysis types"""
    return {
        "analysis_types": [
            {"key": "comprehensive", "name": "Comprehensive Analysis", "description": "Complete financial analysis with all available data"},
            {"key": "portfolio_analysis", "name": "Portfolio Analysis", "description": "Portfolio-focused analysis with allocation insights"},
            {"key": "earnings_analysis", "name": "Earnings Analysis", "description": "Financial results and earnings-focused analysis"},
            {"key": "market_sentiment", "name": "Market Sentiment", "description": "Sentiment analysis from news and market data"},
            {"key": "risk_assessment", "name": "Risk Assessment", "description": "Risk and volatility analysis"},
            {"key": "company_insights", "name": "Company Insights", "description": "Company-specific analysis and insights"},
            {"key": "peer_comparison", "name": "Peer Comparison", "description": "Comparative analysis with peer companies"},
            {"key": "sector_analysis", "name": "Sector Analysis", "description": "Industry and sector trend analysis"}
        ]
    }

# Background task functions
async def cleanup_query_resources(query_id: str):
    """Clean up resources after query processing"""
    try:
        # Placeholder for cleanup logic
        logger.debug(f"Cleaning up resources for query {query_id}")
        # Could include: clearing temporary files, updating usage metrics, etc.
        
    except Exception as e:
        logger.error(f"Error cleaning up query {query_id}: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="500",
            details={"message": str(exc)}
        ).dict()
    )

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )