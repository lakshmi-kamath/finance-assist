from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field

class OrchestratorState(TypedDict):
    """State for the orchestrator graph"""
    user_query: str
    symbols: List[str]
    query_context: Dict[str, Any]
    task_type: str
    task_parameters: Dict[str, Any]
    api_results: List[Dict[str, Any]]
    scraping_results: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    retrieved_context: Dict[str, Any]
    language_response: Dict[str, Any]
    final_results: Dict[str, Any]
    error: Optional[str]
    execution_plan: List[str]
    language_brief: Dict[str, Any] = {}