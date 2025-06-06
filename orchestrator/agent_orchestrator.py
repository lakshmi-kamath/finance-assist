import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional, TypedDict
import logging
from datetime import datetime
import dotenv
dotenv.load_dotenv()

from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

from agents.base_agent import BaseAgent, Task, AgentResult, TaskPriority
from agents.api_agent import create_api_agent
from agents.scraping_agent import create_scraping_agent
from agents.retriever_agent import RetrieverAgent
from orchestrator.orchestrator_nodes import OrchestratorNodes
from orchestrator.state import OrchestratorState
from agents.language_agent import MorningBriefAgent, MarketData

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates API, scraping, and retrieval agents using LangGraph"""
    
    def __init__(self, agent_id: str = "orchestrator_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        
        # Initialize sub-agents
        self.api_agent = create_api_agent(config)
        self.scraping_agent = create_scraping_agent(config)
        retriever_config = config.copy() if config else {}
        retriever_config.update({
            'embedding_model': 'all-MiniLM-L6-v2',  # Add this
            'vector_db_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/faiss.index',  # Add this
            'chunk_index_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/metadata.pkl',  # Add this
            'confidence_threshold': 0.7,
            'max_search_results': 10
        })
        self.retriever_agent = RetrieverAgent(config=retriever_config)
        
        gemini_api_key = config.get('gemini_api_key') if config else None
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if gemini_api_key:
            self.language_agent = MorningBriefAgent(gemini_api_key)
        else:
            self.logger.warning("No Gemini API key found. Language agent will not be available.")
            self.language_agent = None

        # Initialize enhanced nodes container
# Initialize enhanced nodes container
        self.nodes = OrchestratorNodes(
            api_agent=self.api_agent,
            scraping_agent=self.scraping_agent,
            retriever_agent=self.retriever_agent,
            language_agent=self.language_agent,  # ADD THIS
            logger=self.logger
        )
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _define_capabilities(self) -> List[str]:
        """Define orchestrator capabilities"""
        return [
            'process_user_query',
            'coordinate_data_collection',
            'generate_comprehensive_report',
            'analyze_market_sentiment',
            'get_company_insights',
            'contextual_analysis',
            'research_enhanced_analysis'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['api_agent', 'scraping_agent', 'retriever_agent']
    
    def _build_workflow(self) -> StateGraph:
        """Build enhanced LangGraph workflow for orchestration with retrieval and language generation"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.nodes.analyze_query_node)
        workflow.add_node("retrieve_context", self.nodes.retrieve_context_node)
        workflow.add_node("execute_api_tasks", self.nodes.execute_api_tasks_node)
        workflow.add_node("execute_scraping_tasks", self.nodes.execute_scraping_tasks_node)
        workflow.add_node("enhance_with_retrieval", self.nodes.enhance_with_retrieval_node)
        workflow.add_node("generate_language_brief", self.nodes.generate_language_brief_node)  # NEW NODE
        workflow.add_node("combine_results", self.nodes.combine_results_node)
        
        # Set up workflow edges
        workflow.add_edge("analyze_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "execute_api_tasks")
        workflow.add_edge("execute_api_tasks", "execute_scraping_tasks") 
        workflow.add_edge("execute_scraping_tasks", "enhance_with_retrieval")
        workflow.add_edge("enhance_with_retrieval", "generate_language_brief")  # NEW EDGE
        workflow.add_edge("generate_language_brief", "combine_results")  # MODIFIED EDGE
        workflow.add_edge("combine_results", END)
        
        workflow.set_entry_point("analyze_query")
        
        return workflow.compile()
    
    def _create_retrieval_task_type(self, user_query: str, context: Dict[str, Any]) -> str:
        """Determine appropriate retrieval task type based on query and context"""
        query_lower = user_query.lower()
        
        # Map query patterns to retrieval task types
        if any(term in query_lower for term in ['portfolio', 'holdings', 'allocation']):
            return 'portfolio_analysis_retrieval'
        elif any(term in query_lower for term in ['earnings', 'financial results', 'quarterly']):
            return 'earnings_analysis_retrieval'
        elif any(term in query_lower for term in ['risk', 'volatility', 'beta']):
            return 'market_risk_retrieval'
        elif any(term in query_lower for term in ['sentiment', 'news', 'opinion']):
            return 'news_sentiment_retrieval'
        else:
            return 'contextual_retrieval'  # Default
        

    async def execute_task(self, task: Task) -> AgentResult:
        """Execute orchestration tasks"""
        task_type = task.type
        parameters = task.parameters
        
        try:
            if task_type == 'process_user_query':
                return await self._process_user_query(parameters)
            
            elif task_type == 'coordinate_data_collection':
                return await self._coordinate_data_collection(parameters)
            
            elif task_type == 'generate_comprehensive_report':
                return await self._generate_comprehensive_report(parameters)
            
            elif task_type == 'analyze_market_sentiment':
                return await self._analyze_market_sentiment(parameters)
            
            elif task_type == 'get_company_insights':
                return await self._get_company_insights(parameters)
            
            elif task_type == 'contextual_analysis':
                return await self._contextual_analysis(parameters)
            
            elif task_type == 'research_enhanced_analysis':
                return await self._research_enhanced_analysis(parameters)
            
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing task {task_type}: {e}")
            return AgentResult(
                success=False,
                error=str(e)
            )

    async def _process_user_query(self, parameters: Dict[str, Any]) -> AgentResult:
        """Main entry point for processing user queries"""
        user_query = parameters.get('query', '')
        symbols = parameters.get('symbols', [])
        query_context = parameters.get('context', {})
        
        if not user_query:
            return AgentResult(success=False, error="No query provided")
        
        # Initialize state with proper typing including new retrieval fields
        initial_state = OrchestratorState(
            user_query=user_query,
            symbols=symbols,
            query_context=query_context,
            task_type='',
            task_parameters={},
            api_results=[],
            scraping_results=[],
            retrieval_results=[],
            retrieved_context={},
            final_results={},
            error=None,
            execution_plan=[]
        )
        
        try:
            # Execute workflow with config
            config = RunnableConfig(
                callbacks=[],
                tags=["orchestrator", "enhanced"],
                max_concurrency=1  # Ensure sequential processing
            )
            
            final_state = await self.workflow.ainvoke(
                initial_state,
                config=config
            )
            
            # Process results
            if isinstance(final_state, dict) and final_state.get('error'):
                return AgentResult(
                    success=False,
                    error=final_state['error']
                )
            
            return AgentResult(
                success=True,
                data=final_state.get('final_results', {}),
                metadata=self._create_metadata(final_state)
            )
            
        except Exception as e:
            self.logger.error(f"Error in workflow execution: {e}")
            return AgentResult(success=False, error=str(e))
        
    def _create_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata from workflow state"""
        return {
            'query': state.get('user_query', ''),
            'symbols_processed': len(state.get('symbols', [])),
            'execution_plan': state.get('execution_plan', []),
            'api_tasks_executed': len(state.get('api_results', [])),
            'scraping_tasks_executed': len(state.get('scraping_results', [])),
            'retrieval_tasks_executed': len(state.get('retrieval_results', [])),
            'context_retrieved': bool(state.get('retrieved_context', {})),
            'retrieval_confidence': state.get('retrieved_context', {}).get('confidence_score', 0.0),
            'language_brief_generated': bool(state.get('language_brief', {})),  # ADD THIS
            'brief_data_points': state.get('language_brief', {}).get('data_points_used', 0),  # ADD THIS
            'brief_portfolio_focused': state.get('language_brief', {}).get('portfolio_focused', False)  # ADD THIS
        }
    
    # Enhanced methods with retrieval integration
    async def _coordinate_data_collection(self, parameters: Dict[str, Any]) -> AgentResult:
        """Coordinate data collection across multiple agents including retrieval"""
        return await self._process_user_query(parameters)
    
    async def _generate_comprehensive_report(self, parameters: Dict[str, Any]) -> AgentResult:
        """Generate comprehensive report with retrieved context"""
        parameters['query'] = parameters.get('query', 'comprehensive analysis with historical context')
        return await self._process_user_query(parameters)
    
    async def _analyze_market_sentiment(self, parameters: Dict[str, Any]) -> AgentResult:
        """Analyze market sentiment with historical sentiment data"""
        parameters['query'] = parameters.get('query', 'market sentiment analysis with historical trends')
        return await self._process_user_query(parameters)
    
    async def _get_company_insights(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get comprehensive company insights with historical context"""
        symbols = parameters.get('symbols', [])
        if symbols:
            parameters['query'] = f"comprehensive company analysis for {' '.join(symbols)} with historical performance"
            parameters['context'] = {'tickers': symbols, 'analysis_type': 'company_insights'}
        else:
            parameters['query'] = 'company insights analysis with historical benchmarks'
        return await self._process_user_query(parameters)
    
    async def _contextual_analysis(self, parameters: Dict[str, Any]) -> AgentResult:
        """Perform contextual analysis leveraging retrieval agent capabilities"""
        query = parameters.get('query', '')
        context = parameters.get('context', {})
        
        # Enhance query with context-specific terms
        if context.get('analysis_type') == 'portfolio':
            query = f"portfolio analysis {query} with sector comparison"
        elif context.get('analysis_type') == 'earnings':
            query = f"earnings analysis {query} with peer comparison"
        elif context.get('analysis_type') == 'risk':
            query = f"risk assessment {query} with market correlation"
        
        parameters['query'] = query
        parameters['context'] = context
        return await self._process_user_query(parameters)
    
    async def _research_enhanced_analysis(self, parameters: Dict[str, Any]) -> AgentResult:
        """Perform research-enhanced analysis using comprehensive retrieval"""
        query = parameters.get('query', '')
        symbols = parameters.get('symbols', [])
        
        # Build comprehensive context for retrieval
        context = {
            'tickers': symbols,
            'analysis_depth': 'comprehensive',
            'include_historical': True,
            'include_peer_analysis': True,
            'include_sector_trends': True
        }
        
        parameters['context'] = context
        parameters['query'] = f"comprehensive research analysis {query}"
        return await self._process_user_query(parameters)
    
    def health_check(self) -> Dict[str, Any]:
        """Synchronous health check with proper async handling"""
        base_health = super().health_check()
        
        # Check sub-agents synchronously
        sub_agent_health = {}
        
        # API agent (sync)
        if self.api_agent:
            try:
                api_health = self.api_agent.health_check()
                # Normalize to dict format
                if isinstance(api_health, bool):
                    sub_agent_health['api_agent'] = {'healthy': api_health}
                else:
                    sub_agent_health['api_agent'] = api_health
            except Exception as e:
                sub_agent_health['api_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['api_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
        # Scraping agent (sync)
        if self.scraping_agent:
            try:
                scraping_health = self.scraping_agent.health_check()
                # Normalize to dict format
                if isinstance(scraping_health, bool):
                    sub_agent_health['scraping_agent'] = {'healthy': scraping_health}
                else:
                    sub_agent_health['scraping_agent'] = scraping_health
            except Exception as e:
                sub_agent_health['scraping_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['scraping_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
        # Retriever agent (async - handle specially)
        if self.retriever_agent:
            try:
                # Check if health_check is async or sync
                health_result = self.retriever_agent.health_check()
                if hasattr(health_result, '__await__'):
                    # It's async, we can't await in sync context
                    sub_agent_health['retriever_agent'] = {
                        'healthy': True,  # Assume healthy since agent exists
                        'note': 'async_health_check_not_evaluated'
                    }
                else:
                    # Normalize to dict format
                    if isinstance(health_result, bool):
                        sub_agent_health['retriever_agent'] = {'healthy': health_result}
                    else:
                        sub_agent_health['retriever_agent'] = health_result
            except Exception as e:
                sub_agent_health['retriever_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['retriever_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
            # Language agent check
        if self.language_agent:
            try:
                # Language agent doesn't have a health_check method, so just check if it exists
                sub_agent_health['language_agent'] = {
                    'healthy': True,
                    'model': 'gemini-1.5-flash'
                }
            except Exception as e:
                sub_agent_health['language_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['language_agent'] = {'healthy': False, 'error': 'not_initialized'}

        base_health['sub_agents'] = sub_agent_health
        base_health['healthy'] = base_health['healthy'] and all(
            agent.get('healthy', False) for agent in sub_agent_health.values()
        )
        
        return base_health

    async def async_health_check(self) -> Dict[str, Any]:
        """Async health check including retriever agent"""
        base_health = super().health_check()
        
        # Check sub-agents with proper async handling
        sub_agent_health = {}
        
        # API agent
        if self.api_agent:
            try:
                api_health = self.api_agent.health_check()
                # Normalize to dict format
                if isinstance(api_health, bool):
                    sub_agent_health['api_agent'] = {'healthy': api_health}
                else:
                    sub_agent_health['api_agent'] = api_health
            except Exception as e:
                sub_agent_health['api_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['api_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
        # Scraping agent  
        if self.scraping_agent:
            try:
                scraping_health = self.scraping_agent.health_check()
                # Normalize to dict format
                if isinstance(scraping_health, bool):
                    sub_agent_health['scraping_agent'] = {'healthy': scraping_health}
                else:
                    sub_agent_health['scraping_agent'] = scraping_health
            except Exception as e:
                sub_agent_health['scraping_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['scraping_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
        # Retriever agent (async)
        if self.retriever_agent:
            try:
                retriever_health = await self.retriever_agent.health_check()
                # Normalize to dict format
                if isinstance(retriever_health, bool):
                    sub_agent_health['retriever_agent'] = {'healthy': retriever_health}
                else:
                    sub_agent_health['retriever_agent'] = retriever_health
            except Exception as e:
                sub_agent_health['retriever_agent'] = {'healthy': False, 'error': str(e)}
        else:
            sub_agent_health['retriever_agent'] = {'healthy': False, 'error': 'not_initialized'}
        
        base_health['sub_agents'] = sub_agent_health
        base_health['healthy'] = base_health['healthy'] and all(
            agent.get('healthy', False) for agent in sub_agent_health.values()
        )
        
        return base_health

# Utility function to create configured orchestrator agent
def create_orchestrator_agent(config: Dict[str, Any] = None) -> OrchestratorAgent:
    """Factory function to create configured orchestrator agent"""
    default_config = {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'gemini_api_key': os.getenv('GEMINI_API_KEY'),  # ADD THIS
        'rate_limit_delay': 2,
        'max_concurrent_requests': 3,
        'timeout_seconds': 300,
        # Retrieval-specific config - Update these
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_db_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/faiss.index', 
        'chunk_index_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/metadata.pkl',
        'default_freshness_weight': 0.3,
        'default_relevance_weight': 0.4,  
        'default_similarity_weight': 0.3,
        'confidence_threshold': 0.7,
        'max_search_results': 10
}
    
    if config:
        default_config.update(config)
    
    return OrchestratorAgent(config=default_config)


