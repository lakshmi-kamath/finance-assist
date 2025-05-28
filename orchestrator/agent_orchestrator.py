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
from orchestrator_nodes import OrchestratorNodes


class OrchestratorState(TypedDict):
    """State for the orchestrator graph"""
    user_query: str
    symbols: List[str]
    task_type: str
    task_parameters: Dict[str, Any]
    api_results: List[Dict[str, Any]]
    scraping_results: List[Dict[str, Any]]
    final_results: Dict[str, Any]
    error: Optional[str]
    execution_plan: List[str]


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates API and scraping agents using LangGraph"""
    
    def __init__(self, agent_id: str = "orchestrator_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Initialize sub-agents
        self.api_agent = create_api_agent(config)
        self.scraping_agent = create_scraping_agent(config)
        
        # Initialize nodes container
        self.nodes = OrchestratorNodes(
            api_agent=self.api_agent,
            scraping_agent=self.scraping_agent,
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
            'get_company_insights'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['api_agent', 'scraping_agent']
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow for orchestration"""
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes using the separate nodes class
        workflow.add_node("analyze_query", self.nodes.analyze_query_node)
        workflow.add_node("execute_api_tasks", self.nodes.execute_api_tasks_node)
        workflow.add_node("execute_scraping_tasks", self.nodes.execute_scraping_tasks_node)
        workflow.add_node("combine_results", self.nodes.combine_results_node)
        
        # Set up workflow edges (sequential processing)
        workflow.add_edge("analyze_query", "execute_api_tasks")
        workflow.add_edge("execute_api_tasks", "execute_scraping_tasks")
        workflow.add_edge("execute_scraping_tasks", "combine_results")
        workflow.add_edge("combine_results", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow.compile()
    
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
        
        if not user_query:
            return AgentResult(success=False, error="No query provided")
        
        # Initialize state with proper typing
        initial_state = OrchestratorState(
            user_query=user_query,
            symbols=symbols,
            task_type='',
            task_parameters={},
            api_results=[],
            scraping_results=[],
            final_results={},
            error=None,
            execution_plan=[]
        )
        
        try:
            # Execute workflow with config
            config = RunnableConfig(
                callbacks=[],
                tags=["orchestrator"],
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
            'scraping_tasks_executed': len(state.get('scraping_results', []))
        }
    
    # Convenience methods for common workflows
    async def _coordinate_data_collection(self, parameters: Dict[str, Any]) -> AgentResult:
        """Coordinate data collection across multiple agents"""
        return await self._process_user_query(parameters)
    
    async def _generate_comprehensive_report(self, parameters: Dict[str, Any]) -> AgentResult:
        """Generate comprehensive report"""
        parameters['query'] = parameters.get('query', 'comprehensive analysis')
        return await self._process_user_query(parameters)
    
    async def _analyze_market_sentiment(self, parameters: Dict[str, Any]) -> AgentResult:
        """Analyze market sentiment"""
        parameters['query'] = parameters.get('query', 'market sentiment analysis')
        return await self._process_user_query(parameters)
    
    async def _get_company_insights(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get comprehensive company insights"""
        symbols = parameters.get('symbols', [])
        if symbols:
            parameters['query'] = f"comprehensive analysis for {' '.join(symbols)}"
        else:
            parameters['query'] = 'company insights analysis'
        return await self._process_user_query(parameters)
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including sub-agents"""
        base_health = super().health_check()
        
        # Check sub-agents
        sub_agent_health = {
            'api_agent': self.api_agent.health_check() if self.api_agent else {'healthy': False},
            'scraping_agent': self.scraping_agent.health_check() if self.scraping_agent else {'healthy': False}
        }
        
        base_health['sub_agents'] = sub_agent_health
        base_health['healthy'] = base_health['healthy'] and all(
            agent['healthy'] for agent in sub_agent_health.values()
        )
        
        return base_health
    
    def get_structured_output(self, result: AgentResult) -> Dict[str, Any]:
        """Get structured output suitable for passing to next agent"""
        if not result.success:
            return {'error': result.error}
        
        data = result.data
        return {
            'task_type': data.get('task_type'),
            'symbols_analyzed': data.get('symbols_analyzed', []),
            'timestamp': data.get('timestamp'),
            'api_data': data.get('api_data', {}),
            'scraping_data': data.get('scraping_data', {}),
            'insights': data.get('insights', {}),
            'summary': data.get('summary', {}),
            'metadata': result.metadata
        }


# Utility function to create configured orchestrator agent
def create_orchestrator_agent(config: Dict[str, Any] = None) -> OrchestratorAgent:
    """Factory function to create configured orchestrator agent"""
    default_config = {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'rate_limit_delay': 2,
        'max_concurrent_requests': 3,
        'timeout_seconds': 300
    }
    
    if config:
        default_config.update(config)
    
    return OrchestratorAgent(config=default_config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_orchestrator_agent():
        """Test the orchestrator agent functionality"""
        agent = create_orchestrator_agent()
        
        print(f"Orchestrator Status: {agent.get_status()}")
        print(f"Health Check: {agent.health_check()}")
        
        # Test different query types
        test_queries = [
            {
                'query': 'Comprehensive analysis for SAMSUNG including SEC filings',
                'symbols': ['005930.KS','SSNLF']
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*80}")
            print(f"Test Case {i}: {test_case['query']}")
            print(f"{'='*80}")
            
            result = await agent.execute('process_user_query', test_case)
            
            if result.success:
                print(f"✓ Query processed successfully")
                print(f"  Execution plan: {result.metadata.get('execution_plan', [])}")
                print(f"  API tasks: {result.metadata.get('api_tasks_executed', 0)}")
                print(f"  Scraping tasks: {result.metadata.get('scraping_tasks_executed', 0)}")
                print(f"  Symbols processed: {result.metadata.get('symbols_processed', 0)}")
                
                # Show detailed results
                final_results = result.data
                
                print(f"\n--- API RESULTS ---")
                api_data = final_results.get('api_data', {})
                for task_name, task_data in api_data.items():
                    print(f"\n{task_name.upper()}:")
                    data = task_data.get('data', {})
                    metadata = task_data.get('metadata', {})
                    
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (list, dict)) and len(str(value)) > 200:
                                print(f"  {key}: [{type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'complex'} items]")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"  Data: {data}")
                    
                    if metadata:
                        print(f"  Metadata: {metadata}")
                
                print(f"\n--- SCRAPING RESULTS ---")
                scraping_data = final_results.get('scraping_data', {})
                for task_name, task_data in scraping_data.items():
                    print(f"\n{task_name.upper()}:")
                    data = task_data.get('data', {})
                    metadata = task_data.get('metadata', {})
                    
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, (list, dict)) and len(str(value)) > 200:
                                print(f"  {key}: [{type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'complex'} items]")
                            else:
                                print(f"  {key}: {value}")
                    elif isinstance(data, list):
                        print(f"  Data: [List with {len(data)} items]")
                        for idx, item in enumerate(data[:3]):  # Show first 3 items
                            print(f"    Item {idx+1}: {item}")
                        if len(data) > 3:
                            print(f"    ... and {len(data)-3} more items")
                    else:
                        print(f"  Data: {data}")
                    
                    if metadata:
                        print(f"  Metadata: {metadata}")
                
                # Show insights
                insights = final_results.get('insights', {})
                if insights:
                    print(f"\n--- INSIGHTS ---")
                    print(f"  Market sentiment: {insights.get('market_sentiment', {}).get('overall_sentiment', 'N/A')}")
                    print(f"  Key findings: {insights.get('key_findings', [])}")
                    print(f"  Data quality: {insights.get('data_quality', {})}")
                
                # Show summary
                summary = final_results.get('summary', {})
                if summary:
                    print(f"\n--- SUMMARY ---")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                
                print(f"\n--- FULL DATA STRUCTURE FOR NEXT AGENT ---")
                print(f"Available data keys:")
                print(f"  - api_data: {list(api_data.keys())}")
                print(f"  - scraping_data: {list(scraping_data.keys())}")
                print(f"  - insights: {list(insights.keys()) if insights else []}")
                print(f"  - summary: {list(summary.keys()) if summary else []}")
                
            else:
                print(f"✗ Query failed: {result.error}")
        
        print(f"\nFinal Orchestrator Status: {agent.get_status()}")
    
    # Run test
    asyncio.run(test_orchestrator_agent())