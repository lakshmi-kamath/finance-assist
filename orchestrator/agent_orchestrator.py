import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional, TypedDict
import logging
from datetime import datetime
import re
import dotenv
dotenv.load_dotenv()

from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

from agents.base_agent import BaseAgent, Task, AgentResult, TaskPriority
from agents.api_agent import create_api_agent
from agents.scraping_agent import create_scraping_agent

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
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        # Query patterns for routing
        self.query_patterns = {
            'stock_quotes': [r'stock\s+quote', r'current\s+price', r'stock\s+price'],
            'earnings': [r'earnings', r'quarterly\s+results', r'earnings\s+report'],
            'economic_indicators': [r'economic\s+indicator', r'gdp', r'inflation', r'unemployment'],
            'market_overview': [r'market\s+overview', r'market\s+summary', r'overall\s+market'],
            'news_analysis': [r'news', r'sentiment', r'articles', r'financial\s+news'],
            'sec_filings': [r'sec\s+filing', r'10-k', r'10-q', r'8-k', r'annual\s+report'],
            'comprehensive_analysis': [r'comprehensive', r'complete\s+analysis', r'full\s+analysis']
        }
    
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
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("execute_api_tasks", self._execute_api_tasks_node)
        workflow.add_node("execute_scraping_tasks", self._execute_scraping_tasks_node)
        workflow.add_node("combine_results", self._combine_results_node)
        
        # Fix edges - make them sequential instead of parallel
        workflow.add_edge("analyze_query", "execute_api_tasks")
        workflow.add_edge("execute_api_tasks", "execute_scraping_tasks")  # Sequential
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
    async def _analyze_query_node(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze user query and determine execution plan"""
        query = state['user_query'].lower()
        symbols = state['symbols']
        
        # Extract symbols from query if not provided
        if not symbols:
            symbols = self._extract_symbols_from_query(query)
        
        # Determine task type and parameters
        task_type, task_parameters = self._determine_task_type(query, symbols)
        
        # Create execution plan
        execution_plan = self._create_execution_plan(task_type, symbols)
        
        return {
            **state,
            'symbols': symbols,
            'task_type': task_type,
            'task_parameters': task_parameters,
            'execution_plan': execution_plan
        }
    
    async def _execute_api_tasks_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute API-related tasks"""
        task_type = state['task_type']
        symbols = state['symbols']
        task_parameters = state['task_parameters']
        
        api_results = []
        
        try:
            # Always get stock data if symbols are provided
            if symbols:
                # Get stock quotes
                result = await self.api_agent.execute('get_stock_quotes', {'symbols': symbols})
                if result.success:
                    api_results.append({
                        'task': 'stock_quotes',
                        'data': result.data,
                        'metadata': result.metadata
                    })
                
                # Get stock info from Yahoo Finance
                result = await self.api_agent.execute('get_stock_info', {'symbols': symbols})
                if result.success:
                    api_results.append({
                        'task': 'stock_info',
                        'data': result.data,
                        'metadata': result.metadata
                    })
            
            # Always get economic indicators for market context
            days_back = task_parameters.get('days_back', 30)
            result = await self.api_agent.execute('get_economic_indicators', {'days_back': days_back})
            if result.success:
                api_results.append({
                    'task': 'economic_indicators',
                    'data': result.data,
                    'metadata': result.metadata
                })
            
            # Always get market overview for broader context
            result = await self.api_agent.execute('get_market_overview', {})
            if result.success:
                api_results.append({
                    'task': 'market_overview',
                    'data': result.data,
                    'metadata': result.metadata
                })
            
        except Exception as e:
            self.logger.error(f"Error executing API tasks: {e}")
        
        return {
            **state,
            'api_results': api_results
        }
    async def _execute_scraping_tasks_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute scraping-related tasks"""
        task_type = state['task_type']
        symbols = state['symbols']
        task_parameters = state['task_parameters']
        
        scraping_results = []
        
        try:
            # Always get earnings data if symbols are provided
            if symbols:
                result = await self.scraping_agent.execute('batch_earnings_scrape', {
                    'symbols': symbols,
                    'include_calendar': True,
                    'include_analysis': True,
                    'include_reports': True
                })
                if result.success:
                    scraping_results.append({
                        'task': 'earnings_analysis',
                        'data': result.data,
                        'metadata': result.metadata
                    })
                
                # Always get SEC filings for fundamental analysis
                result = await self.scraping_agent.execute('batch_filings_scrape', {
                    'symbols': symbols,
                    'form_types': ['10-K', '10-Q', '8-K'],
                    'include_foreign': True
                })
                if result.success:
                    scraping_results.append({
                        'task': 'sec_filings',
                        'data': result.data,
                        'metadata': result.metadata
                    })
            
            # Always get financial news and sentiment for market context
            hours_back = task_parameters.get('hours_back', 24)
            result = await self.scraping_agent.execute('analyze_news_sentiment', {
                'hours_back': hours_back,
                'max_articles_per_source': 10
            })
            if result.success:
                scraping_results.append({
                    'task': 'news_sentiment',
                    'data': result.data,
                    'metadata': result.metadata
                })
            
            # For comprehensive analysis or when symbols are provided, get detailed company analysis
            if symbols and (task_type == 'comprehensive_analysis' or len(symbols) <= 3):
                result = await self.scraping_agent.execute('comprehensive_company_analysis', {
                    'symbols': symbols[:3]  # Limit to 3 symbols to avoid timeout
                })
                if result.success:
                    scraping_results.append({
                        'task': 'comprehensive_analysis',
                        'data': result.data,
                        'metadata': result.metadata
                    })
            
        except Exception as e:
            self.logger.error(f"Error executing scraping tasks: {e}")
        
        return {
            **state,
            'scraping_results': scraping_results
        }
    async def _combine_results_node(self, state: OrchestratorState) -> OrchestratorState:
        """Combine results from API and scraping tasks"""
        api_results = state['api_results']
        scraping_results = state['scraping_results']
        task_type = state['task_type']
        symbols = state['symbols']
        
        final_results = {
            'task_type': task_type,
            'symbols_analyzed': symbols,
            'timestamp': datetime.now().isoformat(),
            'api_data': {},
            'scraping_data': {},
            'insights': {},
            'summary': {}
        }
        
        # Process API results
        for result in api_results:
            final_results['api_data'][result['task']] = {
                'data': result['data'],
                'metadata': result['metadata']
            }
        
        # Process scraping results
        for result in scraping_results:
            final_results['scraping_data'][result['task']] = {
                'data': result['data'],
                'metadata': result['metadata']
            }
        
        # Generate insights
        final_results['insights'] = self._generate_insights(api_results, scraping_results, task_type)
        
        # Create summary
        final_results['summary'] = self._create_summary(final_results)
        
        return {
            **state,
            'final_results': final_results
        }
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query text"""
        # Skip symbol extraction for market overview queries
        if 'market overview' in query.lower() or 'economic indicators' in query.lower():
            return []
        
        # Common patterns for stock symbols
        symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
            r'\b[A-Z]{1,4}\.[A-Z]{1,3}\b',  # Foreign symbols with exchange
        ]
        
        symbols = []
        for pattern in symbol_patterns:
            matches = re.findall(pattern, query.upper())
            symbols.extend(matches)
        
        # Enhanced filtering for common words
        common_words = {
            'THE', 'AND', 'OR', 'BUT', 'FOR', 'GET', 'API', 'SEC', 'NYSE', 'NEWS',
            'WITH', 'MARKET', 'OVERVIEW', 'ECONOMIC', 'INDICATORS', 'SENTIMENT'
        }
        symbols = [s for s in symbols if s not in common_words]
        
        return list(set(symbols))  # Remove duplicates 
    
    def _determine_task_type(self, query: str, symbols: List[str]) -> tuple[str, Dict[str, Any]]:
        """Determine task type based on query analysis"""
        query_lower = query.lower()
        
        # Check patterns
        for task_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # Set appropriate parameters based on task type
                    parameters = {}
                    if 'hours' in query_lower:
                        hours_match = re.search(r'(\d+)\s*hours?', query_lower)
                        if hours_match:
                            parameters['hours_back'] = int(hours_match.group(1))
                    
                    if 'days' in query_lower:
                        days_match = re.search(r'(\d+)\s*days?', query_lower)
                        if days_match:
                            parameters['days_back'] = int(days_match.group(1))
                    
                    return task_type, parameters
        
        # Default to comprehensive analysis if symbols are provided
        if symbols:
            return 'comprehensive_analysis', {}
        else:
            return 'market_overview', {}
    
    def _create_execution_plan(self, task_type: str, symbols: List[str]) -> List[str]:
        """Create execution plan based on task type"""
        plan = []
        
        if task_type in ['stock_quotes', 'market_overview', 'comprehensive_analysis']:
            plan.append('Collect stock quotes and market data via API')
            if symbols:
                plan.append(f'Get detailed information for symbols: {", ".join(symbols)}')
        
        if task_type in ['economic_indicators', 'market_overview', 'comprehensive_analysis']:
            plan.append('Fetch economic indicators from FRED API')
        
        if task_type in ['earnings', 'comprehensive_analysis']:
            plan.append('Scrape earnings data and analysis')
        
        if task_type in ['sec_filings', 'comprehensive_analysis']:
            plan.append('Collect SEC filings data')
        
        if task_type in ['news_analysis', 'market_overview', 'comprehensive_analysis']:
            plan.append('Scrape financial news and analyze sentiment')
        
        plan.append('Combine and analyze all collected data')
        
        return plan
    
    def _generate_insights(self, api_results: List[Dict], scraping_results: List[Dict], task_type: str) -> Dict[str, Any]:
        """Generate insights from combined results"""
        insights = {
            'data_quality': {},
            'market_sentiment': {},
            'key_findings': []
        }
        
        # Analyze data quality
        total_api_tasks = len(api_results)
        total_scraping_tasks = len(scraping_results)
        
        insights['data_quality'] = {
            'api_tasks_completed': total_api_tasks,
            'scraping_tasks_completed': total_scraping_tasks,
            'overall_success_rate': 1.0 if (total_api_tasks + total_scraping_tasks) > 0 else 0.0
        }
        
        # Analyze market sentiment from news
        for result in scraping_results:
            if result['task'] == 'news_sentiment':
                metadata = result.get('metadata', {})
                insights['market_sentiment'] = {
                    'overall_sentiment': metadata.get('overall_market_sentiment', 'neutral'),
                    'articles_analyzed': metadata.get('articles_analyzed', 0),
                    'positive_articles': metadata.get('positive_articles', 0),
                    'negative_articles': metadata.get('negative_articles', 0)
                }
        
        # Generate key findings
        if task_type == 'comprehensive_analysis':
            insights['key_findings'].append('Comprehensive analysis completed across multiple data sources')
        
        if any(r['task'] == 'stock_info' for r in api_results):
            insights['key_findings'].append('Stock market data successfully collected')
        
        if any(r['task'] == 'earnings_analysis' for r in scraping_results):
            insights['key_findings'].append('Earnings analysis and sentiment data obtained')
        
        return insights
    
    def _create_summary(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of results"""
        summary = {
            'execution_summary': f"Successfully executed {final_results['task_type']} analysis",
            'data_sources_used': [],
            'symbols_processed': len(final_results['symbols_analyzed']),
            'timestamp': final_results['timestamp']
        }
        
        # Identify data sources used
        if final_results['api_data']:
            summary['data_sources_used'].extend(['Alpha Vantage', 'FRED', 'Yahoo Finance'])
        
        if final_results['scraping_data']:
            summary['data_sources_used'].extend(['SEC EDGAR', 'Seeking Alpha', 'Financial News RSS'])
        
        summary['data_sources_used'] = list(set(summary['data_sources_used']))
        
        return summary
    
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
                'query': 'Get stock quotes for AAPL and MSFT with recent earnings analysis',
                'symbols': ['AAPL', 'MSFT']
            },
            {
                'query': 'Market overview with economic indicators and news sentiment',
                'symbols': []
            },
            {
                'query': 'Comprehensive analysis for GOOGL including SEC filings',
                'symbols': ['GOOGL']
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