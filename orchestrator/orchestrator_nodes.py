"""
Node definitions for the orchestrator agent workflow.
This module contains all the node functions used in the LangGraph workflow.
"""

from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime

from agents.base_agent import AgentResult


class OrchestratorNodes:
    """Container class for orchestrator workflow nodes"""
    
    def __init__(self, api_agent, scraping_agent, logger: logging.Logger):
        """
        Initialize orchestrator nodes with dependencies
        
        Args:
            api_agent: The API agent instance
            scraping_agent: The scraping agent instance  
            logger: Logger instance for error handling
        """
        self.api_agent = api_agent
        self.scraping_agent = scraping_agent
        self.logger = logger
        
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
    
    async def analyze_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def execute_api_tasks_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def execute_scraping_tasks_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
    
    async def combine_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
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