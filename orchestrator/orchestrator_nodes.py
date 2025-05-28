"""
Simplified node definitions for the orchestrator agent workflow.
Single class with all necessary functionality for coordinating API, scraping, and retrieval agents.
"""


from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime

from agents.base_agent import AgentResult, Task


class OrchestratorNodes:
    """Unified orchestrator workflow nodes with API, scraping, and retrieval capabilities"""
    
    def __init__(self, api_agent, scraping_agent, retriever_agent, logger: logging.Logger):
        """
        Initialize orchestrator nodes with all required dependencies
        
        Args:
            api_agent: The API agent instance
            scraping_agent: The scraping agent instance
            retriever_agent: The retriever agent instance
            logger: Logger instance for error handling
        """
        self.api_agent = api_agent
        self.scraping_agent = scraping_agent
        self.retriever_agent = retriever_agent
        self.logger = logger
        
        # Query patterns for intelligent routing
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
        """Analyze user query and determine execution strategy"""
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
        """Execute all API-related data collection tasks"""
        symbols = state['symbols']
        task_parameters = state['task_parameters']
        
        api_results = []
        
        try:
            # Stock data collection
            if symbols:
                # Get current stock quotes
                result = await self.api_agent.execute('get_stock_quotes', {'symbols': symbols})
                if result.success:
                    api_results.append({
                        'task': 'stock_quotes',
                        'data': result.data,
                        'metadata': result.metadata
                    })
                
                # Get detailed stock information
                result = await self.api_agent.execute('get_stock_info', {'symbols': symbols})
                if result.success:
                    api_results.append({
                        'task': 'stock_info',
                        'data': result.data,
                        'metadata': result.metadata
                    })
            
            # Economic indicators for market context
            days_back = task_parameters.get('days_back', 30)
            result = await self.api_agent.execute('get_economic_indicators', {'days_back': days_back})
            if result.success:
                api_results.append({
                    'task': 'economic_indicators',
                    'data': result.data,
                    'metadata': result.metadata
                })
            
            # Market overview
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
        """Execute all scraping-related data collection tasks"""
        symbols = state['symbols']
        task_parameters = state['task_parameters']
        task_type = state['task_type']
        
        scraping_results = []
        
        try:
            # Earnings data collection
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
                
                # SEC filings for fundamental analysis
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
            
            # Financial news and sentiment analysis
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
            
            # Comprehensive company analysis for detailed requests
            if symbols and (task_type == 'comprehensive_analysis' or len(symbols) <= 3):
                result = await self.scraping_agent.execute('comprehensive_company_analysis', {
                    'symbols': symbols[:3]  # Limit to prevent timeout
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
    
    async def enhance_with_retrieval_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results with additional retrieval-based analysis"""
        try:
            # Use specific retrieval based on symbols and analysis type
            symbols = state.get('symbols', [])
            if symbols:
                # Portfolio-specific retrieval
                portfolio_task = Task(
                    type='portfolio_analysis_retrieval',  # Use proper task type
                    parameters={
                        'query': f"analysis insights for {' '.join(symbols)}",
                        'context': {
                            'portfolio_tickers': symbols,
                            'focus_sectors': [],
                            'time_preference': 'recent',
                        },
                        'max_results': 10,
                        'time_window_hours': 168,  # 7 days
                    }
                )
                
                portfolio_result = await self.retriever_agent.execute_task(portfolio_task)
                
                if portfolio_result.success:
                    state['retrieval_results'].append({
                        'task': 'portfolio_enhancement',
                        'data': portfolio_result.data,
                        'metadata': portfolio_result.metadata
                    })
            
        except Exception as e:
            self.logger.error(f"Error in enhance_with_retrieval_node: {e}")
        
        return state
    async def retrieve_context_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context using RetrieverAgent"""
        try:
            # Create proper retrieval task based on integration guide
            retrieval_task = Task(
                id=f"portfolio_retrieval_{datetime.now().isoformat()}",
                type='contextual_retrieval',  # Use proper task type
                parameters={
                    'query': state['user_query'],
                    'context': {
                        'portfolio_tickers': state.get('symbols', []),
                        'time_preference': 'recent',
                    },
                    'max_results': 15,
                    'time_window_hours': state.get('query_context', {}).get('time_window_hours', 72),
                }
            )
            
            result = await self.retriever_agent.execute_task(retrieval_task)
            
            if result.success:
                state['retrieved_context'] = result.data
                state['retrieval_results'].append({
                    'task': 'context_retrieval',
                    'data': result.data,
                    'metadata': result.metadata
                })
            else:
                self.logger.warning(f"Context retrieval failed: {result.error}")
                state['retrieved_context'] = {}
                
        except Exception as e:
            self.logger.error(f"Error in retrieve_context_node: {e}")
            state['retrieved_context'] = {}
        
        return state
    def _determine_context_type(self, query: str, query_context: Dict[str, Any]) -> str:
        """Determine the best context type for retrieval based on query and context"""
        query_lower = query.lower()
        analysis_type = query_context.get('analysis_type', '').lower()
        
        # Check for specific analysis types
        if 'earnings' in query_lower or 'earnings' in analysis_type:
            return 'earnings'
        elif 'portfolio' in query_lower or 'portfolio' in analysis_type:
            return 'portfolio'
        elif any(risk_term in query_lower for risk_term in ['risk', 'volatility', 'correlation', 'beta']):
            return 'risk'
        elif any(news_term in query_lower for news_term in ['news', 'sentiment', 'media', 'social']):
            return 'news'
        else:
            return 'general'

    def _build_retrieval_query(self, user_query: str, symbols: List[str], context: Dict[str, Any]) -> str:
        """Build optimized retrieval query"""
        # Start with user query
        query_parts = [user_query]
        
        # Add symbols if available
        if symbols:
            query_parts.append(f"for {' '.join(symbols)}")
        
        # Add context-specific terms
        analysis_type = context.get('analysis_type', '')
        if analysis_type:
            query_parts.append(f"{analysis_type} analysis")
        
        # Add time preference
        if context.get('include_historical'):
            query_parts.append("with historical data")
        
        return ' '.join(query_parts)
    async def combine_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Combine and synthesize all collected data into final results"""
        api_results = state['api_results']
        scraping_results = state['scraping_results']
        retrieval_results = state.get('retrieval_results', [])
        retrieved_context = state.get('retrieved_context', {})
        task_type = state['task_type']
        symbols = state['symbols']
        
        final_results = {
            'task_type': task_type,
            'symbols_analyzed': symbols,
            'timestamp': datetime.now().isoformat(),
            'api_data': {},
            'scraping_data': {},
            'retrieval_data': {},
            'retrieved_context': retrieved_context,
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
        
        # Process retrieval results
        for result in retrieval_results:
            final_results['retrieval_data'][result['task']] = {
                'data': result['data'],
                'metadata': result['metadata']
            }
        
        # Generate comprehensive insights
        final_results['insights'] = self._generate_insights(
            api_results, scraping_results, retrieval_results, retrieved_context, task_type
        )
        
        # Create executive summary
        final_results['summary'] = self._create_summary(final_results)
        
        return {
            **state,
            'final_results': final_results
        }
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query text"""
        # Skip for general market queries
        if any(term in query.lower() for term in ['market overview', 'economic indicators', 'general market']):
            return []
        
        # Pattern matching for stock symbols
        symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # Standard symbols
            r'\b[A-Z]{1,4}\.[A-Z]{1,3}\b',  # Exchange-specific symbols
        ]
        
        symbols = []
        for pattern in symbol_patterns:
            matches = re.findall(pattern, query.upper())
            symbols.extend(matches)
        
        # Filter out common words
        excluded_words = {
            'THE', 'AND', 'OR', 'BUT', 'FOR', 'GET', 'API', 'SEC', 'NYSE', 'NEWS',
            'WITH', 'MARKET', 'OVERVIEW', 'ECONOMIC', 'INDICATORS', 'SENTIMENT', 'ANALYSIS'
        }
        
        return list(set(s for s in symbols if s not in excluded_words))
    
    def _determine_task_type(self, query: str, symbols: List[str]) -> tuple[str, Dict[str, Any]]:
        """Determine appropriate task type and parameters"""
        query_lower = query.lower()
        
        # Pattern matching for task types
        for task_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    parameters = self._extract_time_parameters(query_lower)
                    return task_type, parameters
        
        # Default logic
        if symbols:
            return 'comprehensive_analysis', {}
        else:
            return 'market_overview', {}
    
    def _extract_time_parameters(self, query: str) -> Dict[str, Any]:
        """Extract time-based parameters from query"""
        parameters = {}
        
        # Extract hours
        hours_match = re.search(r'(\d+)\s*hours?', query)
        if hours_match:
            parameters['hours_back'] = int(hours_match.group(1))
        
        # Extract days
        days_match = re.search(r'(\d+)\s*days?', query)
        if days_match:
            parameters['days_back'] = int(days_match.group(1))
        
        return parameters
    
    def _create_execution_plan(self, task_type: str, symbols: List[str]) -> List[str]:
        """Create detailed execution plan"""
        plan = []
        
        # Context retrieval
        plan.append('Retrieve historical context and relevant data')
        
        # API tasks
        if task_type in ['stock_quotes', 'market_overview', 'comprehensive_analysis']:
            plan.append('Collect real-time market data via APIs')
            if symbols:
                plan.append(f'Get detailed stock information for: {", ".join(symbols)}')
        
        if task_type in ['economic_indicators', 'market_overview', 'comprehensive_analysis']:
            plan.append('Fetch economic indicators from FRED')
        
        # Scraping tasks
        if task_type in ['earnings', 'comprehensive_analysis'] and symbols:
            plan.append('Scrape earnings data and analysis')
        
        if task_type in ['sec_filings', 'comprehensive_analysis'] and symbols:
            plan.append('Collect SEC filings and regulatory data')
        
        if task_type in ['news_analysis', 'market_overview', 'comprehensive_analysis']:
            plan.append('Analyze financial news and market sentiment')
        
        # Retrieval enhancement
        plan.append('Enhance analysis with historical performance data')
        if symbols and len(symbols) <= 5:
            plan.append('Perform peer comparison analysis')
        
        # Final synthesis
        plan.append('Combine and synthesize all data sources')
        
        return plan
    
    def _build_retrieval_query(self, query: str, symbols: List[str], context: Dict[str, Any]) -> str:
        """Build optimized query for retrieval agent"""
        search_terms = [query]
        
        if symbols:
            search_terms.extend(symbols)
        
        # Add context-specific terms
        if context.get('analysis_type'):
            search_terms.append(context['analysis_type'])
        
        if context.get('include_historical'):
            search_terms.append('historical performance trends')
        
        if context.get('include_peer_analysis'):
            search_terms.append('peer comparison industry analysis')
        
        return ' '.join(search_terms)
    
    def _generate_insights(self, api_results: List[Dict], scraping_results: List[Dict], 
                          retrieval_results: List[Dict], retrieved_context: Dict[str, Any], 
                          task_type: str) -> Dict[str, Any]:
        """Generate comprehensive insights from all data sources"""
        insights = {
            'data_quality': {
                'api_tasks_completed': len(api_results),
                'scraping_tasks_completed': len(scraping_results),
                'retrieval_tasks_completed': len(retrieval_results),
                'context_retrieved': bool(retrieved_context),
                'context_confidence': retrieved_context.get('confidence_score', 0.0)
            },
            'market_sentiment': {},
            'key_findings': [],
            'retrieval_insights': {
                'historical_context_available': bool(retrieved_context),
                'peer_comparison_available': any(r['task'] == 'peer_comparison' for r in retrieval_results),
                'sector_analysis_available': any(r['task'] == 'sector_trends' for r in retrieval_results)
            }
        }
        
        # Extract market sentiment from news analysis
        for result in scraping_results:
            if result['task'] == 'news_sentiment':
                metadata = result.get('metadata', {})
                insights['market_sentiment'] = {
                    'overall_sentiment': metadata.get('overall_market_sentiment', 'neutral'),
                    'articles_analyzed': metadata.get('articles_analyzed', 0),
                    'confidence': metadata.get('sentiment_confidence', 0.0)
                }
        
        # Generate key findings based on available data
        if task_type == 'comprehensive_analysis':
            insights['key_findings'].append('Comprehensive multi-source analysis completed')
        
        if api_results:
            insights['key_findings'].append('Real-time market data successfully collected')
        
        if scraping_results:
            insights['key_findings'].append('Web-scraped financial data obtained')
        
        if retrieval_results:
            insights['key_findings'].append('Historical context and comparative analysis available')
        
        if retrieved_context and retrieved_context.get('confidence_score', 0) > 0.7:
            insights['key_findings'].append('High-confidence contextual information retrieved')
        
        return insights
    
    def _create_summary(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of the analysis"""
        summary = {
            'execution_summary': f"Successfully executed {final_results['task_type']} analysis",
            'data_sources_used': [],
            'symbols_processed': len(final_results['symbols_analyzed']),
            'timestamp': final_results['timestamp'],
            'analysis_depth': 'comprehensive' if len(final_results['symbols_analyzed']) <= 3 else 'broad'
        }
        
        # Identify data sources
        if final_results['api_data']:
            summary['data_sources_used'].extend(['Alpha Vantage', 'FRED', 'Yahoo Finance'])
        
        if final_results['scraping_data']:
            summary['data_sources_used'].extend(['SEC EDGAR', 'Seeking Alpha', 'Financial News'])
        
        if final_results['retrieval_data']:
            summary['data_sources_used'].append('Historical Database')
        
        if final_results['retrieved_context']:
            summary['context_retrieved'] = True
            summary['context_confidence'] = final_results['retrieved_context'].get('confidence_score', 0.0)
        else:
            summary['context_retrieved'] = False
            summary['context_confidence'] = 0.0
        
        summary['data_sources_used'] = list(set(summary['data_sources_used']))
        
        return summary