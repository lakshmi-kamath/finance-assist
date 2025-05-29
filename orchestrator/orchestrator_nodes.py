
"""
Fixed orchestrator_nodes.py - Language Brief Generation Fix
Key fixes:
1. Handle both dict and OrchestratorState types in generate_language_brief_node
2. Fix attribute access/assignment to work with both types
3. Add better error handling and debugging
4. Ensure market data extraction works correctly
"""

from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime
from agents.base_agent import AgentResult, Task
from orchestrator.state import OrchestratorState
from agents.language_agent import MarketData

class OrchestratorNodes:
    """Unified orchestrator workflow nodes with API, scraping, and retrieval capabilities"""
    
    def __init__(self, api_agent, scraping_agent, retriever_agent, language_agent, logger):
        self.api_agent = api_agent
        self.scraping_agent = scraping_agent
        self.retriever_agent = retriever_agent
        self.language_agent = language_agent
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

    def _get_state_value(self, state, key: str, default=None):
        """Safely get value from state (works with both dict and OrchestratorState)"""
        if hasattr(state, key):
            return getattr(state, key, default)
        elif isinstance(state, dict):
            return state.get(key, default)
        else:
            return default

    def _set_state_value(self, state, key: str, value):
        """Safely set value in state (works with both dict and OrchestratorState)"""
        if hasattr(state, key):
            setattr(state, key, value)
        elif isinstance(state, dict):
            state[key] = value
        else:
            self.logger.warning(f"Cannot set {key} on state type {type(state)}")

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
                        'task_type': 'get_stock_quotes',
                        'data': result.data,
                        'metadata': result.metadata,
                        'success': True
                    })
                
                # Get detailed stock information
                result = await self.api_agent.execute('get_stock_info', {'symbols': symbols})
                if result.success:
                    api_results.append({
                        'task': 'stock_info',
                        'task_type': 'get_company_overview',
                        'data': result.data,
                        'metadata': result.metadata,
                        'success': True
                    })
            
            # Economic indicators for market context
            days_back = task_parameters.get('days_back', 30)
            result = await self.api_agent.execute('get_economic_indicators', {'days_back': days_back})
            if result.success:
                api_results.append({
                    'task': 'economic_indicators',
                    'task_type': 'get_economic_indicators',
                    'data': result.data,
                    'metadata': result.metadata,
                    'success': True
                })
            
            # Market overview
            result = await self.api_agent.execute('get_market_overview', {})
            if result.success:
                api_results.append({
                    'task': 'market_overview',
                    'task_type': 'get_market_overview',
                    'data': result.data,
                    'metadata': result.metadata,
                    'success': True
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
                        'task_type': 'batch_earnings_scrape',
                        'data': result.data,
                        'metadata': result.metadata,
                        'success': True
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
                        'task_type': 'scrape_sec_filings',
                        'data': result.data,
                        'metadata': result.metadata,
                        'success': True
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
                    'task_type': 'scrape_financial_news',
                    'data': result.data,
                    'metadata': result.metadata,
                    'success': True
                })
            
            # Comprehensive company analysis for detailed requests
            if symbols and (task_type == 'comprehensive_analysis' or len(symbols) <= 3):
                result = await self.scraping_agent.execute('comprehensive_company_analysis', {
                    'symbols': symbols[:3]  # Limit to prevent timeout
                })
                if result.success:
                    scraping_results.append({
                        'task': 'comprehensive_analysis',
                        'task_type': 'comprehensive_company_analysis',
                        'data': result.data,
                        'metadata': result.metadata,
                        'success': True
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
            # Initialize retrieval_results if not present
            if 'retrieval_results' not in state:
                state['retrieval_results'] = []
                
            # Use specific retrieval based on symbols and analysis type
            symbols = state.get('symbols', [])
            if symbols:
                # Portfolio-specific retrieval
                portfolio_task = Task(
                    id=f"portfolio_analysis_{datetime.now().isoformat()}",
                    type='portfolio_analysis_retrieval',
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
                        'task_type': 'portfolio_analysis_retrieval',
                        'data': portfolio_result.data,
                        'metadata': portfolio_result.metadata,
                        'success': True
                    })
            
        except Exception as e:
            self.logger.error(f"Error in enhance_with_retrieval_node: {e}")
        
        return state

    async def retrieve_context_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context using RetrieverAgent"""
        try:
            # Initialize retrieval_results if not present
            if 'retrieval_results' not in state:
                state['retrieval_results'] = []
                
            # Create proper retrieval task
            retrieval_task = Task(
                id=f"portfolio_retrieval_{datetime.now().isoformat()}",
                type='contextual_retrieval',
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
                    'task_type': 'contextual_retrieval',
                    'data': result.data,
                    'metadata': result.metadata,
                    'success': True
                })
            else:
                self.logger.warning(f"Context retrieval failed: {result.error}")
                state['retrieved_context'] = {}
                
        except Exception as e:
            self.logger.error(f"Error in retrieve_context_node: {e}")
            state['retrieved_context'] = {}
        
        return state

    async def combine_results_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Combine and synthesize all collected data into final results"""
        api_results = state['api_results']
        scraping_results = state['scraping_results']
        retrieval_results = state.get('retrieval_results', [])
        retrieved_context = state.get('retrieved_context', {})
        language_brief = state.get('language_brief', {})  # ADD THIS LINE
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
            'language_brief': language_brief,  # ADD THIS LINE
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
    
    async def generate_language_brief_node(self, state) -> Dict[str, Any]:
        """Generate language brief using all collected data - FIXED VERSION"""
        
        if not self.language_agent:
            self.logger.warning("Language agent not available, skipping brief generation")
            return state
        
        try:
            self.logger.info(f"Generating language brief - State type: {type(state)}")
            
            # Convert all results to MarketData format
            market_data = []
            
            # Process API results - USE SAFE GETTER
            api_results = self._get_state_value(state, 'api_results', [])
            self.logger.info(f"Found {len(api_results)} API results")
            
            for api_result in api_results:
                if api_result.get('success') and api_result.get('data'):
                    content = self._format_api_data_for_brief(
                        api_result['data'], 
                        api_result.get('task_type', 'unknown')
                    )
                    market_data.append(MarketData(
                        content=content,
                        data_type='api_data',
                        timestamp=datetime.now(),
                        importance=0.8
                    ))
                    self.logger.info(f"Added API data: {content[:100]}...")
            
            # Process scraping results - USE SAFE GETTER
            scraping_results = self._get_state_value(state, 'scraping_results', [])
            self.logger.info(f"Found {len(scraping_results)} scraping results")
            
            for scraping_result in scraping_results:
                if scraping_result.get('success') and scraping_result.get('data'):
                    content = self._format_scraping_data_for_brief(
                        scraping_result['data'], 
                        scraping_result.get('task_type', 'unknown')
                    )
                    market_data.append(MarketData(
                        content=content,
                        data_type='news',
                        timestamp=datetime.now(),
                        importance=0.7
                    ))
                    self.logger.info(f"Added scraping data: {content[:100]}...")
            
            # Process retrieval results - USE SAFE GETTER
            retrieval_results = self._get_state_value(state, 'retrieval_results', [])
            self.logger.info(f"Found {len(retrieval_results)} retrieval results")
            
            for retrieval_result in retrieval_results:
                if retrieval_result.get('success') and retrieval_result.get('data'):
                    content = self._format_retrieval_data_for_brief(retrieval_result['data'])
                    market_data.append(MarketData(
                        content=content,
                        data_type='historical',
                        timestamp=datetime.now(),
                        importance=0.6
                    ))
                    self.logger.info(f"Added retrieval data: {content[:100]}...")
            
            # Extract portfolio symbols if available - USE SAFE GETTER
            user_portfolio = self._get_state_value(state, 'symbols', [])
            
            self.logger.info(f"About to generate brief with {len(market_data)} market data points")
            
            # Generate the brief
            brief_result = await self.language_agent.generate_morning_brief(
                market_data=market_data,
                user_portfolio=user_portfolio if user_portfolio else None
            )
            
            # Add brief to state - USE SAFE SETTER
            self._set_state_value(state, 'language_brief', brief_result)
            
            self.logger.info(f"Generated language brief with {brief_result.get('data_points_used', 0)} data points used by language model")
            
        except Exception as e:
            self.logger.error(f"Error generating language brief: {e}", exc_info=True)
            fallback_brief = {
                'brief': f"Unable to generate brief due to error: {str(e)}",
                'error': str(e),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points_used': 0,
                'portfolio_focused': False
            }
            self._set_state_value(state, 'language_brief', fallback_brief)
        
        return state

    def _format_api_data_for_brief(self, api_data: Dict[str, Any], task_type: str) -> str:
        """Format API data for language brief with more specific formatting"""
        
        if task_type == 'get_stock_quotes':
            # Handle stock quote data
            if isinstance(api_data, dict):
                formatted_quotes = []
                for symbol, quote_data in api_data.items():
                    if isinstance(quote_data, dict):
                        price = quote_data.get('05. price', quote_data.get('price', 'N/A'))
                        change = quote_data.get('09. change', quote_data.get('change', 'N/A'))
                        formatted_quotes.append(f"{symbol}: ${price} (Change: {change})")
                return "Stock Quotes: " + "; ".join(formatted_quotes)
        
        elif task_type == 'get_company_overview':
            # Handle company overview data
            if isinstance(api_data, dict):
                name = api_data.get('Name', 'Unknown Company')
                sector = api_data.get('Sector', 'N/A')
                market_cap = api_data.get('MarketCapitalization', 'N/A')
                return f"Company Overview: {name} (Sector: {sector}, Market Cap: {market_cap})"
        
        elif task_type == 'get_economic_indicators':
            # Handle FRED economic data
            if isinstance(api_data, dict):
                indicators = []
                for key, value in api_data.items():
                    if isinstance(value, (int, float, str)):
                        indicators.append(f"{key}: {value}")
                return "Economic Indicators: " + "; ".join(indicators[:3])  # Limit to 3
        
        elif task_type == 'get_market_overview':
            # Handle market overview data
            if isinstance(api_data, dict):
                overview_items = []
                for key, value in api_data.items():
                    if isinstance(value, (int, float, str)):
                        overview_items.append(f"{key}: {value}")
                return "Market Overview: " + "; ".join(overview_items[:3])
        
        # Default fallback
        return f"API Data ({task_type}): {str(api_data)[:200]}..."

    def _format_scraping_data_for_brief(self, scraping_data: Dict[str, Any], task_type: str) -> str:
        """Format scraping data for language brief"""
        
        if task_type == 'scrape_sec_filings':
            # Handle SEC filings
            if isinstance(scraping_data, dict) and 'filings' in scraping_data:
                filings = scraping_data['filings'][:2]  # Recent 2 filings
                filing_info = []
                for filing in filings:
                    if isinstance(filing, dict):
                        form_type = filing.get('form_type', 'Unknown')
                        date = filing.get('date', 'N/A')
                        filing_info.append(f"{form_type} filed {date}")
                return "SEC Filings: " + "; ".join(filing_info)
        
        elif task_type == 'scrape_financial_news':
            # Handle financial news scraping
            if isinstance(scraping_data, dict):
                if 'headlines' in scraping_data:
                    headlines = scraping_data['headlines'][:3]  # Top 3 headlines
                    return "Market Headlines: " + "; ".join(headlines)
                elif 'articles' in scraping_data:
                    articles = scraping_data['articles'][:2]
                    summaries = []
                    for article in articles:
                        if isinstance(article, dict):
                            title = article.get('title', article.get('headline', 'Article'))
                            summaries.append(title)
                    return "News Articles: " + "; ".join(summaries)
        
        # Default fallback
        return f"Scraped Data ({task_type}): {str(scraping_data)[:200]}..."

    def _format_retrieval_data_for_brief(self, retrieval_data: Dict[str, Any]) -> str:
        """Format retrieval data for language brief"""
        
        if isinstance(retrieval_data, dict):
            chunks = retrieval_data.get('chunks', [])
            if chunks:
                # Get the top chunk with highest score
                top_chunk = chunks[0] if chunks else {}
                content = top_chunk.get('content', '')
                source = top_chunk.get('source', 'Historical Data')
                
                # Truncate content for brief
                truncated_content = content[:150] + "..." if len(content) > 150 else content
                return f"Historical Context from {source}: {truncated_content}"
            else:
                confidence = retrieval_data.get('confidence_score', 0.0)
                return f"Retrieved historical context with {confidence:.2f} confidence"
        
        return f"Historical Data: {str(retrieval_data)[:150]}..."

    # Include all other helper methods from the original class...
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
    
    # DEBUGGING HELPER - Add this to your orchestrator_nodes.py for debugging
    def debug_state_contents(self, state: OrchestratorState, step_name: str):
        """Debug helper to see what's in the state"""
        self.logger.info(f"=== DEBUG {step_name} ===")
        self.logger.info(f"API Results: {len(getattr(state, 'api_results', []) or [])}")
        self.logger.info(f"Scraping Results: {len(getattr(state, 'scraping_results', []) or [])}")
        self.logger.info(f"Retrieval Results: {len(getattr(state, 'retrieval_results', []) or [])}")
        self.logger.info(f"Symbols: {getattr(state, 'symbols', [])}")
        
        # Debug API results content
        api_results = getattr(state, 'api_results', []) or []
        for i, result in enumerate(api_results):
            self.logger.info(f"API Result {i}: success={result.get('success')}, task_type={result.get('task_type')}, data_keys={list(result.get('data', {}).keys()) if isinstance(result.get('data'), dict) else type(result.get('data'))}")
        
        # Debug scraping results content  
        scraping_results = getattr(state, 'scraping_results', []) or []
        for i, result in enumerate(scraping_results):
            self.logger.info(f"Scraping Result {i}: success={result.get('success')}, task_type={result.get('task_type')}, data_keys={list(result.get('data', {}).keys()) if isinstance(result.get('data'), dict) else type(result.get('data'))}")


    # ADD THIS CALL AT THE START OF YOUR generate_language_brief_node:
    # self.debug_state_contents(state, "LANGUAGE_BRIEF_START")