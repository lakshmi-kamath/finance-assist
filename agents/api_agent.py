import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import dotenv
dotenv.load_dotenv()

from base_agent import BaseAgent, Task, AgentResult, TaskPriority
from data_ingestion.api_collectors.alpha_vantage_collector import AlphaVantageCollector
from data_ingestion.api_collectors.fred_collector import FREDCollector
from data_ingestion.api_collectors.yahoo_finance_collector import YahooFinanceCollector

class APIAgent(BaseAgent):
    """Agent that wraps all API-based data collectors"""
    
    def __init__(self, agent_id: str = "api_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Initialize collectors based on config
        self.alpha_vantage = None
        self.fred = None
        self.yahoo_finance = YahooFinanceCollector()  # No API key needed
        
        # Initialize API collectors if keys are available
        self._initialize_collectors()
    
    def _define_capabilities(self) -> List[str]:
        """Define what this agent can do"""
        return [
            'get_stock_quotes',
            'get_earnings_data',
            'get_economic_indicators',
            'get_asia_tech_stocks',
            'get_stock_info',
            'test_api_connections',
            'get_market_overview'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['alpha_vantage_api', 'fred_api', 'yahoo_finance']
    
    def _initialize_collectors(self):
        """Initialize API collectors with keys from config"""
        try:
            # Alpha Vantage
            if self.config.get('alphavantage_api_key'):
                self.alpha_vantage = AlphaVantageCollector(
                    api_key=self.config['alphavantage_api_key']
                )
                self.logger.info("Alpha Vantage collector initialized")
            else:
                self.logger.warning("No Alpha Vantage API key provided")
            
            # FRED
            if self.config.get('fred_api_key'):
                self.fred = FREDCollector(
                    api_key=self.config['fred_api_key']
                )
                self.logger.info("FRED collector initialized")
            else:
                self.logger.warning("No FRED API key provided")
                # Initialize with demo mode
                self.fred = FREDCollector(api_key='demo')
                
        except Exception as e:
            self.logger.error(f"Error initializing collectors: {e}")
    
    async def execute_task(self, task: Task) -> AgentResult:
        """Execute API-related tasks"""
        task_type = task.type
        parameters = task.parameters
        
        try:
            if task_type == 'get_stock_quotes':
                return await self._get_stock_quotes(parameters)
            
            elif task_type == 'get_earnings_data':
                return await self._get_earnings_data(parameters)
            
            elif task_type == 'get_economic_indicators':
                return await self._get_economic_indicators(parameters)
            
            elif task_type == 'get_asia_tech_stocks':
                return await self._get_asia_tech_stocks(parameters)
            
            elif task_type == 'get_stock_info':
                return await self._get_stock_info(parameters)
            
            elif task_type == 'test_api_connections':
                return await self._test_api_connections()
            
            elif task_type == 'get_market_overview':
                return await self._get_market_overview(parameters)
            
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
    
    async def _get_stock_quotes(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get stock quotes for given symbols"""
        symbols = parameters.get('symbols', [])
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        if not self.alpha_vantage:
            return AgentResult(success=False, error="Alpha Vantage not initialized")
        
        quotes = []
        for symbol in symbols:
            try:
                quote = self.alpha_vantage.get_stock_quote(symbol)
                if quote:
                    quotes.append(quote)
            except Exception as e:
                self.logger.error(f"Error getting quote for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=quotes,
            metadata={'symbols_requested': len(symbols), 'quotes_retrieved': len(quotes)}
        )
    
    async def _get_earnings_data(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get earnings data for given symbols"""
        symbols = parameters.get('symbols', [])
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        if not self.alpha_vantage:
            return AgentResult(success=False, error="Alpha Vantage not initialized")
        
        all_earnings = []
        for symbol in symbols:
            try:
                earnings = self.alpha_vantage.get_earnings_data(symbol)
                all_earnings.extend(earnings)
            except Exception as e:
                self.logger.error(f"Error getting earnings for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=all_earnings,
            metadata={'symbols_requested': len(symbols), 'earnings_records': len(all_earnings)}
        )
    
    async def _get_economic_indicators(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get economic indicators from FRED"""
        days_back = parameters.get('days_back', 30)
        
        if not self.fred:
            return AgentResult(success=False, error="FRED collector not initialized")
        
        try:
            indicators = self.fred.get_economic_indicators(days_back=days_back)
            
            return AgentResult(
                success=True,
                data=indicators,
                metadata={
                    'indicators_count': len(indicators),
                    'days_back': days_back,
                    'data_sources': list(set(ind.get('source', 'unknown') for ind in indicators))
                }
            )
        except Exception as e:
            self.logger.error(f"Error fetching economic indicators: {e}")
            return AgentResult(success=False, error=str(e))
    
    async def _get_asia_tech_stocks(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get Asia tech stock symbols"""
        try:
            symbols = self.yahoo_finance.get_asia_tech_stocks()
            return AgentResult(
                success=True,
                data=symbols,
                metadata={'symbols_count': len(symbols)}
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _get_stock_info(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get detailed stock information"""
        symbols = parameters.get('symbols')
        
        if not symbols:
            # Get default Asia tech stocks if no symbols provided
            symbols = self.yahoo_finance.get_asia_tech_stocks()
        
        try:
            stock_data = self.yahoo_finance.get_stock_info(symbols)
            return AgentResult(
                success=True,
                data=stock_data,
                metadata={
                    'symbols_requested': len(symbols),
                    'stocks_retrieved': len(stock_data)
                }
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _test_api_connections(self) -> AgentResult:
        """Test all API connections"""
        results = {}
        
        # Test Alpha Vantage
        if self.alpha_vantage:
            try:
                test_quote = self.alpha_vantage.get_stock_quote('AAPL')
                results['alpha_vantage'] = {
                    'status': 'connected' if test_quote else 'no_data',
                    'message': 'Connection successful' if test_quote else 'No data returned'
                }
            except Exception as e:
                results['alpha_vantage'] = {
                    'status': 'error',
                    'message': str(e)
                }
        else:
            results['alpha_vantage'] = {
                'status': 'not_configured',
                'message': 'API key not provided'
            }
        
        # Test FRED
        if self.fred:
            try:
                fred_test = self.fred.test_api_connection()
                results['fred'] = fred_test
            except Exception as e:
                results['fred'] = {
                    'status': 'error',
                    'message': str(e)
                }
        else:
            results['fred'] = {
                'status': 'not_configured',
                'message': 'API key not provided'
            }
        
        # Test Yahoo Finance
        try:
            test_symbols = ['AAPL']
            yahoo_data = self.yahoo_finance.get_stock_info(test_symbols)
            results['yahoo_finance'] = {
                'status': 'connected' if yahoo_data else 'no_data',
                'message': 'Connection successful' if yahoo_data else 'No data returned'
            }
        except Exception as e:
            results['yahoo_finance'] = {
                'status': 'error',
                'message': str(e)
            }
        
        return AgentResult(
            success=True,
            data=results,
            metadata={'apis_tested': len(results)}
        )
    
    async def _get_market_overview(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get comprehensive market overview combining multiple data sources"""
        overview = {
            'asia_tech_stocks': [],
            'economic_indicators': [],
            'market_summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get Asia tech stocks
            asia_tech_task = await self._get_asia_tech_stocks({})
            if asia_tech_task.success:
                symbols = asia_tech_task.data[:5]  # Limit to top 5
                stock_info_task = await self._get_stock_info({'symbols': symbols})
                if stock_info_task.success:
                    overview['asia_tech_stocks'] = stock_info_task.data
            
            # Get economic indicators
            econ_task = await self._get_economic_indicators({'days_back': 30})
            if econ_task.success:
                overview['economic_indicators'] = econ_task.data[:10]  # Top 10
            
            # Create market summary
            if overview['asia_tech_stocks']:
                total_market_cap = sum(stock.get('market_cap', 0) for stock in overview['asia_tech_stocks'])
                avg_pe_ratio = sum(stock.get('pe_ratio', 0) for stock in overview['asia_tech_stocks'] if stock.get('pe_ratio', 0) > 0) / len([s for s in overview['asia_tech_stocks'] if s.get('pe_ratio', 0) > 0])
                
                overview['market_summary'] = {
                    'total_market_cap': total_market_cap,
                    'average_pe_ratio': avg_pe_ratio,
                    'stocks_analyzed': len(overview['asia_tech_stocks']),
                    'economic_indicators_count': len(overview['economic_indicators'])
                }
            
            return AgentResult(
                success=True,
                data=overview,
                metadata={
                    'data_sources': ['yahoo_finance', 'fred', 'alpha_vantage'],
                    'overview_sections': len(overview)
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including API status"""
        base_health = super().health_check()
        
        # Add API-specific health info
        api_status = {
            'alpha_vantage': self.alpha_vantage is not None,
            'fred': self.fred is not None,
            'yahoo_finance': True  # Always available
        }
        
        base_health['api_collectors'] = api_status
        base_health['healthy'] = base_health['healthy'] and any(api_status.values())
        
        return base_health


# Utility function to create configured API agent
def create_api_agent(config: Dict[str, Any] = None) -> APIAgent:
    """Factory function to create configured API agent"""
    default_config = {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY'),
        'fred_api_key': os.getenv('FRED_API_KEY'),
        'max_retries': 3,
        'timeout_seconds': 300
    }
    
    if config:
        default_config.update(config)
    
    return APIAgent(config=default_config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_api_agent():
        """Test the API agent functionality"""
        agent = create_api_agent()
        
        print(f"Agent Status: {agent.get_status()}")
        
        # Test API connections
        print("\n--- Testing API Connections ---")
        result = await agent.execute('test_api_connections', {})
        print(f"Connection Test: {result.to_dict()}")
        
        # Test market overview
        print("\n--- Getting Market Overview ---")
        result = await agent.execute('get_market_overview', {})
        if result.success:
            print(f"Market Overview Retrieved: {len(result.data)} sections")
        else:
            print(f"Error: {result.error}")
        
        print(f"\nFinal Agent Status: {agent.get_status()}")
    
    # Run test
    asyncio.run(test_api_agent())