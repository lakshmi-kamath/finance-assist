import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

class AlphaVantageCollector:
    """Collects real-time and historical market data from Alpha Vantage API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.logger = logging.getLogger(__name__)
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote"""
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'volume': int(quote.get('06. volume', 0)),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'alphavantage'
                }
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_earnings_data(self, symbol: str) -> List[Dict]:
        """Get earnings data for a symbol"""
        params = {
            'function': 'EARNINGS',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            earnings_list = []
            if 'quarterlyEarnings' in data:
                for earning in data['quarterlyEarnings'][:4]:  # Last 4 quarters
                    earnings_list.append({
                        'symbol': symbol,
                        'fiscal_date_ending': earning.get('fiscalDateEnding'),
                        'reported_eps': float(earning.get('reportedEPS', 0)),
                        'estimated_eps': float(earning.get('estimatedEPS', 0)),
                        'surprise': float(earning.get('surprise', 0)),
                        'surprise_percentage': float(earning.get('surprisePercentage', 0)),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'alphavantage_earnings'
                    })
            
            return earnings_list
        except Exception as e:
            self.logger.error(f"Error fetching earnings for {symbol}: {e}")
            return []