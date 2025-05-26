import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

class FREDCollector:
    """Collects economic data from Federal Reserve Economic Data (FRED)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.logger = logging.getLogger(__name__)
        
        # Key economic indicators for Asia tech analysis
        self.key_indicators = {
            'GDP': 'GDP',
            'INFLATION': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE',
            'INTEREST_RATE': 'FEDFUNDS',
            'VIX': 'VIXCLS',
            'USD_INDEX': 'DTWEXBGS',
            'ASIA_YIELDS': 'DGS10',  # 10-Year Treasury for comparison
            'TECH_SECTOR_PE': 'MULTPL/SHILLER_PE_RATIO_MONTH'
        }
    
    def get_economic_indicators(self, days_back: int = 30) -> List[Dict]:
        """Get recent economic indicators relevant to market analysis"""
        indicators = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for indicator_name, series_id in self.key_indicators.items():
            try:
                data = self._fetch_series_data(
                    series_id, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if data:
                    for observation in data[-5:]:  # Last 5 observations
                        indicators.append({
                            'indicator_name': indicator_name,
                            'series_id': series_id,
                            'date': observation['date'],
                            'value': float(observation['value']) if observation['value'] != '.' else None,
                            'content_type': 'economic_indicator',
                            'source': 'fred',
                            'timestamp': datetime.now().isoformat(),
                            'tags': ['economics', 'macro', indicator_name.lower()],
                            'relevance_score': self._calculate_relevance(indicator_name)
                        })
                        
            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name} ({series_id}): {e}")
        
        return indicators
    
    def _fetch_series_data(self, series_id: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch data for a specific FRED series"""
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'desc',
            'limit': 10
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('observations', [])
        else:
            self.logger.error(f"FRED API error for {series_id}: {response.status_code}")
            return []
    
    def _calculate_relevance(self, indicator_name: str) -> float:
        """Calculate relevance score for Asia tech analysis"""
        relevance_weights = {
            'INTEREST_RATE': 0.9,  # High impact on tech valuations
            'VIX': 0.8,           # Market volatility indicator
            'USD_INDEX': 0.7,     # Affects Asian markets
            'INFLATION': 0.6,
            'GDP': 0.5,
            'UNEMPLOYMENT': 0.4,
            'ASIA_YIELDS': 0.8,
            'TECH_SECTOR_PE': 0.9
        }
        return relevance_weights.get(indicator_name, 0.5)
