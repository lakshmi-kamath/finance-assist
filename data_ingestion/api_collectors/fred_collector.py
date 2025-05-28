import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

class FREDCollector:
    """Collects economic data from Federal Reserve Economic Data (FRED)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.logger = logging.getLogger(__name__)
        
        # Validated FRED series IDs that work with the API
        self.key_indicators = {
            'GDP': 'GDP',
            'INFLATION_CPI': 'CPIAUCSL',
            'UNEMPLOYMENT': 'UNRATE', 
            'FEDERAL_FUNDS_RATE': 'FEDFUNDS',
            'VIX_VOLATILITY': 'VIXCLS',
            'USD_INDEX': 'DTWEXBGS',
            'TREASURY_10Y': 'DGS10',
            'TREASURY_2Y': 'DGS2',
            'TREASURY_3M': 'DGS3MO',
            'INDUSTRIAL_PRODUCTION': 'INDPRO',
            'CONSUMER_SENTIMENT': 'UMCSENT',
            'HOUSING_STARTS': 'HOUST',
            'RETAIL_SALES': 'RSXFS',
            'CORE_PCE': 'PCEPILFE',
            'BREAKEVEN_10Y': 'T10YIE',
            'CORPORATE_AAA_SPREAD': 'AAA10Y',
            'HIGH_YIELD_SPREAD': 'BAMLH0A0HYM2'
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def get_economic_indicators(self, days_back: int = 30) -> List[Dict]:
        indicators = []
        end_date = datetime.now()
        
        # Adjust lookback based on frequency
        frequency_lookback = {
            'Quarterly': 365,  # 1 year for quarterly data
            'Monthly': 90,    # 3 months for monthly
            'Daily': days_back  # Use default for daily
        }
        
        self.logger.info(f"Fetching FRED data with dynamic lookback...")
        
        successful_fetches = set()
        
        for indicator_name, series_id in self.key_indicators.items():
            try:
                # Determine appropriate lookback
                data_frequency = self._get_data_frequency(series_id)
                lookback_days = frequency_lookback.get(data_frequency, days_back)
                start_date = end_date - timedelta(days=lookback_days)
                
                self._rate_limit()
                
                data = self._fetch_series_data(
                    series_id, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if data:
                    # Get valid observations
                    valid_observations = [obs for obs in data if obs['value'] != '.']
                    recent_observations = valid_observations[:3]
                    
                    for observation in recent_observations:
                        try:
                            value = float(observation['value'])
                            indicators.append({
                                'indicator_name': indicator_name,
                                'series_id': series_id,
                                'date': observation['date'],
                                'value': value,
                                'content_type': 'economic_indicator',
                                'source': 'fred',
                                'timestamp': datetime.now().isoformat(),
                                'tags': ['economics', 'macro', indicator_name.lower().replace('_', '_')],
                                'relevance_score': self._calculate_relevance(indicator_name),
                                'data_frequency': data_frequency,
                                'units': self._get_series_units(series_id)
                            })
                            successful_fetches.add(indicator_name)
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Invalid value for {indicator_name}: {observation['value']}")
                    
                    self.logger.debug(f"✅ {indicator_name}: {len(recent_observations)} observations")
                else:
                    self.logger.warning(f"No data for {indicator_name} ({series_id})")
                
            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name} ({series_id}): {e}")
        
        self.logger.info(f"FRED data collection: {len(successful_fetches)}/{len(self.key_indicators)} successful, {len(indicators)} observations")
        return indicators

    def _fetch_series_data(self, series_id: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch data for a specific FRED series with improved error handling"""
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
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if not observations:
                    self.logger.warning(f"No observations returned for series {series_id}")
                
                return observations
            
            elif response.status_code == 400:
                self.logger.error(f"Bad request for series {series_id} - possibly invalid series ID")
                return []
            
            elif response.status_code == 429:
                self.logger.warning(f"Rate limit hit for series {series_id}, waiting...")
                time.sleep(1)
                return self._fetch_series_data(series_id, start_date, end_date)
            
            else:
                self.logger.error(f"FRED API error for {series_id}: HTTP {response.status_code}")
                if response.text:
                    self.logger.error(f"Response: {response.text[:200]}")
                return []
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching data for series {series_id}")
            return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for series {series_id}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {series_id}: {e}")
            return []
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API throttling"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _calculate_relevance(self, indicator_name: str) -> float:
        """Calculate relevance score for Asia tech analysis"""
        relevance_weights = {
            'FEDERAL_FUNDS_RATE': 0.95,  # Critical for tech valuations
            'VIX_VOLATILITY': 0.90,      # Market volatility indicator
            'USD_INDEX': 0.85,           # Affects Asian markets directly
            'TREASURY_10Y': 0.85,        # Tech sector discount rate
            'TREASURY_2Y': 0.80,         # Yield curve analysis
            'CORE_PCE': 0.75,           # Fed's preferred inflation measure
            'INFLATION_CPI': 0.70,       # Traditional inflation measure
            'CONSUMER_SENTIMENT': 0.65,  # Consumer spending on tech
            'INDUSTRIAL_PRODUCTION': 0.60, # Economic activity
            'GDP': 0.55,                 # Overall economic health
            'UNEMPLOYMENT': 0.50,        # Labor market
            'HOUSING_STARTS': 0.40,      # Less relevant for tech
            'RETAIL_SALES': 0.45,        # Consumer spending
            'TREASURY_3M': 0.70,         # Short-term rates
            'BREAKEVEN_10Y': 0.65,       # Inflation expectations
            'CORPORATE_AAA_SPREAD': 0.80, # Credit conditions
            'HIGH_YIELD_SPREAD': 0.75    # Risk appetite
        }
        return relevance_weights.get(indicator_name, 0.5)
    
    def _get_data_frequency(self, series_id: str) -> str:
        """Get typical frequency for different series"""
        daily_series = ['VIXCLS', 'DGS10', 'DGS2', 'DGS3MO', 'DTWEXBGS']
        monthly_series = ['CPIAUCSL', 'UNRATE', 'INDPRO', 'UMCSENT', 'HOUST', 'RSXFS', 'PCEPILFE']
        quarterly_series = ['GDP']
        
        if series_id in daily_series:
            return 'Daily'
        elif series_id in monthly_series:
            return 'Monthly'
        elif series_id in quarterly_series:
            return 'Quarterly'
        else:
            return 'Unknown'
    
    def _get_series_units(self, series_id: str) -> str:
        """Get units for different series"""
        units_map = {
            'GDP': 'Billions of Dollars',
            'CPIAUCSL': 'Index 1982-84=100',
            'UNRATE': 'Percent',
            'FEDFUNDS': 'Percent',
            'VIXCLS': 'Index',
            'DTWEXBGS': 'Index Jan-1997=100',
            'DGS10': 'Percent',
            'DGS2': 'Percent',
            'DGS3MO': 'Percent',
            'INDPRO': 'Index 2017=100',
            'UMCSENT': 'Index 1966:Q1=100',
            'HOUST': 'Thousands of Units',
            'RSXFS': 'Millions of Dollars',
            'PCEPILFE': 'Index 2012=100'
        }
        return units_map.get(series_id, 'Units not specified')
    
    def _get_demo_economic_data(self, days_back: int) -> List[Dict]:
        """Generate demo economic data when API key is not available"""
        demo_data = []
        base_date = datetime.now() - timedelta(days=days_back//2)
        
        demo_indicators = [
            {'name': 'TREASURY_10Y', 'value': 4.25, 'relevance': 0.85},
            {'name': 'VIX_VOLATILITY', 'value': 18.5, 'relevance': 0.90},
            {'name': 'FEDERAL_FUNDS_RATE', 'value': 5.25, 'relevance': 0.95},
            {'name': 'INFLATION_CPI', 'value': 3.2, 'relevance': 0.70},
            {'name': 'UNEMPLOYMENT', 'value': 3.8, 'relevance': 0.50},
        ]
        
        for indicator in demo_indicators:
            demo_data.append({
                'indicator_name': indicator['name'],
                'series_id': f"DEMO_{indicator['name']}",
                'date': base_date.strftime('%Y-%m-%d'),
                'value': indicator['value'],
                'content_type': 'economic_indicator',
                'source': 'fred_demo',
                'timestamp': datetime.now().isoformat(),
                'tags': ['economics', 'macro', 'demo', indicator['name'].lower().replace('_', '-')],
                'relevance_score': indicator['relevance'],
                'data_frequency': 'Demo',
                'units': 'Demo Units'
            })
        
        self.logger.info(f"Generated {len(demo_data)} demo economic indicators")
        return demo_data
    
    def test_api_connection(self) -> Dict:
        """Test FRED API connection and return status"""
        if not self.api_key or self.api_key == 'demo':
            # Switch to demo mode
            self.logger.info("No API key provided, using demo data")
            return {
                'status': 'demo_mode',
                'message': 'Using demo data - economic indicators will be simulated',
                'api_key_valid': False
            }
        try:
            # Test with a simple, reliable series
            test_url = f"{self.base_url}/series/observations"
            params = {
                'series_id': 'GDP',
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': 1
            }
            
            response = requests.get(test_url, params=params, timeout=5)
            
            if response.status_code == 200:
                return {
                    'status': 'connected',
                    'message': 'FRED API connection successful',
                    'api_key_valid': True
                }
            else:
                return {
                    'status': 'error',
                    'message': f'FRED API error: HTTP {response.status_code}',
                    'api_key_valid': False
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connection test failed: {str(e)}',
                'api_key_valid': False
            }