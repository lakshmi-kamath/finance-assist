import yfinance as yf
from typing import Dict, List
import pandas as pd
import logging
from datetime import datetime

class YahooFinanceCollector:
    """Collects market data from Yahoo Finance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_asia_tech_stocks(self) -> List[str]:
        """Get list of major Asia tech stocks"""
        return [
            'TSM',    # Taiwan Semiconductor
            '005930.KS',  # Samsung Electronics
            'BABA',   # Alibaba
            'TCEHY',  # Tencent
            '6758.T', # Sony
            'ASML',   # ASML Holding
            '9984.T'  # SoftBank
        ]
    
    def get_stock_info(self, symbols: List[str]) -> List[Dict]:
        """Get comprehensive stock information"""
        stock_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    
                    stock_data.append({
                        'symbol': symbol,
                        'company_name': info.get('longName', symbol),
                        'sector': info.get('sector', 'Technology'),
                        'current_price': float(latest_price),
                        'previous_close': float(prev_close),
                        'market_cap': info.get('marketCap', 0),
                        'pe_ratio': info.get('trailingPE', 0),
                        'volume': int(hist['Volume'].iloc[-1]),
                        'avg_volume': info.get('averageVolume', 0),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'yahoo_finance'
                    })
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
        
        return stock_data
