import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import logging
class EarningsTranscriptScraper:
    """Scrapes earnings call transcripts and earnings-related content"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.logger = logging.getLogger(__name__)
    
    def get_earnings_calendar(self, symbols: List[str]) -> List[Dict]:
        """Get earnings calendar for specified symbols using yfinance"""
        earnings_calendar = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
                
                if calendar is not None and not calendar.empty:
                    for date, row in calendar.iterrows():
                        earnings_calendar.append({
                            'symbol': symbol,
                            'earnings_date': date.strftime('%Y-%m-%d'),
                            'eps_estimate': row.get('Earnings Estimate', 0),
                            'revenue_estimate': row.get('Revenue Estimate', 0),
                            'content_type': 'earnings_calendar',
                            'source': 'yahoo_finance_calendar',
                            'timestamp': datetime.now().isoformat(),
                            'tags': ['earnings_calendar', symbol.lower()]
                        })
                        
            except Exception as e:
                self.logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
        
        return earnings_calendar
    
    def scrape_seeking_alpha_earnings(self, symbol: str) -> List[Dict]:
        """Scrape earnings-related articles from Seeking Alpha"""
        articles = []
        base_url = f"https://seekingalpha.com/symbol/{symbol}/earnings"
        
        try:
            response = requests.get(base_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find earnings articles
            article_containers = soup.find_all('article', {'data-test-id': 'post-list-item'})
            
            for container in article_containers[:5]:  # Limit to 5 recent articles
                title_elem = container.find('a', {'data-test-id': 'post-list-item-title'})
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = "https://seekingalpha.com" + title_elem.get('href', '')
                    
                    # Extract summary if available
                    summary_elem = container.find('span', {'data-test-id': 'post-list-content'})
                    summary = summary_elem.get_text(strip=True) if summary_elem else ""
                    
                    # Extract author and date
                    author_elem = container.find('span', {'data-test-id': 'post-list-author'})
                    author = author_elem.get_text(strip=True) if author_elem else "Unknown"
                    
                    articles.append({
                        'symbol': symbol,
                        'title': title,
                        'summary': summary,
                        'link': link,
                        'author': author,
                        'content_type': 'earnings_analysis',
                        'source': 'seeking_alpha',
                        'timestamp': datetime.now().isoformat(),
                        'tags': ['earnings', 'analysis', symbol.lower()]
                    })
                    
        except Exception as e:
            self.logger.error(f"Error scraping Seeking Alpha for {symbol}: {e}")
        
        return articles
    
    def extract_earnings_sentiment(self, text: str) -> Dict:
        """Extract sentiment indicators from earnings-related text"""
        positive_keywords = [
            'beat', 'exceeded', 'strong', 'growth', 'positive', 'outperformed',
            'guidance raised', 'better than expected', 'solid results'
        ]
        
        negative_keywords = [
            'missed', 'disappointed', 'weak', 'decline', 'negative', 'underperformed',
            'guidance lowered', 'worse than expected', 'poor results'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0.0
        
        return {
            'sentiment_score': sentiment_score,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
        }
