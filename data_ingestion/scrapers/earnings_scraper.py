import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import logging
import pandas as pd

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
                
                # Try to get calendar data - handle both DataFrame and dict responses
                try:
                    calendar = ticker.calendar
                    
                    # Handle case where calendar is a DataFrame
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
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
                    
                    # Handle case where calendar is a dictionary
                    elif isinstance(calendar, dict) and calendar:
                        # Extract earnings date from the dictionary structure
                        if 'Earnings Date' in calendar:
                            earnings_date = calendar.get('Earnings Date')
                            if earnings_date:
                                earnings_calendar.append({
                                    'symbol': symbol,
                                    'earnings_date': earnings_date,
                                    'eps_estimate': calendar.get('EPS Estimate', 0),
                                    'revenue_estimate': calendar.get('Revenue Estimate', 0),
                                    'content_type': 'earnings_calendar',
                                    'source': 'yahoo_finance_calendar',
                                    'timestamp': datetime.now().isoformat(),
                                    'tags': ['earnings_calendar', symbol.lower()]
                                })
                    
                    # Try alternative approach using info
                    else:
                        info = ticker.info
                        if info and 'earningsDate' in info:
                            earnings_date = info.get('earningsDate')
                            if earnings_date:
                                # Convert timestamp to date string if needed
                                if isinstance(earnings_date, (int, float)):
                                    earnings_date = datetime.fromtimestamp(earnings_date).strftime('%Y-%m-%d')
                                elif hasattr(earnings_date, 'strftime'):
                                    earnings_date = earnings_date.strftime('%Y-%m-%d')
                                
                                earnings_calendar.append({
                                    'symbol': symbol,
                                    'earnings_date': str(earnings_date),
                                    'eps_estimate': info.get('forwardEps', 0),
                                    'revenue_estimate': info.get('totalRevenue', 0),
                                    'content_type': 'earnings_calendar',
                                    'source': 'yahoo_finance_info',
                                    'timestamp': datetime.now().isoformat(),
                                    'tags': ['earnings_calendar', symbol.lower()]
                                })
                
                except Exception as calendar_error:
                    self.logger.warning(f"Calendar data unavailable for {symbol}: {calendar_error}")
                    
                    # Fallback: try to get basic earnings info from ticker.info
                    try:
                        info = ticker.info
                        if info and any(key in info for key in ['nextEarningsDate', 'earningsDate']):
                            earnings_date = info.get('nextEarningsDate') or info.get('earningsDate')
                            if earnings_date:
                                earnings_calendar.append({
                                    'symbol': symbol,
                                    'earnings_date': str(earnings_date),
                                    'eps_estimate': info.get('forwardEps', 0),
                                    'revenue_estimate': info.get('totalRevenue', 0),
                                    'content_type': 'earnings_calendar',
                                    'source': 'yahoo_finance_fallback',
                                    'timestamp': datetime.now().isoformat(),
                                    'tags': ['earnings_calendar', symbol.lower(), 'fallback']
                                })
                    except Exception as fallback_error:
                        self.logger.debug(f"Fallback earnings data also unavailable for {symbol}: {fallback_error}")
                        
                        # Create placeholder entry to indicate we tried
                        earnings_calendar.append({
                            'symbol': symbol,
                            'earnings_date': 'TBD',
                            'eps_estimate': 0,
                            'revenue_estimate': 0,
                            'content_type': 'earnings_calendar_placeholder',
                            'source': 'yahoo_finance_placeholder',
                            'timestamp': datetime.now().isoformat(),
                            'tags': ['earnings_calendar', symbol.lower(), 'no_data'],
                            'note': 'Earnings calendar data not available from Yahoo Finance'
                        })
                        
            except Exception as e:
                self.logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
                # Add error entry to track failed attempts
                earnings_calendar.append({
                    'symbol': symbol,
                    'earnings_date': 'ERROR',
                    'eps_estimate': 0,
                    'revenue_estimate': 0,
                    'content_type': 'earnings_calendar_error',
                    'source': 'yahoo_finance_error',
                    'timestamp': datetime.now().isoformat(),
                    'tags': ['earnings_calendar', symbol.lower(), 'error'],
                    'error': str(e)
                })
        
        self.logger.info(f"Retrieved earnings calendar data for {len([e for e in earnings_calendar if e['earnings_date'] not in ['TBD', 'ERROR']])} out of {len(symbols)} symbols")
        return earnings_calendar
    
    def scrape_seeking_alpha_earnings(self, symbol: str) -> List[Dict]:
        """Scrape earnings-related articles from Seeking Alpha with enhanced anti-bot handling"""
        articles = []
        
        # Enhanced headers to appear more like a real browser
        enhanced_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        # Try multiple URL patterns
        urls_to_try = [
            f"https://seekingalpha.com/symbol/{symbol}/earnings",
            f"https://seekingalpha.com/symbol/{symbol}/analysis",
            f"https://seekingalpha.com/symbol/{symbol}"
        ]
        
        for base_url in urls_to_try:
            try:
                # Add random delay to avoid rate limiting
                import time
                import random
                time.sleep(random.uniform(1, 3))
                
                # Use session for better cookie handling
                session = requests.Session()
                session.headers.update(enhanced_headers)
                
                response = session.get(base_url, timeout=15, allow_redirects=True)
                
                if response.status_code == 403:
                    self.logger.warning(f"Access denied (403) for {base_url} - trying alternative approach")
                    continue
                elif response.status_code == 429:
                    self.logger.warning(f"Rate limited (429) for {base_url} - backing off")
                    time.sleep(5)
                    continue
                
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try multiple selectors for article containers
                article_containers = (
                    soup.find_all('article', {'data-test-id': 'post-list-item'}) or
                    soup.find_all('div', class_=re.compile(r'.*article.*', re.I)) or
                    soup.find_all('div', class_=re.compile(r'.*post.*', re.I)) or
                    soup.find_all('a', href=re.compile(rf'/article/.*{symbol.lower()}.*', re.I))
                )
                
                for container in article_containers[:3]:  # Reduced to 3 to be less aggressive
                    try:
                        # Try multiple selectors for title
                        title_elem = (
                            container.find('a', {'data-test-id': 'post-list-item-title'}) or
                            container.find('a', class_=re.compile(r'.*title.*', re.I)) or
                            container.find('h3') or
                            container.find('h2') or
                            container if container.name == 'a' else None
                        )
                        
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            link = title_elem.get('href', '')
                            if link and not link.startswith('http'):
                                link = "https://seekingalpha.com" + link
                            
                            # Extract summary if available
                            summary_elem = (
                                container.find('span', {'data-test-id': 'post-list-content'}) or
                                container.find('div', class_=re.compile(r'.*summary.*', re.I)) or
                                container.find('p')
                            )
                            summary = summary_elem.get_text(strip=True) if summary_elem else ""
                            
                            # Extract author and date
                            author_elem = (
                                container.find('span', {'data-test-id': 'post-list-author'}) or
                                container.find('div', class_=re.compile(r'.*author.*', re.I))
                            )
                            author = author_elem.get_text(strip=True) if author_elem else "Seeking Alpha"
                            
                            # Filter for earnings-related content
                            earnings_keywords = ['earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', 'results', 'revenue']
                            title_lower = title.lower()
                            
                            # Only add if we have meaningful content and it's earnings-related
                            if (title and len(title) > 10 and 
                                any(keyword in title_lower for keyword in earnings_keywords)):
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
                                
                    except Exception as container_error:
                        self.logger.debug(f"Error processing container for {symbol}: {container_error}")
                        continue
                
                # If we found articles, break from trying other URLs
                if articles:
                    break
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    self.logger.warning(f"Access denied for {base_url} - website may have anti-bot protection")
                elif e.response.status_code == 429:
                    self.logger.warning(f"Rate limited for {base_url} - backing off")
                    time.sleep(5)
                else:
                    self.logger.error(f"HTTP error scraping {base_url}: {e}")
            except requests.RequestException as e:
                self.logger.error(f"Network error scraping {base_url}: {e}")
            except Exception as e:
                self.logger.error(f"Error scraping {base_url}: {e}")
        
        # If Seeking Alpha fails, try alternative sources
        if not articles:
            articles.extend(self._get_alternative_earnings_analysis(symbol))
        
        self.logger.info(f"Scraped {len(articles)} earnings articles for {symbol}")
        return articles
    
    def _get_alternative_earnings_analysis(self, symbol: str) -> List[Dict]:
        """Get earnings analysis from alternative sources when Seeking Alpha is blocked"""
        articles = []
        
        try:
            # Try Yahoo Finance earnings insights
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info:
                # Create synthetic analysis based on available data
                pe_ratio = info.get('forwardPE', info.get('trailingPE'))
                price_to_sales = info.get('priceToSalesTrailing12Months')
                profit_margins = info.get('profitMargins')
                
                if any([pe_ratio, price_to_sales, profit_margins]):
                    analysis_content = f"Fundamental Analysis for {symbol}:\n"
                    
                    if pe_ratio:
                        analysis_content += f"Forward P/E Ratio: {pe_ratio:.2f}\n"
                        if pe_ratio < 15:
                            analysis_content += "P/E suggests potentially undervalued stock. "
                        elif pe_ratio > 25:
                            analysis_content += "High P/E may indicate growth expectations or overvaluation. "
                    
                    if price_to_sales:
                        analysis_content += f"Price-to-Sales: {price_to_sales:.2f}\n"
                    
                    if profit_margins:
                        analysis_content += f"Profit Margins: {profit_margins:.1%}\n"
                        if profit_margins > 0.15:
                            analysis_content += "Strong profit margins indicate efficient operations. "
                    
                    articles.append({
                        'symbol': symbol,
                        'title': f"{symbol} Fundamental Analysis - Investment Metrics",
                        'summary': analysis_content,
                        'link': f"https://finance.yahoo.com/quote/{symbol}",
                        'author': "Yahoo Finance Data",
                        'content_type': 'earnings_analysis',
                        'source': 'yahoo_finance_analysis',
                        'timestamp': datetime.now().isoformat(),
                        'tags': ['earnings', 'fundamental_analysis', symbol.lower()]
                    })
        
        except Exception as e:
            self.logger.debug(f"Error getting alternative earnings analysis for {symbol}: {e}")
        
        return articles
    
    def extract_earnings_sentiment(self, text: str) -> Dict:
        """Extract sentiment indicators from earnings-related text"""
        if not text or not isinstance(text, str):
            return {
                'sentiment_score': 0.0,
                'positive_indicators': 0,
                'negative_indicators': 0,
                'sentiment_label': 'neutral'
            }
        
        positive_keywords = [
            'beat', 'exceeded', 'strong', 'growth', 'positive', 'outperformed',
            'guidance raised', 'better than expected', 'solid results', 'impressive',
            'robust', 'accelerating', 'expanding', 'momentum', 'bullish',
            'upside', 'optimistic', 'record', 'breakthrough', 'success'
        ]
        
        negative_keywords = [
            'missed', 'disappointed', 'weak', 'decline', 'negative', 'underperformed',
            'guidance lowered', 'worse than expected', 'poor results', 'concerning',
            'challenges', 'headwinds', 'pressure', 'bearish', 'downside',
            'pessimistic', 'struggling', 'difficulties', 'setback', 'warning'
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
        
        # Determine sentiment label with more nuanced thresholds
        if sentiment_score > 0.2:
            sentiment_label = 'positive'
        elif sentiment_score < -0.2:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': round(sentiment_score, 3),
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_label': sentiment_label
        }
    
    def get_recent_earnings_reports(self, symbol: str, quarters: int = 4) -> List[Dict]:
        """Get recent earnings reports for a symbol"""
        reports = []
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get quarterly earnings
            quarterly_earnings = ticker.quarterly_earnings
            
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                # Get the most recent quarters
                recent_quarters = quarterly_earnings.head(quarters)
                
                for date, row in recent_quarters.iterrows():
                    reports.append({
                        'symbol': symbol,
                        'quarter_end': date.strftime('%Y-%m-%d'),
                        'revenue': row.get('Revenue', 0),
                        'earnings': row.get('Earnings', 0),
                        'content_type': 'earnings_report',
                        'source': 'yahoo_finance_earnings',
                        'timestamp': datetime.now().isoformat(),
                        'tags': ['earnings_report', symbol.lower()]
                    })
                    
        except Exception as e:
            self.logger.error(f"Error fetching earnings reports for {symbol}: {e}")
        
        return reports