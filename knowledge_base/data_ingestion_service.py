from typing import List, Dict
import asyncio
from datetime import datetime, date
import logging
from data_ingestion.api_collectors.alpha_vantage_collector import AlphaVantageCollector
from data_ingestion.api_collectors.yahoo_finance_collector import YahooFinanceCollector
from data_ingestion.api_collectors.fred_collector import FREDCollector
from data_ingestion.scrapers.news_scraper import NewsSourceScraper
from data_ingestion.scrapers.sec_filing_scraper import SECFilingScraper
from data_ingestion.scrapers.earnings_scraper import EarningsTranscriptScraper

class DataIngestionService:
    """Handles data collection from various sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alpha_vantage = AlphaVantageCollector(config['api_keys']['alphavantage'])
        self.yahoo_finance = YahooFinanceCollector()
        self.fred_collector = FREDCollector(config['api_keys']['fred'])
        self.news_scraper = NewsSourceScraper()
        self.sec_scraper = SECFilingScraper()
        self.earnings_scraper = EarningsTranscriptScraper()
        self.asia_tech_symbols = config['data_collection']['asia_tech_symbols']
    
    async def collect_economic_data(self) -> List[Dict]:
        """Collect economic indicators from FRED."""
        documents = []
        try:
            indicators = self.fred_collector.get_economic_indicators(days_back=30)
            for indicator in indicators:
                if indicator.get('value') is not None and indicator.get('value') != '':
                    doc = {
                        'content_type': 'economic_indicator',
                        'title': f"{indicator['indicator_name']} Economic Indicator - {indicator['date']}",
                        'content': f"""
Economic Indicator: {indicator['indicator_name']}
Series ID: {indicator['series_id']}
Date: {indicator['date']}
Value: {indicator['value']}
Relevance Score: {indicator.get('relevance_score', 0.5):.2f}

This economic indicator is relevant for Asia tech market analysis.
                        """.strip(),
                        'source': 'fred_economic_data',
                        'exchange': 'FRED',
                        'timestamp': indicator.get('timestamp', datetime.now().isoformat()),
                        'date': indicator['date'],
                        'value': indicator['value'],
                        'relevance_score': indicator.get('relevance_score', 0.5),
                        'tags': indicator.get('tags', []) + ['fred', 'economic_data'],
                        'metadata': {
                            'indicator_name': indicator['indicator_name'],
                            'series_id': indicator['series_id'],
                            'date': indicator['date'],
                            'value': indicator['value'],
                            'source': 'fred',
                            'data_type': 'economic_indicator'
                        }
                    }
                    documents.append(doc)
        except Exception as e:
            self.logger.error(f"Error collecting economic data: {e}")
            raise
        return documents
    
    async def collect_market_data(self) -> List[Dict]:
        """Collect current market data."""
        documents = []
        try:
            stock_data = self.yahoo_finance.get_stock_info(self.asia_tech_symbols)
            for stock in stock_data:
                symbol = stock['symbol']
                exchange = self._get_primary_exchange(symbol)
                doc = {
                    'content_type': 'market_data',
                    'symbol': symbol,
                    'company_name': stock['company_name'],
                    'title': f"{stock['company_name']} ({symbol}) Market Data",
                    'content': f"""
Current Market Data: {stock['company_name']} ({symbol})
Current Price: ${stock['current_price']:.2f}
Previous Close: ${stock['previous_close']:.2f}
Price Change: ${stock['current_price'] - stock['previous_close']:.2f} ({((stock['current_price'] - stock['previous_close']) / stock['previous_close'] * 100):.2f}%)
Market Cap: ${stock['market_cap']:,}
P/E Ratio: {stock['pe_ratio']}
Volume: {stock['volume']:,}
Sector: {stock['sector']}
Exchange: {exchange}
                    """.strip(),
                    'source': 'yahoo_finance_market',
                    'exchange': exchange,
                    'timestamp': stock['timestamp'],
                    'current_price': stock['current_price'],
                    'previous_close': stock['previous_close'],
                    'tags': ['market_data', 'asia_tech', symbol.lower(), stock['sector'].lower().replace(' ', '_')],
                    'metadata': stock
                }
                documents.append(doc)
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            raise
        return documents
    
    async def collect_news_data(self) -> List[Dict]:
        """Collect recent financial news."""
        documents = []
        try:
            news_articles = self.news_scraper.scrape_rss_feeds(hours_back=12)
            for article in news_articles:
                news_text = f"{article['title']} {article.get('summary', '')}"
                sentiment_analysis = self.earnings_scraper.extract_earnings_sentiment(news_text)
                doc = {
                    'content_type': 'news_article',
                    'title': article['title'],
                    'content': f"""
Financial News Article
Title: {article['title']}
Summary: {article.get('summary', 'No summary available')}
Source: {article['source']}
Published: {article['published_date']}
URL: {article['link']}

Sentiment Analysis:
- Sentiment Score: {sentiment_analysis['sentiment_score']:.2f}
- Sentiment Label: {sentiment_analysis['sentiment_label']}
                    """.strip(),
                    'summary': article.get('summary', ''),
                    'source': f"news_{article['source']}",
                    'exchange': 'news_source',
                    'timestamp': article['timestamp'],
                    'link': article['link'],
                    'published_date': article['published_date'],
                    'sentiment_score': sentiment_analysis['sentiment_score'],
                    'sentiment_label': sentiment_analysis['sentiment_label'],
                    'tags': article.get('tags', []) + ['financial_news', 'market_news', sentiment_analysis['sentiment_label']],
                    'metadata': {
                        **article,
                        'sentiment_analysis': sentiment_analysis
                    }
                }
                if any(tag in ['asia', 'tech', 'tsmc', 'samsung', 'earnings', 'semiconductor', 'sony', 'softbank', 'alibaba', 'tencent'] 
                       for tag in doc['tags']):
                    documents.append(doc)
        except Exception as e:
            self.logger.error(f"Error collecting news data: {e}")
            raise
        return documents
    
    async def collect_earnings_data(self) -> List[Dict]:
        """Collect enhanced earnings data."""
        documents = []
        try:
            calendar_data = self.earnings_scraper.get_earnings_calendar(self.asia_tech_symbols)
            for entry in calendar_data:
                doc = {
                    'content_type': 'earnings_calendar',
                    'symbol': entry['symbol'],
                    'title': f"{entry['symbol']} Earnings Calendar - {entry['earnings_date']}",
                    'content': f"""
Upcoming Earnings: {entry['symbol']}
Earnings Date: {entry['earnings_date']}
EPS Estimate: ${entry['eps_estimate']}
Revenue Estimate: ${entry['revenue_estimate']}
                    """.strip(),
                    'source': 'earnings_calendar',
                    'exchange': self._get_primary_exchange(entry['symbol']),
                    'timestamp': entry['timestamp'],
                    'earnings_date': entry['earnings_date'],
                    'tags': entry['tags'] + ['earnings_calendar', 'forward_looking'],
                    'metadata': entry
                }
                if entry['eps_estimate'] != 0 or entry['revenue_estimate'] != 0:
                    documents.append(doc)
            
            us_symbols = [s for s in self.asia_tech_symbols if not ('.' in s and s.split('.')[1] in ['KS', 'T', 'HK'])]
            for symbol in us_symbols:
                try:
                    earnings_data = self.alpha_vantage.get_earnings_data(symbol)
                    for earning in earnings_data:
                        if self.validate_earnings_date(earning['fiscal_date_ending']):
                            combined_text = f"{earning.get('reported_eps', '')} {earning.get('estimated_eps', '')} {earning.get('surprise', '')}"
                            sentiment_analysis = self.earnings_scraper.extract_earnings_sentiment(combined_text)
                            doc = {
                                'content_type': 'earnings_data',
                                'symbol': earning['symbol'],
                                'title': f"{earning['symbol']} Earnings - {earning['fiscal_date_ending']}",
                                'content': f"""
Historical Earnings Report: {earning['symbol']}
Fiscal Date: {earning['fiscal_date_ending']}
Reported EPS: ${earning['reported_eps']}
Estimated EPS: ${earning['estimated_eps']}
Surprise: ${earning['surprise']} ({earning['surprise_percentage']}%)
Exchange: SEC/US Markets

Sentiment Analysis:
- Sentiment Score: {sentiment_analysis['sentiment_score']:.2f}
- Sentiment Label: {sentiment_analysis['sentiment_label']}
                                """.strip(),
                                'source': 'alphavantage_earnings',
                                'exchange': 'SEC',
                                'timestamp': earning['timestamp'],
                                'sentiment_score': sentiment_analysis['sentiment_score'],
                                'sentiment_label': sentiment_analysis['sentiment_label'],
                                'tags': ['earnings', 'asia_tech', earning['symbol'].lower(), sentiment_analysis['sentiment_label']],
                                'metadata': {
                                    **earning,
                                    'sentiment_analysis': sentiment_analysis,
                                    'source': 'alphavantage_earnings'
                                }
                            }
                            documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error collecting Alpha Vantage earnings for {symbol}: {e}")
            
            major_symbols = ['TSM', 'BABA', 'ASML']
            for symbol in major_symbols:
                try:
                    articles = self.earnings_scraper.scrape_seeking_alpha_earnings(symbol)
                    for article in articles:
                        article_text = f"{article['title']} {article.get('summary', '')}"
                        sentiment_analysis = self.earnings_scraper.extract_earnings_sentiment(article_text)
                        doc = {
                            'content_type': 'earnings_analysis',
                            'symbol': article['symbol'],
                            'title': article['title'],
                            'content': f"""
Earnings Analysis Article: {article['symbol']}
Title: {article['title']}
Author: {article['author']}
Summary: {article.get('summary', 'No summary available')}
Source URL: {article['link']}

Sentiment Analysis:
- Sentiment Score: {sentiment_analysis['sentiment_score']:.2f}
- Sentiment Label: {sentiment_analysis['sentiment_label']}
                            """.strip(),
                            'source': 'seeking_alpha_earnings',
                            'exchange': self._get_primary_exchange(article['symbol']),
                            'timestamp': article['timestamp'],
                            'link': article['link'],
                            'author': article['author'],
                            'sentiment_score': sentiment_analysis['sentiment_score'],
                            'sentiment_label': sentiment_analysis['sentiment_label'],
                            'tags': article['tags'] + [sentiment_analysis['sentiment_label'], 'third_party_analysis'],
                            'metadata': {
                                **article,
                                'sentiment_analysis': sentiment_analysis
                            }
                        }
                        documents.append(doc)
                except Exception as e:
                    self.logger.error(f"Error scraping Seeking Alpha earnings for {symbol}: {e}")
        except Exception as e:
            self.logger.error(f"Error in earnings data collection: {e}")
            raise
        return documents
    
    async def collect_regulatory_filings(self) -> List[Dict]:
        """Collect regulatory filings from multiple exchanges."""
        documents = []
        try:
            all_filings = self.sec_scraper.batch_collect_filings(self.asia_tech_symbols)
            for filing in all_filings:
                exchange = filing.get('exchange', 'Unknown')
                symbol = filing.get('symbol', 'Unknown')
                doc = {
                    'content_type': 'regulatory_filing',
                    'symbol': symbol,
                    'title': f"{filing.get('company_name', symbol)} - {filing.get('form_type', 'Filing')}",
                    'content': f"""
Regulatory Filing: {filing.get('company_name', symbol)}
Symbol: {symbol}
Form Type: {filing.get('form_type', 'N/A')}
Filing Date: {filing.get('filing_date', 'N/A')}
Description: {filing.get('description', 'N/A')}
Exchange: {exchange}
Document URL: {filing.get('document_url', 'N/A')}
                    """.strip(),
                    'source': f"regulatory_{exchange.lower()}",
                    'exchange': exchange,
                    'timestamp': filing.get('timestamp', datetime.now().isoformat()),
                    'document_url': filing.get('document_url'),
                    'filing_date': filing.get('filing_date'),
                    'form_type': filing.get('form_type'),
                    'tags': ['regulatory_filing', exchange.lower(), symbol.lower()],
                    'metadata': filing
                }
                if filing.get('document_url'):
                    try:
                        content = self.sec_scraper.get_filing_content(filing['document_url'])
                        if content:
                            doc['content'] += f"\n\nFiling Content Preview:\n{content[:1000]}..."
                    except Exception as e:
                        self.logger.warning(f"Could not retrieve content for {filing['document_url']}: {e}")
                documents.append(doc)
        except Exception as e:
            self.logger.error(f"Error collecting regulatory filings: {e}")
            raise
        return documents
    
    def _get_primary_exchange(self, symbol: str) -> str:
        """Determine the primary exchange for a symbol."""
        if '.KS' in symbol:
            return 'KSE'
        elif '.T' in symbol:
            return 'TSE'
        elif '.HK' in symbol:
            return 'SEHK'
        else:
            return 'NYSE/NASDAQ'
    
    def validate_earnings_date(self, fiscal_date_str: str) -> bool:
        """Validate that earnings date is not in the future."""
        try:
            fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d').date()
            if fiscal_date > date.today():
                self.logger.warning(f"Future earnings date detected: {fiscal_date_str}")
                return False
            return True
        except ValueError:
            self.logger.error(f"Invalid date format: {fiscal_date_str}")
            return False