from typing import List, Dict
import asyncio
import logging
from datetime import datetime, date
import schedule
import time
import sys
import os
import json
from typing import Any, Dict, List

# Correct imports based on the directory structure
from data_ingestion.api_collectors.alpha_vantage_collector import AlphaVantageCollector
from data_ingestion.api_collectors.yahoo_finance_collector import YahooFinanceCollector
from data_ingestion.api_collectors.fred_collector import FREDCollector
from data_ingestion.scrapers.news_scraper import NewsSourceScraper
from data_ingestion.scrapers.sec_filing_scraper import SECFilingScraper
from data_ingestion.scrapers.earnings_scraper import EarningsTranscriptScraper  # Added earnings scraper import
from knowledge_base.vector_store.faiss_manager import FAISSVectorStore

class DataIngestionPipeline:
    """Orchestrates all data collection and knowledge base updates with enhanced foreign exchange support and earnings integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize collectors
        self.alpha_vantage = AlphaVantageCollector(config['alphavantage_api_key'])
        self.yahoo_finance = YahooFinanceCollector()
        self.fred_collector = FREDCollector(config['fred_api_key'])
        self.news_scraper = NewsSourceScraper()
        self.sec_scraper = SECFilingScraper()
        self.earnings_scraper = EarningsTranscriptScraper()  # Added earnings scraper
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore()
        
        # Track symbols of interest with their exchange information
        self.asia_tech_symbols = [
            'TSM',        # TSMC (NYSE ADR)
            '005930.KS',  # Samsung Electronics (KSE)
            'BABA',       # Alibaba (NYSE ADR)
            'TCEHY',      # Tencent (OTC ADR)
            '6758.T',     # Sony Group (TSE)
            'ASML',       # ASML (NASDAQ ADR)
            '9984.T',     # SoftBank Group (TSE)
            '0700.HK'     # Tencent Holdings (HKEX) - if available
        ]
        
        # Exchange-specific configurations
        self.exchange_configs = {
            'SEC': {'rate_limit': 1.0, 'retry_attempts': 3},
            'KSE': {'rate_limit': 2.0, 'retry_attempts': 2},
            'TSE': {'rate_limit': 2.0, 'retry_attempts': 2},
            'SEHK': {'rate_limit': 1.5, 'retry_attempts': 2}
        }
        
        # Track pipeline statistics
        self.pipeline_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'documents_processed': 0,
            'last_run_timestamp': None,
            'data_source_stats': {}
        }
    
    async def run_full_pipeline(self) -> Dict:
        """Run complete data collection pipeline with enhanced foreign exchange support and earnings integration"""
        self.pipeline_stats['total_runs'] += 1
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'documents_added': 0,
            'errors': [],
            'exchange_breakdown': {},
            'data_sources': {},  # Track documents by data source
            'fred_status': 'pending',  # Track FRED data specifically
            'earnings_status': 'pending'  # Track earnings data specifically
        }
        
        try:
            all_documents = []
            
            # 1. Collect economic indicators (FRED data) - HIGH PRIORITY
            self.logger.info("Collecting economic indicators from FRED...")
            try:
                economic_docs = await self._collect_economic_data()
                all_documents.extend(economic_docs)
                results['data_sources']['economic_indicators'] = len(economic_docs)
                results['fred_status'] = 'success' if economic_docs else 'no_data'
                self.logger.info(f"FRED data collection: {len(economic_docs)} documents collected")
            except Exception as e:
                results['errors'].append(f"FRED collection error: {str(e)}")
                results['fred_status'] = 'failed'
                self.logger.error(f"FRED data collection failed: {e}")
            
            # 2. Collect market data
            self.logger.info("Collecting market data...")
            try:
                market_docs = await self._collect_market_data()
                all_documents.extend(market_docs)
                results['data_sources']['market_data'] = len(market_docs)
            except Exception as e:
                results['errors'].append(f"Market data error: {str(e)}")
                self.logger.error(f"Market data collection failed: {e}")
            
            # 3. Collect news articles
            self.logger.info("Collecting news articles...")
            try:
                news_docs = await self._collect_news_data()
                all_documents.extend(news_docs)
                results['data_sources']['news_articles'] = len(news_docs)
            except Exception as e:
                results['errors'].append(f"News collection error: {str(e)}")
                self.logger.error(f"News collection failed: {e}")
            
            # 4. Collect earnings data (ENHANCED with scraper integration)
            self.logger.info("Collecting earnings data with enhanced scraping...")
            try:
                earnings_docs = await self._collect_enhanced_earnings_data()
                all_documents.extend(earnings_docs)
                results['data_sources']['earnings_data'] = len(earnings_docs)
                results['earnings_status'] = 'success' if earnings_docs else 'no_data'
                self.logger.info(f"Enhanced earnings collection: {len(earnings_docs)} documents collected")
            except Exception as e:
                results['errors'].append(f"Earnings collection error: {str(e)}")
                results['earnings_status'] = 'failed'
                self.logger.error(f"Enhanced earnings collection failed: {e}")
            
            # 5. Collect regulatory filings (SEC and foreign exchanges)
            self.logger.info("Collecting regulatory filings...")
            try:
                filing_docs = await self._collect_regulatory_filings()
                all_documents.extend(filing_docs)
                results['data_sources']['regulatory_filings'] = len(filing_docs)
            except Exception as e:
                results['errors'].append(f"Regulatory filings error: {str(e)}")
                self.logger.error(f"Regulatory filings collection failed: {e}")
            
            # 6. Add to vector store with enhanced logging
            if all_documents:
                self.logger.info(f"Adding {len(all_documents)} documents to FAISS vector store...")
                try:
                    # Verify vector store is working
                    pre_add_count = self.vector_store.get_document_count() if hasattr(self.vector_store, 'get_document_count') else 0
                    
                    # FIX: Serialize documents before adding to vector store
                    serialized_documents = self._serialize_documents_for_storage(all_documents)
                    self.logger.info(f"Serialized {len(serialized_documents)} documents for vector store")
                    
                    self.vector_store.add_documents(serialized_documents)
                    
                    post_add_count = self.vector_store.get_document_count() if hasattr(self.vector_store, 'get_document_count') else 0
                    
                    results['documents_added'] = len(all_documents)
                    results['vector_store_before'] = pre_add_count
                    results['vector_store_after'] = post_add_count
                    
                    self.logger.info(f"Successfully added {len(all_documents)} documents to vector store")
                    self.logger.info(f"Vector store document count: {pre_add_count} -> {post_add_count}")
                    
                except Exception as e:
                    results['errors'].append(f"Vector store error: {str(e)}")
                    self.logger.error(f"Failed to add documents to vector store: {e}")
            else:
                self.logger.warning("No documents collected to add to vector store")
            
            results['status'] = 'completed' if not results['errors'] else 'completed_with_errors'
            self.pipeline_stats['successful_runs'] += 1
            self.pipeline_stats['documents_processed'] += len(all_documents)
            
            self.logger.info(f"Pipeline completed. Status: {results['status']}, Documents: {len(all_documents)}")
            
            # Analyze document breakdown by type and exchange
            doc_types = {}
            exchange_breakdown = {}
            
            for doc in all_documents:
                content_type = doc.get('content_type', 'unknown')
                doc_types[content_type] = doc_types.get(content_type, 0) + 1
                
                exchange = doc.get('exchange', doc.get('source', 'unknown'))
                exchange_breakdown[exchange] = exchange_breakdown.get(exchange, 0) + 1
            
            results['document_types'] = doc_types
            results['exchange_breakdown'] = exchange_breakdown
            
            # Update data source statistics
            for source, count in results['data_sources'].items():
                if source not in self.pipeline_stats['data_source_stats']:
                    self.pipeline_stats['data_source_stats'][source] = {'total': 0, 'runs': 0}
                self.pipeline_stats['data_source_stats'][source]['total'] += count
                self.pipeline_stats['data_source_stats'][source]['runs'] += 1
            
            self.logger.info(f"Document types collected: {doc_types}")
            self.logger.info(f"Exchange breakdown: {exchange_breakdown}")
            self.logger.info(f"Data sources breakdown: {results['data_sources']}")
            self.logger.info(f"FRED status: {results['fred_status']}, Earnings status: {results['earnings_status']}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Pipeline critical error: {str(e)}")
            self.pipeline_stats['failed_runs'] += 1
            self.logger.error(f"Pipeline critical failure: {e}")
        
        self.pipeline_stats['last_run_timestamp'] = results['timestamp']
        return results
    
    async def _collect_economic_data(self) -> List[Dict]:
        """Collect economic indicators from FRED with enhanced validation"""
        documents = []
        
        try:
            self.logger.info("Fetching economic indicators from FRED API...")
            
            # Get economic indicators for the last 30 days
            indicators = self.fred_collector.get_economic_indicators(days_back=30)
            
            self.logger.info(f"FRED API returned {len(indicators)} indicators")
            
            valid_indicators = 0
            for indicator in indicators:
                if indicator.get('value') is not None and indicator.get('value') != '':
                    valid_indicators += 1
                    doc = {
                        'content_type': 'economic_indicator',
                        'title': f"{indicator['indicator_name']} Economic Indicator - {indicator['date']}",
                        'content': f"""
Economic Indicator: {indicator['indicator_name']}
Series ID: {indicator['series_id']}
Date: {indicator['date']}
Value: {indicator['value']}
Relevance Score: {indicator.get('relevance_score', 0.5):.2f}

This economic indicator is relevant for Asia tech market analysis with a relevance score of {indicator.get('relevance_score', 0.5):.2f}.
Data sourced from Federal Reserve Economic Data (FRED).

Analysis Context: Economic indicators provide macro-economic context for tech sector performance,
particularly relevant for understanding market conditions affecting Asian technology companies
and their ADR counterparts trading in US markets.
                        """.strip(),
                        'source': 'fred_economic_data',
                        'exchange': 'FRED',
                        'timestamp': indicator.get('timestamp', datetime.now().isoformat()),
                        'date': indicator['date'],
                        'value': indicator['value'],
                        'relevance_score': indicator.get('relevance_score', 0.5),
                        'tags': indicator.get('tags', []) + ['fred', 'economic_data', 'macro_analysis'],
                        'metadata': {
                            'indicator_name': indicator['indicator_name'],
                            'series_id': indicator['series_id'],
                            'date': indicator['date'],
                            'value': indicator['value'],
                            'relevance_score': indicator.get('relevance_score', 0.5),
                            'source': 'fred',
                            'data_type': 'economic_indicator'
                        }
                    }
                    documents.append(doc)
            
            self.logger.info(f"FRED data processing complete: {valid_indicators} valid indicators out of {len(indicators)} total")
            self.logger.info(f"Created {len(documents)} FRED documents for vector store")
            
        except Exception as e:
            self.logger.error(f"Error collecting economic data from FRED: {e}")
            # Re-raise to ensure calling function knows about the failure
            raise
        
        return documents
    
    async def _collect_enhanced_earnings_data(self) -> List[Dict]:
        """Enhanced earnings data collection using both API and scraper"""
        documents = []
        
        try:
            # 1. Get earnings calendar for all symbols
            self.logger.info("Collecting earnings calendar...")
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

This earnings calendar entry provides forward-looking earnings expectations
for {entry['symbol']}, helping anticipate market-moving events.
                    """.strip(),
                    'source': 'earnings_calendar',
                    'exchange': self._get_primary_exchange(entry['symbol']),
                    'timestamp': entry['timestamp'],
                    'earnings_date': entry['earnings_date'],
                    'tags': entry['tags'] + ['earnings_calendar', 'forward_looking'],
                    'metadata': entry
                }
                documents.append(doc)
            
            # 2. Get traditional earnings data from Alpha Vantage (for US symbols only)
            us_symbols = [s for s in self.asia_tech_symbols if not ('.' in s and s.split('.')[1] in ['KS', 'T', 'HK'])]
            
            for symbol in us_symbols:
                try:
                    earnings_data = self.alpha_vantage.get_earnings_data(symbol)
                    
                    for earning in earnings_data:
                        if self.validate_earnings_date(earning['fiscal_date_ending']):
                            # Extract sentiment from earnings data
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
- Positive Indicators: {sentiment_analysis['positive_indicators']}
- Negative Indicators: {sentiment_analysis['negative_indicators']}

This earnings report provides historical performance context and sentiment analysis
for investment decision-making and market trend analysis.
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
            
            # 3. Scrape earnings analysis from Seeking Alpha (for major symbols)
            major_symbols = ['TSM', 'BABA', 'ASML']  # Focus on most liquid ADRs
            
            for symbol in major_symbols:
                try:
                    articles = self.earnings_scraper.scrape_seeking_alpha_earnings(symbol)
                    
                    for article in articles:
                        # Extract sentiment from article content
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

This third-party earnings analysis provides market sentiment and expert opinion
on {article['symbol']} earnings performance and outlook.
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
            
            self.logger.info(f"Enhanced earnings collection completed: {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error in enhanced earnings data collection: {e}")
            raise
            
        return documents
    
    async def _collect_market_data(self) -> List[Dict]:
        """Collect current market data"""
        documents = []
        
        try:
            # Get Yahoo Finance data for Asia tech stocks
            stock_data = self.yahoo_finance.get_stock_info(self.asia_tech_symbols)
            
            for stock in stock_data:
                # Determine exchange for metadata
                symbol = stock['symbol']
                exchange = self._get_primary_exchange(symbol)
                
                # Create document for each stock
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

This real-time market data provides current valuation and trading metrics
for {stock['company_name']}, essential for market analysis and investment decisions.
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
    
    async def _collect_news_data(self) -> List[Dict]:
        """Collect recent financial news"""
        documents = []
        
        try:
            news_articles = self.news_scraper.scrape_rss_feeds(hours_back=12)
            
            # Transform news articles for vector store compatibility
            for article in news_articles:
                # Extract sentiment from news content
                news_text = f"{article['title']} {article.get('summary', '')}"
                sentiment_analysis = self.earnings_scraper.extract_earnings_sentiment(news_text)
                
                # Create proper document structure
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

This financial news article provides market context and sentiment indicators
relevant to Asia tech sector analysis and investment decision-making.
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
                
                # Filter for Asia tech related news
                if any(tag in ['asia', 'tech', 'tsmc', 'samsung', 'earnings', 'semiconductor', 'sony', 'softbank', 'alibaba', 'tencent'] 
                    for tag in doc['tags']):
                    documents.append(doc)
        
        except Exception as e:
            self.logger.error(f"Error collecting news data: {e}")
            raise
            
        return documents
    
    async def _collect_regulatory_filings(self) -> List[Dict]:
        """Collect regulatory filings from multiple exchanges"""
        documents = []
        
        try:
            # Use the enhanced SEC scraper with foreign exchange support
            all_filings = self.sec_scraper.batch_collect_filings(self.asia_tech_symbols)
            
            for filing in all_filings:
                # Transform filing data into document format
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

This regulatory filing provides official company disclosure information
required by {exchange} regulations, essential for compliance and investment analysis.
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
                
                # Attempt to get filing content if URL is available
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
    
    def get_pipeline_statistics(self) -> Dict:
        """Get comprehensive pipeline statistics"""
        success_rate = (self.pipeline_stats['successful_runs'] / max(self.pipeline_stats['total_runs'], 1)) * 100
        
        return {
            'total_runs': self.pipeline_stats['total_runs'],
            'successful_runs': self.pipeline_stats['successful_runs'],
            'failed_runs': self.pipeline_stats['failed_runs'],
            'success_rate_percent': round(success_rate, 2),
            'total_documents_processed': self.pipeline_stats['documents_processed'],
            'last_run_timestamp': self.pipeline_stats['last_run_timestamp'],
            'data_source_statistics': self.pipeline_stats['data_source_stats'],
            'average_documents_per_run': round(self.pipeline_stats['documents_processed'] / max(self.pipeline_stats['total_runs'], 1), 2)
        }
    
    def _get_primary_exchange(self, symbol: str) -> str:
        """Determine the primary exchange for a symbol"""
        if '.KS' in symbol:
            return 'KSE'
        elif '.T' in symbol:
            return 'TSE' 
        elif '.HK' in symbol:
            return 'SEHK'
        else:
            return 'NYSE/NASDAQ'
    
    def validate_earnings_date(self, fiscal_date_str: str) -> bool:
        """Validate that earnings date is not in the future"""
        try:
            fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d').date()
            if fiscal_date > date.today():
                self.logger.warning(f"Future earnings date detected: {fiscal_date_str} - possible demo data")
                return False
            return True
        except ValueError:
            self.logger.error(f"Invalid date format: {fiscal_date_str}")
            return False
    
    # ... (rest of the methods remain the same as original)
    def schedule_regular_updates(self):
        """Schedule regular pipeline runs with exchange-specific timing"""
        # Economic data updates (less frequent - daily)
        schedule.every().day.at("06:00").do(lambda: asyncio.run(self._collect_and_update_economic_data()))
        
        # Market data updates (every 4 hours during market hours)
        schedule.every(4).hours.do(lambda: asyncio.run(self.run_full_pipeline()))
        
        # Daily comprehensive update at 7 AM (before market open)
        schedule.every().day.at("07:00").do(lambda: asyncio.run(self.run_full_pipeline()))
        
        # Weekly comprehensive update on Sundays
        schedule.every().sunday.at("06:00").do(lambda: asyncio.run(self._weekly_comprehensive_update()))
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _collect_and_update_economic_data(self):
        """Dedicated method for updating economic indicators"""
        self.logger.info("Starting economic data update...")
        try:
            economic_docs = await self._collect_economic_data()
            if economic_docs:
                self.vector_store.add_documents(economic_docs)
                self.logger.info(f"Added {len(economic_docs)} economic indicators to knowledge base")
            else:
                self.logger.warning("No economic indicators collected in dedicated update")
        except Exception as e:
            self.logger.error(f"Error in economic data update: {e}")
    
    def _serialize_documents_for_storage(self, documents: List[Dict]) -> List[Dict]:
        """Ensure all documents are JSON serializable before adding to vector store"""
        
        def serialize_value(value):
            if isinstance(value, (datetime, date)):
                return value.isoformat()
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif hasattr(value, '__dict__'):
                return serialize_value(value.__dict__)
            elif not isinstance(value, (str, int, float, bool, type(None))):
                return str(value)
            return value
        
        fixed_documents = []
        for doc in documents:
            try:
                fixed_doc = serialize_value(doc.copy())
                # Test serialization
                json.dumps(fixed_doc)
                fixed_documents.append(fixed_doc)
            except Exception as e:
                self.logger.warning(f"Document serialization issue for '{doc.get('title', 'Unknown')}': {e}")
                # Create minimal fallback document
                minimal_doc = {
                    'content_type': str(doc.get('content_type', 'unknown')),
                    'title': str(doc.get('title', 'Unknown')),
                    'content': str(doc.get('content', '')),
                    'source': str(doc.get('source', 'unknown')),
                    'exchange': str(doc.get('exchange', 'unknown')),
                    'timestamp': datetime.now().isoformat(),
                    'tags': [str(tag) for tag in doc.get('tags', [])],
                    'metadata': {'serialization_fallback': True, 'original_error': str(e)}
                }
                fixed_documents.append(minimal_doc)
        
        return fixed_documents

    async def _weekly_comprehensive_update(self):
        """Weekly comprehensive update including historical filings and extended economic data"""
        self.logger.info("Starting weekly comprehensive update...")
        
        # Expand symbol list for comprehensive update
        extended_symbols = self.asia_tech_symbols + [
            '000660.KS',  # SK Hynix
            '035420.KS',  # NAVER
            '7203.T',     # Toyota
            '6861.T'      # Keyence
        ]
        
        # Run pipeline with extended symbol list
        original_symbols = self.asia_tech_symbols
        self.asia_tech_symbols = extended_symbols
        
        try:
            # Collect extended economic data (90 days)
            extended_economic_docs = self.fred_collector.get_economic_indicators(days_back=90)
            economic_documents = []
            
            for indicator in extended_economic_docs:
                if indicator['value'] is not None:
                    doc = {
                        'content_type': 'economic_indicator',
                        'title': f"{indicator['indicator_name']} Historical Data - {indicator['date']}",
                        'content': f"""
                        Economic Indicator: {indicator['indicator_name']}
                        Series ID: {indicator['series_id']}
                        Date: {indicator['date']}
                        Value: {indicator['value']}
                        Relevance Score: {indicator['relevance_score']:.2f}
                        
                        Historical economic data for comprehensive market analysis.
                        """,
                        'source': 'fred_historical_data',
                        'exchange': 'FRED',
                        'timestamp': indicator['timestamp'],
                        'tags': indicator['tags'] + ['fred', 'historical', 'comprehensive'],
                        'metadata': indicator
                    }
                    economic_documents.append(doc)
            
            # Add historical economic data
            if economic_documents:
                self.vector_store.add_documents(economic_documents)
                self.logger.info(f"Added {len(economic_documents)} historical economic indicators")
            
            # Run regular pipeline with extended symbols
            result = await self.run_full_pipeline()
            result['historical_economic_indicators'] = len(economic_documents)
            
            self.logger.info(f"Weekly comprehensive update completed: {result}")
            
        except Exception as e:
            self.logger.error(f"Error in weekly comprehensive update: {e}")
        finally:
            # Restore original symbol list
            self.asia_tech_symbols = original_symbols
    
    async def run_targeted_update(self, symbols: List[str] = None, data_types: List[str] = None):
        """Run targeted update for specific symbols or data types"""
        if symbols:
            original_symbols = self.asia_tech_symbols
            self.asia_tech_symbols = symbols
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'targeted_symbols': symbols,
            'targeted_data_types': data_types,
            'documents_added': 0
        }
        
        try:
            all_documents = []
            
            # Collect only specified data types or all if not specified
            if not data_types or 'economic' in data_types:
                economic_docs = await self._collect_economic_data()
                all_documents.extend(economic_docs)
            
            if not data_types or 'market' in data_types:
                market_docs = await self._collect_market_data()
                all_documents.extend(market_docs)
            
            if not data_types or 'news' in data_types:
                news_docs = await self._collect_news_data()
                all_documents.extend(news_docs)
            
            if not data_types or 'earnings' in data_types:
                earnings_docs = await self._collect_earnings_data()
                all_documents.extend(earnings_docs)
            
            if not data_types or 'filings' in data_types:
                filing_docs = await self._collect_regulatory_filings()
                all_documents.extend(filing_docs)
            
            if all_documents:
                self.vector_store.add_documents(all_documents)
                results['documents_added'] = len(all_documents)
            
            results['status'] = 'completed'
            self.logger.info(f"Targeted update completed: {results}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"Targeted update failed: {e}")
        finally:
            if symbols:
                self.asia_tech_symbols = original_symbols
        
        return results