from typing import List, Dict
import asyncio
import logging
from datetime import datetime,date
import schedule
import time
import sys
import os
import logging

# Correct imports based on the directory structure
from data_ingestion.api_collectors.alpha_vantage_collector import AlphaVantageCollector
from data_ingestion.api_collectors.yahoo_finance_collector import YahooFinanceCollector
from data_ingestion.scrapers.news_scraper import NewsSourceScraper
from data_ingestion.scrapers.sec_filing_scraper import SECFilingScraper
from knowledge_base.vector_store.faiss_manager import FAISSVectorStore
class DataIngestionPipeline:
    """Orchestrates all data collection and knowledge base updates"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize collectors
        self.alpha_vantage = AlphaVantageCollector(config['alphavantage_api_key'])
        self.yahoo_finance = YahooFinanceCollector()
        self.news_scraper = NewsSourceScraper()
        self.sec_scraper = SECFilingScraper()
        
        # Initialize vector store
        self.vector_store = FAISSVectorStore()
        
        # Track symbols of interest
        self.asia_tech_symbols = [
            'TSM', '005930.KS', 'BABA', 'TCEHY', '6758.T', 'ASML', '9984.T'
        ]
    
    async def run_full_pipeline(self) -> Dict:
        """Run complete data collection pipeline"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'documents_added': 0,
            'errors': []
        }
        
        try:
            all_documents = []
            
            # 1. Collect market data
            self.logger.info("Collecting market data...")
            market_docs = await self._collect_market_data()
            all_documents.extend(market_docs)
            
            # 2. Collect news articles
            self.logger.info("Collecting news articles...")
            news_docs = await self._collect_news_data()
            all_documents.extend(news_docs)
            
            # 3. Collect earnings data
            self.logger.info("Collecting earnings data...")
            earnings_docs = await self._collect_earnings_data()
            all_documents.extend(earnings_docs)
            
            # 4. Add to vector store
            if all_documents:
                self.logger.info(f"Adding {len(all_documents)} documents to vector store...")
                self.vector_store.add_documents(all_documents)
                results['documents_added'] = len(all_documents)
            
            results['status'] = 'completed'
            self.logger.info(f"Pipeline completed successfully. Added {len(all_documents)} documents.")
            
            doc_types = {}
            for doc in all_documents:
                content_type = doc.get('content_type', 'unknown')
                doc_types[content_type] = doc_types.get(content_type, 0) + 1

            self.logger.info(f"Document types collected: {doc_types}")
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(str(e))
            self.logger.error(f"Pipeline failed: {e}")
        
        return results
    
    async def _collect_market_data(self) -> List[Dict]:
        """Collect current market data"""
        documents = []
        
        # Get Yahoo Finance data for Asia tech stocks
        stock_data = self.yahoo_finance.get_stock_info(self.asia_tech_symbols)
        
        for stock in stock_data:
            # Create document for each stock
            doc = {
                'content_type': 'market_data',
                'symbol': stock['symbol'],
                'company_name': stock['company_name'],
                'title': f"{stock['company_name']} ({stock['symbol']}) Market Data",
                'content': f"""
                Current Price: ${stock['current_price']:.2f}
                Previous Close: ${stock['previous_close']:.2f}
                Market Cap: ${stock['market_cap']:,}
                P/E Ratio: {stock['pe_ratio']}
                Volume: {stock['volume']:,}
                Sector: {stock['sector']}
                """,
                'source': 'yahoo_finance_market',
                'timestamp': stock['timestamp'],
                'tags': ['market_data', 'asia_tech', stock['symbol'].lower()],
                'metadata': stock
            }
            documents.append(doc)
        
        return documents
    
    async def _collect_news_data(self) -> List[Dict]:
        """Collect recent financial news"""
        news_articles = self.news_scraper.scrape_rss_feeds(hours_back=12)
        
        # Transform news articles for vector store compatibility
        documents = []
        for article in news_articles:
            # Create proper document structure
            doc = {
                'content_type': 'news_article',
                'title': article['title'],
                'content': f"{article['title']} {article.get('summary', '')}",  # Combine title and summary
                'summary': article.get('summary', ''),
                'source': f"news_{article['source']}",
                'timestamp': article['timestamp'],
                'link': article['link'],
                'published_date': article['published_date'],
                'tags': article.get('tags', []) + ['financial_news', 'market_news'],
                'metadata': article
            }
            
            # Filter for Asia tech related news
            if any(tag in ['asia', 'tech', 'tsmc', 'samsung', 'earnings', 'semiconductor'] 
                for tag in doc['tags']):
                documents.append(doc)
        
        return documents
    
    async def _collect_earnings_data(self) -> List[Dict]:
        """Collect earnings data for tracked symbols"""
        documents = []
        
        for symbol in self.asia_tech_symbols:  # Limit API calls
            earnings_data = self.alpha_vantage.get_earnings_data(symbol)
            
            for earning in earnings_data:
                doc = {
                'content_type': 'earnings_data',
                'symbol': earning['symbol'],
                'title': f"{earning['symbol']} Earnings - {earning['fiscal_date_ending']}",
                'content': f"""
                Fiscal Date: {earning['fiscal_date_ending']}
                Reported EPS: ${earning['reported_eps']}
                Estimated EPS: ${earning['estimated_eps']}
                Surprise: ${earning['surprise']} ({earning['surprise_percentage']}%)
                """,
                'source': 'alphavantage_earnings',
                'timestamp': earning['timestamp'],
                'tags': ['earnings', 'asia_tech', earning['symbol'].lower()],
                'metadata': {
                    'symbol': earning['symbol'],
                    'fiscal_date_ending': earning['fiscal_date_ending'],
                    'reported_eps': earning['reported_eps'],
                    'estimated_eps': earning['estimated_eps'],
                    'surprise': earning['surprise'],
                    'surprise_percentage': earning['surprise_percentage'],
                    'timestamp': earning['timestamp'],
                    'source': 'alphavantage_earnings'
                    # Remove the nested metadata structure
                }
            }
                documents.append(doc)
        
        return documents
    
    def schedule_regular_updates(self):
        """Schedule regular pipeline runs"""
        # Run every 4 hours during market hours
        schedule.every(4).hours.do(lambda: asyncio.run(self.run_full_pipeline()))
        
        # Run daily at 7 AM (before market open)
        schedule.every().day.at("07:00").do(lambda: asyncio.run(self.run_full_pipeline()))
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def validate_earnings_date(fiscal_date_str):
        """Validate that earnings date is not in the future"""
        try:
            fiscal_date = datetime.strptime(fiscal_date_str, '%Y-%m-%d').date()
            if fiscal_date > date.today():
                schedule.logger.warning(f"Future earnings date detected: {fiscal_date_str} - possible demo data")
                return False
            return True
        except ValueError:
            schedule.logger.error(f"Invalid date format: {fiscal_date_str}")
            return False