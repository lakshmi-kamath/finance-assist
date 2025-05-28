import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import time
import random

from agents.base_agent import BaseAgent, Task, AgentResult, TaskPriority
from data_ingestion.scrapers.earnings_scraper import EarningsTranscriptScraper
from data_ingestion.scrapers.sec_filing_scraper import SECFilingScraper
from data_ingestion.scrapers.news_scraper import NewsSourceScraper

class ScrapingAgent(BaseAgent):
    """Agent that wraps all web scraping capabilities"""
    
    def __init__(self, agent_id: str = "scraping_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Initialize scrapers
        self.earnings_scraper = EarningsTranscriptScraper()
        self.sec_scraper = SECFilingScraper()
        self.news_scraper = NewsSourceScraper() 
        
        # Rate limiting configuration
        self.rate_limit_delay = self.config.get('rate_limit_delay', 2)
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 3)
        self.last_request_time = {}
    
    def _define_capabilities(self) -> List[str]:
        """Define what this agent can do"""
        return [
            'get_earnings_calendar',
            'scrape_earnings_analysis',
            'get_earnings_reports',
            'get_sec_filings',
            'get_foreign_filings',
            'extract_earnings_sentiment',
            'get_filing_content',
            'batch_earnings_scrape',
            'batch_filings_scrape',
            'comprehensive_company_analysis',
            'scrape_financial_news',
            'test_news_feeds',
            'get_recent_news',
            'analyze_news_sentiment'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['web_access', 'sec_edgar', 'seeking_alpha', 'yahoo_finance']
    
    async def execute_task(self, task: Task) -> AgentResult:
        """Execute scraping-related tasks"""
        task_type = task.type
        parameters = task.parameters
        
        try:
            if task_type == 'get_earnings_calendar':
                return await self._get_earnings_calendar(parameters)
            
            elif task_type == 'scrape_earnings_analysis':
                return await self._scrape_earnings_analysis(parameters)
            
            elif task_type == 'get_earnings_reports':
                return await self._get_earnings_reports(parameters)
            
            elif task_type == 'get_sec_filings':
                return await self._get_sec_filings(parameters)
            
            elif task_type == 'get_foreign_filings':
                return await self._get_foreign_filings(parameters)
            
            elif task_type == 'extract_earnings_sentiment':
                return await self._extract_earnings_sentiment(parameters)
            
            elif task_type == 'get_filing_content':
                return await self._get_filing_content(parameters)
            
            elif task_type == 'batch_earnings_scrape':
                return await self._batch_earnings_scrape(parameters)
            
            elif task_type == 'batch_filings_scrape':
                return await self._batch_filings_scrape(parameters)
            
            
            elif task_type == 'comprehensive_company_analysis':
                return await self._comprehensive_company_analysis(parameters)
        
        # ADD THESE NEWS-RELATED HANDLERS:
            elif task_type == 'scrape_financial_news':
                return await self._scrape_financial_news(parameters)
            
            elif task_type == 'test_news_feeds':
                return await self._test_news_feeds(parameters)
            
            elif task_type == 'get_recent_news':
                return await self._get_recent_news(parameters)
            
            elif task_type == 'analyze_news_sentiment':
                return await self._analyze_news_sentiment(parameters)
            
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
    
    async def _get_earnings_calendar(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get earnings calendar for given symbols"""
        symbols = parameters.get('symbols', [])
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        try:
            await self._apply_rate_limit('earnings_calendar')
            
            calendar_data = self.earnings_scraper.get_earnings_calendar(symbols)
            
            return AgentResult(
                success=True,
                data=calendar_data,
                metadata={
                    'symbols_requested': len(symbols),
                    'calendar_entries': len(calendar_data),
                    'data_sources': list(set(entry.get('source', 'unknown') for entry in calendar_data))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error getting earnings calendar: {e}")
            return AgentResult(success=False, error=str(e))
    
    async def _scrape_earnings_analysis(self, parameters: Dict[str, Any]) -> AgentResult:
        """Scrape earnings analysis articles"""
        symbols = parameters.get('symbols', [])
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        all_articles = []
        
        for symbol in symbols:
            try:
                await self._apply_rate_limit('seeking_alpha')
                
                articles = self.earnings_scraper.scrape_seeking_alpha_earnings(symbol)
                
                # Add sentiment analysis to each article
                for article in articles:
                    text_content = f"{article.get('title', '')} {article.get('summary', '')}"
                    sentiment = self.earnings_scraper.extract_earnings_sentiment(text_content)
                    article.update(sentiment)
                
                all_articles.extend(articles)
                
            except Exception as e:
                self.logger.error(f"Error scraping earnings analysis for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=all_articles,
            metadata={
                'symbols_requested': len(symbols),
                'articles_scraped': len(all_articles),
                'positive_articles': len([a for a in all_articles if a.get('sentiment_label') == 'positive']),
                'negative_articles': len([a for a in all_articles if a.get('sentiment_label') == 'negative'])
            }
        )
    
    async def _get_earnings_reports(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get earnings reports for symbols"""
        symbols = parameters.get('symbols', [])
        quarters = parameters.get('quarters', 4)
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        all_reports = []
        
        for symbol in symbols:
            try:
                await self._apply_rate_limit('yahoo_finance')
                
                reports = self.earnings_scraper.get_recent_earnings_reports(symbol, quarters)
                all_reports.extend(reports)
                
            except Exception as e:
                self.logger.error(f"Error getting earnings reports for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=all_reports,
            metadata={
                'symbols_requested': len(symbols),
                'reports_retrieved': len(all_reports),
                'quarters_per_symbol': quarters
            }
        )
    
    async def _get_sec_filings(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get SEC filings for symbols"""
        symbols = parameters.get('symbols', [])
        form_types = parameters.get('form_types', ['10-K', '10-Q', '8-K'])
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        all_filings = []
        
        for symbol in symbols:
            try:
                await self._apply_rate_limit('sec_edgar')
                
                filings = self.sec_scraper.get_company_filings(symbol, form_types)
                all_filings.extend(filings)
                
            except Exception as e:
                self.logger.error(f"Error getting SEC filings for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=all_filings,
            metadata={
                'symbols_requested': len(symbols),
                'filings_retrieved': len(all_filings),
                'form_types': form_types,
                'exchanges': list(set(filing.get('exchange', 'unknown') for filing in all_filings))
            }
        )
    
    async def _get_foreign_filings(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get foreign exchange filings"""
        symbols = parameters.get('symbols', [])
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        # Filter for foreign symbols
        foreign_symbols = [s for s in symbols if any(ext in s for ext in ['.KS', '.T', '.HK'])]
        
        if not foreign_symbols:
            return AgentResult(
                success=True,
                data=[],
                metadata={'message': 'No foreign symbols found in the provided list'}
            )
        
        all_filings = []
        
        for symbol in foreign_symbols:
            try:
                await self._apply_rate_limit('foreign_exchange')
                
                filings = self.sec_scraper.get_company_filings(symbol)
                all_filings.extend(filings)
                
            except Exception as e:
                self.logger.error(f"Error getting foreign filings for {symbol}: {e}")
        
        return AgentResult(
            success=True,
            data=all_filings,
            metadata={
                'foreign_symbols': len(foreign_symbols),
                'filings_retrieved': len(all_filings),
                'exchanges': list(set(filing.get('exchange', 'unknown') for filing in all_filings))
            }
        )
    
    async def _extract_earnings_sentiment(self, parameters: Dict[str, Any]) -> AgentResult:
        """Extract sentiment from earnings text"""
        texts = parameters.get('texts', [])
        text = parameters.get('text', '')
        
        if not texts and not text:
            return AgentResult(success=False, error="No text provided for sentiment analysis")
        
        if text:
            texts = [text]
        
        sentiment_results = []
        
        for idx, text_content in enumerate(texts):
            try:
                sentiment = self.earnings_scraper.extract_earnings_sentiment(text_content)
                sentiment['text_id'] = idx
                sentiment['text_preview'] = text_content[:100] + "..." if len(text_content) > 100 else text_content
                sentiment_results.append(sentiment)
                
            except Exception as e:
                self.logger.error(f"Error extracting sentiment for text {idx}: {e}")
        
        # Calculate aggregate sentiment
        if sentiment_results:
            avg_sentiment = sum(s['sentiment_score'] for s in sentiment_results) / len(sentiment_results)
            total_positive = sum(s['positive_indicators'] for s in sentiment_results)
            total_negative = sum(s['negative_indicators'] for s in sentiment_results)
        else:
            avg_sentiment = 0.0
            total_positive = 0
            total_negative = 0
        
        return AgentResult(
            success=True,
            data=sentiment_results,
            metadata={
                'texts_analyzed': len(texts),
                'average_sentiment': round(avg_sentiment, 3),
                'total_positive_indicators': total_positive,
                'total_negative_indicators': total_negative,
                'overall_sentiment': 'positive' if avg_sentiment > 0.2 else 'negative' if avg_sentiment < -0.2 else 'neutral'
            }
        )
    
    async def _get_filing_content(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get content from filing URLs"""
        urls = parameters.get('urls', [])
        url = parameters.get('url', '')
        
        if not urls and not url:
            return AgentResult(success=False, error="No URLs provided")
        
        if url:
            urls = [url]
        
        contents = []
        
        for filing_url in urls:
            try:
                await self._apply_rate_limit('filing_content')
                
                content = self.sec_scraper.get_filing_content(filing_url)
                
                if content:
                    contents.append({
                        'url': filing_url,
                        'content': content,
                        'content_length': len(content),
                        'extracted_at': datetime.now().isoformat()
                    })
                
            except Exception as e:
                self.logger.error(f"Error getting content from {filing_url}: {e}")
                contents.append({
                    'url': filing_url,
                    'content': None,
                    'error': str(e),
                    'extracted_at': datetime.now().isoformat()
                })
        
        return AgentResult(
            success=True,
            data=contents,
            metadata={
                'urls_requested': len(urls),
                'successful_extractions': len([c for c in contents if c.get('content')]),
                'total_content_length': sum(c.get('content_length', 0) for c in contents)
            }
        )
    
    async def _batch_earnings_scrape(self, parameters: Dict[str, Any]) -> AgentResult:
        """Comprehensive earnings scraping for multiple symbols"""
        symbols = parameters.get('symbols', [])
        include_calendar = parameters.get('include_calendar', True)
        include_analysis = parameters.get('include_analysis', True)
        include_reports = parameters.get('include_reports', True)
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        results = {
            'earnings_calendar': [],
            'earnings_analysis': [],
            'earnings_reports': [],
            'summary': {}
        }
        
        # Get earnings calendar
        if include_calendar:
            calendar_result = await self._get_earnings_calendar({'symbols': symbols})
            if calendar_result.success:
                results['earnings_calendar'] = calendar_result.data
        
        # Get earnings analysis
        if include_analysis:
            analysis_result = await self._scrape_earnings_analysis({'symbols': symbols})
            if analysis_result.success:
                results['earnings_analysis'] = analysis_result.data
        
        # Get earnings reports
        if include_reports:
            reports_result = await self._get_earnings_reports({'symbols': symbols})
            if reports_result.success:
                results['earnings_reports'] = reports_result.data
        
        # Create summary
        results['summary'] = {
            'symbols_processed': len(symbols),
            'calendar_entries': len(results['earnings_calendar']),
            'analysis_articles': len(results['earnings_analysis']),
            'earnings_reports': len(results['earnings_reports']),
            'positive_sentiment_articles': len([
                a for a in results['earnings_analysis'] 
                if a.get('sentiment_label') == 'positive'
            ]),
            'negative_sentiment_articles': len([
                a for a in results['earnings_analysis'] 
                if a.get('sentiment_label') == 'negative'
            ])
        }
        
        return AgentResult(
            success=True,
            data=results,
            metadata=results['summary']
        )
    
    async def _batch_filings_scrape(self, parameters: Dict[str, Any]) -> AgentResult:
        """Comprehensive filings scraping for multiple symbols"""
        symbols = parameters.get('symbols', [])
        form_types = parameters.get('form_types', ['10-K', '10-Q', '8-K'])
        include_foreign = parameters.get('include_foreign', True)
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        results = {
            'sec_filings': [],
            'foreign_filings': [],
            'summary': {}
        }
        
        # Get SEC filings
        sec_result = await self._get_sec_filings({
            'symbols': symbols, 
            'form_types': form_types
        })
        if sec_result.success:
            results['sec_filings'] = sec_result.data
        
        # Get foreign filings if requested
        if include_foreign:
            foreign_result = await self._get_foreign_filings({'symbols': symbols})
            if foreign_result.success:
                results['foreign_filings'] = foreign_result.data
        
        # Create summary
        all_filings = results['sec_filings'] + results['foreign_filings']
        exchanges = list(set(filing.get('exchange', 'unknown') for filing in all_filings))
        
        results['summary'] = {
            'symbols_processed': len(symbols),
            'total_filings': len(all_filings),
            'sec_filings': len(results['sec_filings']),
            'foreign_filings': len(results['foreign_filings']),
            'exchanges': exchanges,
            'form_types': form_types
        }
        
        return AgentResult(
            success=True,
            data=results,
            metadata=results['summary']
        )
    
    async def _comprehensive_company_analysis(self, parameters: Dict[str, Any]) -> AgentResult:
        """Comprehensive analysis combining all scraping capabilities"""
        symbols = parameters.get('symbols', [])
        
        if not symbols:
            return AgentResult(success=False, error="No symbols provided")
        
        # Limit symbols to avoid overwhelming requests
        symbols = symbols[:5]  # Max 5 symbols at once
        
        comprehensive_data = {}
        
        for symbol in symbols:
            self.logger.info(f"Starting comprehensive analysis for {symbol}")
            
            symbol_data = {
                'symbol': symbol,
                'earnings_calendar': [],
                'earnings_analysis': [],
                'earnings_reports': [],
                'sec_filings': [],
                'foreign_filings': [],
                'sentiment_summary': {},
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            try:
                # Get earnings calendar
                calendar_result = await self._get_earnings_calendar({'symbols': [symbol]})
                if calendar_result.success:
                    symbol_data['earnings_calendar'] = calendar_result.data
                
                # Get earnings analysis with sentiment
                analysis_result = await self._scrape_earnings_analysis({'symbols': [symbol]})
                if analysis_result.success:
                    symbol_data['earnings_analysis'] = analysis_result.data
                    
                    # Calculate sentiment summary
                    articles = analysis_result.data
                    if articles:
                        avg_sentiment = sum(a.get('sentiment_score', 0) for a in articles) / len(articles)
                        symbol_data['sentiment_summary'] = {
                            'average_sentiment_score': round(avg_sentiment, 3),
                            'total_articles': len(articles),
                            'positive_articles': len([a for a in articles if a.get('sentiment_label') == 'positive']),
                            'negative_articles': len([a for a in articles if a.get('sentiment_label') == 'negative']),
                            'neutral_articles': len([a for a in articles if a.get('sentiment_label') == 'neutral']),
                            'overall_sentiment': 'positive' if avg_sentiment > 0.2 else 'negative' if avg_sentiment < -0.2 else 'neutral'
                        }
                
                # Get earnings reports
                reports_result = await self._get_earnings_reports({'symbols': [symbol]})
                if reports_result.success:
                    symbol_data['earnings_reports'] = reports_result.data
                
                # Get SEC filings
                sec_result = await self._get_sec_filings({'symbols': [symbol]})
                if sec_result.success:
                    symbol_data['sec_filings'] = sec_result.data
                
                # Get foreign filings if applicable
                if any(ext in symbol for ext in ['.KS', '.T', '.HK']):
                    foreign_result = await self._get_foreign_filings({'symbols': [symbol]})
                    if foreign_result.success:
                        symbol_data['foreign_filings'] = foreign_result.data
                
                comprehensive_data[symbol] = symbol_data
                
            except Exception as e:
                self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
                symbol_data['error'] = str(e)
                comprehensive_data[symbol] = symbol_data
        
        # Create overall summary
        total_articles = sum(len(data.get('earnings_analysis', [])) for data in comprehensive_data.values())
        total_filings = sum(
            len(data.get('sec_filings', [])) + len(data.get('foreign_filings', []))
            for data in comprehensive_data.values()
        )
        
        overall_summary = {
            'symbols_analyzed': len(symbols),
            'total_earnings_articles': total_articles,
            'total_filings': total_filings,
            'analysis_completed_at': datetime.now().isoformat(),
            'data_sources': ['seeking_alpha', 'yahoo_finance', 'sec_edgar', 'foreign_exchanges']
        }
        
        return AgentResult(
            success=True,
            data=comprehensive_data,
            metadata=overall_summary
        )
    
    async def _apply_rate_limit(self, source: str):
        """Apply rate limiting for different sources"""
        current_time = time.time()
        last_time = self.last_request_time.get(source, 0)
        
        if current_time - last_time < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - (current_time - last_time)
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s for {source}")
            await asyncio.sleep(wait_time)
        
        self.last_request_time[source] = time.time()
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including scraper status"""
        base_health = super().health_check()
        
        # Add scraper-specific health info
        scraper_status = {
            'earnings_scraper': self.earnings_scraper is not None,
            'sec_scraper': self.sec_scraper is not None,
            'news_scraper': self.news_scraper is not None,
            'rate_limiting_active': bool(self.rate_limit_delay > 0)
        }
        
        base_health['scrapers'] = scraper_status
        base_health['healthy'] = base_health['healthy'] and all(scraper_status.values())
        
        return base_health
    
    async def _scrape_financial_news(self, parameters: Dict[str, Any]) -> AgentResult:
        """Scrape financial news from RSS feeds"""
        hours_back = parameters.get('hours_back', 24)
        max_articles_per_source = parameters.get('max_articles_per_source', 15)
        
        try:
            await self._apply_rate_limit('news_feeds')
            
            articles = self.news_scraper.scrape_rss_feeds(
                hours_back=hours_back,
                max_articles_per_source=max_articles_per_source
            )
            
            return AgentResult(
                success=True,
                data=articles,
                metadata={
                    'articles_collected': len(articles),
                    'hours_back': hours_back,
                    'sources_used': list(set(article.get('source', 'unknown') for article in articles)),
                    'avg_quality_score': round(sum(article.get('quality_score', 0.5) for article in articles) / len(articles), 3) if articles else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error scraping financial news: {e}")
            return AgentResult(success=False, error=str(e))

    async def _test_news_feeds(self, parameters: Dict[str, Any]) -> AgentResult:
        """Test all news feed sources"""
        try:
            await self._apply_rate_limit('news_feeds_test')
            
            feed_results = self.news_scraper.test_feeds()
            
            # Summarize results
            working_feeds = [name for name, result in feed_results.items() if result['status'] == 'working']
            failed_feeds = [name for name, result in feed_results.items() if result['status'] not in ['working', 'rate_limited']]
            rate_limited_feeds = [name for name, result in feed_results.items() if result['status'] == 'rate_limited']
            
            return AgentResult(
                success=True,
                data=feed_results,
                metadata={
                    'total_feeds': len(feed_results),
                    'working_feeds': len(working_feeds),
                    'failed_feeds': len(failed_feeds),
                    'rate_limited_feeds': len(rate_limited_feeds),
                    'working_feed_names': working_feeds,
                    'failed_feed_names': failed_feeds
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error testing news feeds: {e}")
            return AgentResult(success=False, error=str(e))

    async def _get_recent_news(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get recent news with optional filtering"""
        hours_back = parameters.get('hours_back', 12)
        tags_filter = parameters.get('tags_filter', [])  # Filter by specific tags
        min_quality_score = parameters.get('min_quality_score', 0.5)
        
        try:
            await self._apply_rate_limit('news_feeds')
            
            articles = self.news_scraper.scrape_rss_feeds(hours_back=hours_back)
            
            # Apply filters
            filtered_articles = []
            for article in articles:
                # Quality filter
                if article.get('quality_score', 0.5) < min_quality_score:
                    continue
                
                # Tags filter
                if tags_filter:
                    article_tags = article.get('tags', [])
                    if not any(tag in article_tags for tag in tags_filter):
                        continue
                
                filtered_articles.append(article)
            
            return AgentResult(
                success=True,
                data=filtered_articles,
                metadata={
                    'total_articles_scraped': len(articles),
                    'articles_after_filtering': len(filtered_articles),
                    'filters_applied': {
                        'min_quality_score': min_quality_score,
                        'tags_filter': tags_filter
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error getting recent news: {e}")
            return AgentResult(success=False, error=str(e))

    async def _analyze_news_sentiment(self, parameters: Dict[str, Any]) -> AgentResult:
        """Analyze sentiment of news articles"""
        articles = parameters.get('articles', [])
        hours_back = parameters.get('hours_back', 24)
        max_articles_per_source = parameters.get('max_articles_per_source', 15)
        
        # If no articles provided, scrape recent ones with the same parameters
        if not articles:
            try:
                articles = self.news_scraper.scrape_rss_feeds(
                    hours_back=hours_back,
                    max_articles_per_source=max_articles_per_source
                )
            except Exception as e:
                return AgentResult(success=False, error=f"Error scraping articles for sentiment analysis: {e}")
        
        if not articles:
            return AgentResult(success=False, error="No articles available for sentiment analysis")
        
        sentiment_results = []
        
        for article in articles:
            try:
                # Combine title and summary for sentiment analysis
                text_content = f"{article.get('title', '')} {article.get('summary', '')}"
                
                # Use earnings scraper's sentiment analysis (assuming it's general enough)
                sentiment = self.earnings_scraper.extract_earnings_sentiment(text_content)
                
                # Add article metadata
                sentiment_result = {
                    'article_title': article.get('title', ''),
                    'article_source': article.get('source', ''),
                    'article_tags': article.get('tags', []),
                    'sentiment_score': sentiment.get('sentiment_score', 0),
                    'sentiment_label': sentiment.get('sentiment_label', 'neutral'),
                    'positive_indicators': sentiment.get('positive_indicators', 0),
                    'negative_indicators': sentiment.get('negative_indicators', 0),
                    'article_link': article.get('link', '')
                }
                
                sentiment_results.append(sentiment_result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment for article: {e}")
                continue
        
        # Calculate aggregate sentiment
        if sentiment_results:
            avg_sentiment = sum(r['sentiment_score'] for r in sentiment_results) / len(sentiment_results)
            positive_count = len([r for r in sentiment_results if r['sentiment_label'] == 'positive'])
            negative_count = len([r for r in sentiment_results if r['sentiment_label'] == 'negative'])
            neutral_count = len(sentiment_results) - positive_count - negative_count
        else:
            avg_sentiment = 0.0
            positive_count = negative_count = neutral_count = 0
        
        return AgentResult(
            success=True,
            data=sentiment_results,
            metadata={
                'articles_analyzed': len(sentiment_results),
                'average_sentiment_score': round(avg_sentiment, 3),
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': neutral_count,
                'overall_market_sentiment': 'positive' if avg_sentiment > 0.2 else 'negative' if avg_sentiment < -0.2 else 'neutral'
            }
        )

# Utility function to create configured scraping agent
def create_scraping_agent(config: Dict[str, Any] = None) -> ScrapingAgent:
    """Factory function to create configured scraping agent"""
    default_config = {
        'rate_limit_delay': 2,  # seconds between requests
        'max_concurrent_requests': 3,
        'timeout_seconds': 300
    }
    
    if config:
        default_config.update(config)
    
    return ScrapingAgent(config=default_config)



# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_scraping_agent():
        """Test the scraping agent functionality"""
        agent = create_scraping_agent()
        
        print(f"Agent Status: {agent.get_status()}")
        
        # Test earnings calendar
        print("\n--- Testing Earnings Calendar ---")
        result = await agent.execute('get_earnings_calendar', {'symbols': ['AAPL', 'MSFT']})
        print(f"Earnings Calendar: {result.success}, Data count: {len(result.data) if result.success else 0}")
        
        # Test earnings analysis
        print("\n--- Testing Earnings Analysis ---")
        result = await agent.execute('scrape_earnings_analysis', {'symbols': ['AAPL']})
        print(f"Earnings Analysis: {result.success}, Articles: {len(result.data) if result.success else 0}")
        
        # Test comprehensive analysis
        print("\n--- Testing Comprehensive Analysis ---")
        result = await agent.execute('comprehensive_company_analysis', {'symbols': ['AAPL']})
        if result.success:
            print(f"Comprehensive Analysis Complete: {len(result.data)} companies analyzed")
            print(f"Summary: {result.metadata}")
        else:
            print(f"Error: {result.error}")
        
        print("\n--- Testing News Feeds ---")
        result = await agent.execute('test_news_feeds', {})
        if result.success:
            print(f"Feed Test Complete: {result.metadata['working_feeds']}/{result.metadata['total_feeds']} feeds working")
        
        print("\n--- Testing Financial News Scraping ---")
        result = await agent.execute('scrape_financial_news', {'hours_back': 6, 'max_articles_per_source': 5})
        print(f"News Scraping: {result.success}, Articles: {len(result.data) if result.success else 0}")
        
        print("\n--- Testing News Sentiment Analysis ---")
        result = await agent.execute('analyze_news_sentiment', {'hours_back': 12})
        if result.success:
            print(f"Sentiment Analysis: {result.metadata['articles_analyzed']} articles, Overall: {result.metadata['overall_market_sentiment']}")
        
        print(f"\nFinal Agent Status: {agent.get_status()}")
    
    # Run test
    asyncio.run(test_scraping_agent())

