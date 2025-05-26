#!/usr/bin/env python3
"""
Standalone RSS feed scraper test that works independently
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict
from datetime import datetime, timedelta
import logging
import time
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StandaloneNewsScraper:
    """Standalone version of the news scraper for testing"""
    
    def __init__(self):
        # Updated sources based on your test results
        self.sources = {
            'marketwatch_topstories': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'marketwatch_realtime': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'cnbc_finance': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'cnbc_tech': 'https://www.cnbc.com/id/19854910/device/rss/rss.html',
            'google_finance': 'https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en',
            # Optional backup sources (test these periodically)
            # 'seeking_alpha': 'https://seekingalpha.com/api/sa/combined/LT.xml',
            # 'finviz_news': 'https://finviz.com/news.ashx',
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def scrape_all_sources(self, hours_back: int = 24) -> List[Dict]:
        """Main scraping method"""
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        successful_sources = 0
        failed_sources = []
        
        logger.info(f"🚀 Starting news scraping from {len(self.sources)} sources...")
        logger.info(f"📅 Looking for articles from the last {hours_back} hours")
        
        for source_name, feed_url in self.sources.items():
            try:
                logger.info(f"🔄 Fetching from {source_name}...")
                articles = self._scrape_single_source(source_name, feed_url, cutoff_time)
                
                if articles:
                    all_articles.extend(articles)
                    successful_sources += 1
                    logger.info(f"✅ {source_name}: {len(articles)} articles collected")
                else:
                    failed_sources.append(source_name)
                    logger.warning(f"⚠️  {source_name}: No recent articles found")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                failed_sources.append(source_name)
                logger.error(f"❌ {source_name} failed: {str(e)}")
                continue
        
        # Summary
        logger.info(f"📊 Scraping completed: {successful_sources}/{len(self.sources)} sources successful")
        logger.info(f"📰 Total articles collected: {len(all_articles)}")
        
        if failed_sources:
            logger.warning(f"❌ Failed sources: {', '.join(failed_sources)}")
        
        return all_articles
    
    def _scrape_single_source(self, source_name: str, feed_url: str, cutoff_time: datetime) -> List[Dict]:
        """Scrape a single RSS source"""
        articles = []
        
        try:
            # Primary method: requests + feedparser
            response = self.session.get(feed_url, timeout=15, allow_redirects=True)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                
                if feed.entries:
                    logger.debug(f"📡 {source_name}: Found {len(feed.entries)} total entries")
                    articles = self._process_feed_entries(feed, source_name, cutoff_time)
                else:
                    logger.warning(f"📡 {source_name}: Feed parsed but no entries found")
                    
            elif response.status_code == 429:
                logger.warning(f"🚫 {source_name}: Rate limited, waiting...")
                time.sleep(5)
                # Retry once
                response = self.session.get(feed_url, timeout=15)
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    if feed.entries:
                        articles = self._process_feed_entries(feed, source_name, cutoff_time)
            else:
                logger.warning(f"🌐 {source_name}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 {source_name}: Network error - {e}")
        except Exception as e:
            logger.error(f"⚠️  {source_name}: Unexpected error - {e}")
        
        return articles
    
    def _process_feed_entries(self, feed, source_name: str, cutoff_time: datetime) -> List[Dict]:
        """Process feed entries into standardized format"""
        articles = []
        processed = 0
        recent_count = 0
        
        for entry in feed.entries[:20]:  # Process max 20 per source to avoid overload
            try:
                processed += 1
                pub_date = self._parse_publication_date(entry)
                
                # Check if article is recent enough
                if pub_date and pub_date < cutoff_time:
                    continue
                
                recent_count += 1
                
                # Extract and clean content
                title = self._clean_text(entry.get('title', 'No Title'))
                summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                
                # Skip if title is too short (likely not a real article)
                if len(title) < 10:
                    continue
                
                # Create standardized article
                article = {
                    'title': title,
                    'summary': summary[:500] + '...' if len(summary) > 500 else summary,  # Truncate long summaries
                    'link': entry.get('link', ''),
                    'published_date': pub_date.isoformat() if pub_date else datetime.now().isoformat(),
                    'source': source_name,
                    'content_type': 'news_article',
                    'tags': self._extract_finance_tags(title + ' ' + summary),
                    'timestamp': datetime.now().isoformat(),
                    'word_count': len((title + ' ' + summary).split())
                }
                
                articles.append(article)
                
            except Exception as e:
                logger.debug(f"⚠️  Error processing entry from {source_name}: {e}")
                continue
        
        logger.debug(f"📊 {source_name}: Processed {processed} entries, {recent_count} recent, {len(articles)} added")
        return articles
    
    def _parse_publication_date(self, entry) -> datetime:
        """Parse publication date with fallbacks"""
        
        # Method 1: published_parsed
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except (TypeError, ValueError, OverflowError):
                pass
        
        # Method 2: published string
        if hasattr(entry, 'published') and entry.published:
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%d %b %Y %H:%M:%S %Z',
                '%Y-%m-%d'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(entry.published, fmt)
                except ValueError:
                    continue
        
        # Method 3: updated_parsed
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:6])
            except (TypeError, ValueError, OverflowError):
                pass
        
        # Fallback: assume recent
        return datetime.now() - timedelta(hours=2)
    
    def _clean_text(self, text: str) -> str:
        """Clean text content"""
        if not text:
            return ""
        
        # Remove HTML if present
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Clean whitespace and normalize
        text = ' '.join(text.split())
        return text.strip()
    
    def _extract_finance_tags(self, text: str) -> List[str]:
        """Extract financial/business tags"""
        if not text:
            return []
        
        keywords = {
            'market': ['market', 'trading', 'stocks', 'shares', 'nasdaq', 'nyse', 'dow', 's&p'],
            'earnings': ['earnings', 'eps', 'profit', 'revenue', 'results', 'quarterly'],
            'tech': ['technology', 'tech', 'ai', 'artificial intelligence', 'semiconductor', 'chips'],
            'finance': ['finance', 'investment', 'merger', 'acquisition', 'ipo', 'funding', 'bank'],
            'crypto': ['bitcoin', 'cryptocurrency', 'crypto', 'blockchain', 'ethereum'],
            'economy': ['economy', 'economic', 'gdp', 'inflation', 'fed', 'federal reserve'],
            'energy': ['oil', 'energy', 'renewable', 'solar', 'gas', 'petroleum'],
            'asia': ['china', 'japan', 'korea', 'asia', 'asian', 'taiwan', 'singapore'],
            'europe': ['europe', 'european', 'uk', 'germany', 'france', 'brexit'],
        }
        
        text_lower = text.lower()
        found_tags = set()
        
        for category, terms in keywords.items():
            if any(term in text_lower for term in terms):
                found_tags.add(category)
        
        # Add specific company/ticker tags if found
        tickers = ['tsmc', 'apple', 'microsoft', 'google', 'amazon', 'tesla', 'nvidia', 'meta']
        for ticker in tickers:
            if ticker in text_lower:
                found_tags.add(f'company_{ticker}')
        
        return list(found_tags)

def run_comprehensive_test():
    """Run comprehensive test of the scraper"""
    print("🧪 COMPREHENSIVE NEWS SCRAPER TEST")
    print("=" * 60)
    
    scraper = StandaloneNewsScraper()
    
    # Test 1: Quick connectivity test
    print("\n1️⃣  CONNECTIVITY TEST")
    print("-" * 30)
    
    working_sources = []
    for source_name, url in scraper.sources.items():
        try:
            response = scraper.session.get(url, timeout=10)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                if feed.entries:
                    print(f"✅ {source_name}: {len(feed.entries)} entries")
                    working_sources.append(source_name)
                else:
                    print(f"⚠️  {source_name}: No entries")
            else:
                print(f"❌ {source_name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {source_name}: {str(e)[:50]}...")
        
        time.sleep(0.5)
    
    print(f"\n📊 Working sources: {len(working_sources)}/{len(scraper.sources)}")
    
    # Test 2: Full scraping test
    print("\n2️⃣  FULL SCRAPING TEST")
    print("-" * 30)
    
    start_time = time.time()
    articles = scraper.scrape_all_sources(hours_back=48)  # Look back 48 hours for more results
    end_time = time.time()
    
    print(f"\n📈 RESULTS:")
    print(f"   ⏱️  Time taken: {end_time - start_time:.2f} seconds")
    print(f"   📰 Total articles: {len(articles)}")
    
    if articles:
        # Analyze results
        sources = {}
        tags = {}
        
        for article in articles:
            source = article['source']
            sources[source] = sources.get(source, 0) + 1
            
            for tag in article['tags']:
                tags[tag] = tags.get(tag, 0) + 1
        
        print(f"\n📊 BREAKDOWN BY SOURCE:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"   📰 {source}: {count} articles")
        
        print(f"\n🏷️  TOP TAGS:")
        for tag, count in sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   🏷️  {tag}: {count}")
        
        # Show sample articles
        print(f"\n📄 SAMPLE ARTICLES:")
        for i, article in enumerate(articles[:3], 1):
            print(f"\n   {i}. {article['title'][:80]}...")
            print(f"      📡 Source: {article['source']}")
            print(f"      📅 Published: {article['published_date'][:19]}")
            print(f"      🏷️  Tags: {', '.join(article['tags'][:5])}")
            if article['summary']:
                print(f"      📝 Summary: {article['summary'][:100]}...")
    
    # Test 3: Data quality check
    print(f"\n3️⃣  DATA QUALITY CHECK")
    print("-" * 30)
    
    if articles:
        # Check for duplicates
        titles = [a['title'] for a in articles]
        unique_titles = set(titles)
        duplicates = len(titles) - len(unique_titles)
        
        # Check date distribution
        now = datetime.now()
        recent_articles = sum(1 for a in articles 
                            if (now - datetime.fromisoformat(a['published_date'].replace('Z', '+00:00').replace('+00:00', ''))).total_seconds() < 24*3600)
        
        # Check content quality
        good_summaries = sum(1 for a in articles if len(a.get('summary', '')) > 50)
        
        print(f"   🔄 Duplicate titles: {duplicates}")
        print(f"   📅 Articles from last 24h: {recent_articles}")
        print(f"   📝 Articles with good summaries: {good_summaries}")
        print(f"   🔗 Articles with links: {sum(1 for a in articles if a.get('link'))}")
        
        quality_score = (
            (1 - duplicates/len(articles)) * 0.3 +
            (recent_articles/len(articles)) * 0.3 +
            (good_summaries/len(articles)) * 0.4
        ) * 100
        
        print(f"   📊 Overall quality score: {quality_score:.1f}%")
    
    return len(articles) > 0

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    print(f"\n🏁 FINAL RESULT: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("✅ News scraper is working and collecting articles!")
        sys.exit(0)
    else:
        print("❌ News scraper failed to collect articles.")
        sys.exit(1)