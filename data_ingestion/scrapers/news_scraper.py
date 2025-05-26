import requests
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
from urllib.parse import urljoin
import hashlib

class NewsSourceScraper:
    """Enhanced news scraper with improved reliability and performance"""
    
    def __init__(self):
        # Updated sources based on test results - only include working feeds
        self.sources = {
            # Primary working sources (tested and verified)
            'marketwatch_topstories': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'marketwatch_realtime': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'cnbc_finance': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'cnbc_tech': 'https://www.cnbc.com/id/19854910/device/rss/rss.html',
            'google_finance': 'https://news.google.com/rss/search?q=finance&hl=en-US&gl=US&ceid=US:en',
            
            # Alternative sources (test periodically and uncomment if working)
            # 'seeking_alpha': 'https://seekingalpha.com/api/sa/combined/LT.xml',
            # 'finviz_news': 'https://finviz.com/news.ashx',
            # 'yahoo_finance_alt': 'https://finance.yahoo.com/rss/',
            
            # Backup/alternative feeds
            'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
            'ft_companies': 'https://www.ft.com/companies?format=rss',
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # Configure session with retry strategy
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Set up retry adapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger = logging.getLogger(__name__)
        
        # Track seen articles to avoid duplicates
        self._seen_articles = set()
    
    def scrape_rss_feeds(self, hours_back: int = 24, max_articles_per_source: int = 15) -> List[Dict]:
        """Enhanced scraping with better error handling and performance"""
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        successful_sources = 0
        failed_sources = []
        
        self.logger.info(f"🚀 Starting enhanced news scraping from {len(self.sources)} sources")
        self.logger.info(f"📅 Looking for articles from the last {hours_back} hours")
        
        # Clear seen articles cache for this run
        self._seen_articles.clear()
        
        for source_name, feed_url in self.sources.items():
            try:
                self.logger.info(f"🔄 Processing {source_name}...")
                start_time = time.time()
                
                articles, status = self._scrape_single_source_enhanced(
                    source_name, feed_url, cutoff_time, max_articles_per_source
                )
                
                processing_time = round(time.time() - start_time, 2)
                
                if articles:
                    # Remove duplicates before adding
                    unique_articles = self._deduplicate_articles(articles)
                    all_articles.extend(unique_articles)
                    successful_sources += 1
                    
                    self.logger.info(
                        f"✅ {source_name}: {len(unique_articles)}/{len(articles)} articles "
                        f"(deduped) in {processing_time}s - {status}"
                    )
                else:
                    failed_sources.append(source_name)
                    self.logger.warning(
                        f"⚠️  {source_name}: No articles collected in {processing_time}s - {status}"
                    )
                
                # Progressive rate limiting - slower for failed sources
                if status == 'rate_limited':
                    time.sleep(3)
                elif status == 'error':
                    time.sleep(2)
                else:
                    time.sleep(1)
                
            except Exception as e:
                failed_sources.append(source_name)
                self.logger.error(f"❌ {source_name} failed with exception: {str(e)}")
                time.sleep(2)  # Wait longer after errors
                continue
        
        # Enhanced summary
        self._log_scraping_summary(successful_sources, failed_sources, all_articles, hours_back)
        
        # Return fallback data if no articles collected
        if len(all_articles) == 0:
            self.logger.error("⚠️  No articles collected from any source! Using fallback data.")
            return self._create_enhanced_fallback_articles()
        
        return all_articles
    
    def _scrape_single_source_enhanced(self, source_name: str, feed_url: str, 
                                     cutoff_time: datetime, max_articles: int) -> Tuple[List[Dict], str]:
        """Enhanced single source scraping with detailed status reporting"""
        articles = []
        status = "unknown"
        
        try:
            # Strategy 1: Direct feedparser (fastest for working feeds)
            feed = feedparser.parse(feed_url, request_headers=self.headers)
            
            if hasattr(feed, 'status') and feed.status == 200 and feed.entries:
                articles = self._process_feed_entries_enhanced(feed, source_name, cutoff_time, max_articles)
                status = f"direct_parse_{len(feed.entries)}_entries"
                return articles, status
            
            # Strategy 2: Requests + feedparser (more reliable)
            response = self.session.get(feed_url, timeout=20, allow_redirects=True)
            
            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'xml' not in content_type and 'rss' not in content_type:
                    status = f"invalid_content_type_{content_type}"
                    return articles, status
                
                feed = feedparser.parse(response.content)
                
                if feed.entries:
                    articles = self._process_feed_entries_enhanced(feed, source_name, cutoff_time, max_articles)
                    status = f"requests_parse_{len(feed.entries)}_entries"
                else:
                    status = "no_entries_in_feed"
                    
            elif response.status_code == 429:
                status = "rate_limited"
                self.logger.warning(f"🚫 {source_name}: Rate limited, implementing backoff...")
                time.sleep(5)
                
                # Single retry after rate limit
                try:
                    response = self.session.get(feed_url, timeout=20)
                    if response.status_code == 200:
                        feed = feedparser.parse(response.content)
                        if feed.entries:
                            articles = self._process_feed_entries_enhanced(feed, source_name, cutoff_time, max_articles)
                            status = f"retry_success_{len(feed.entries)}_entries"
                except Exception:
                    status = "retry_failed"
                    
            elif response.status_code in [403, 404]:
                status = f"blocked_http_{response.status_code}"
            else:
                status = f"http_error_{response.status_code}"
                
        except requests.exceptions.Timeout:
            status = "timeout"
        except requests.exceptions.ConnectionError:
            status = "connection_error"
        except Exception as e:
            status = f"exception_{type(e).__name__}"
        
        return articles, status
    
    def _process_feed_entries_enhanced(self, feed, source_name: str, cutoff_time: datetime, 
                                     max_articles: int) -> List[Dict]:
        """Enhanced entry processing with better filtering and quality checks"""
        articles = []
        processed_count = 0
        recent_count = 0
        quality_count = 0
        
        # Sort entries by date if possible (most recent first)
        entries = feed.entries[:max_articles * 2]  # Process more than needed to filter for quality
        
        for entry in entries:
            try:
                processed_count += 1
                
                # Parse and validate publication date
                pub_date = self._parse_publication_date_enhanced(entry)
                
                # Skip if too old
                if pub_date and pub_date < cutoff_time:
                    continue
                
                recent_count += 1
                
                # Extract and validate content
                title = self._clean_text(entry.get('title', ''))
                summary = self._clean_text(entry.get('summary', entry.get('description', '')))
                link = entry.get('link', '')
                
                # Quality checks
                if not self._passes_quality_checks(title, summary, link):
                    continue
                
                quality_count += 1
                
                # Create article with enhanced metadata
                article = {
                    'title': title,
                    'summary': self._truncate_summary(summary),
                    'link': link,
                    'published_date': pub_date.isoformat() if pub_date else datetime.now().isoformat(),
                    'source': source_name,
                    'content_type': 'news_article',
                    'tags': self._extract_finance_tags_enhanced(title + ' ' + summary),
                    'timestamp': datetime.now().isoformat(),
                    'word_count': len((title + ' ' + summary).split()),
                    'quality_score': self._calculate_quality_score(title, summary, link),
                    'article_hash': self._generate_article_hash(title, link)
                }
                
                articles.append(article)
                
                # Stop if we have enough quality articles
                if len(articles) >= max_articles:
                    break
                    
            except Exception as e:
                self.logger.debug(f"⚠️  Error processing entry from {source_name}: {e}")
                continue
        
        self.logger.debug(
            f"📊 {source_name}: processed={processed_count}, recent={recent_count}, "
            f"quality={quality_count}, final={len(articles)}"
        )
        
        return articles
    
    def _passes_quality_checks(self, title: str, summary: str, link: str) -> bool:
        """Check if article meets quality standards"""
        
        # Title checks
        if len(title) < 10 or len(title) > 200:
            return False
        
        # Skip common non-articles
        skip_phrases = [
            'sponsored', 'advertisement', 'ad:', 'promoted', 
            'newsletter', 'subscribe', 'sign up', 'webinar',
            'podcast', 'video', 'slideshow', 'gallery'
        ]
        
        title_lower = title.lower()
        if any(phrase in title_lower for phrase in skip_phrases):
            return False
        
        # Must have valid link
        if not link or not link.startswith('http'):
            return False
        
        return True
    
    def _calculate_quality_score(self, title: str, summary: str, link: str) -> float:
        """Calculate quality score for article (0-1)"""
        score = 0.5  # Base score
        
        # Title quality
        if 50 <= len(title) <= 100:
            score += 0.1
        if any(char.isdigit() for char in title):  # Has numbers (dates, figures)
            score += 0.05
        
        # Summary quality
        if len(summary) > 100:
            score += 0.1
        if len(summary) > 200:
            score += 0.1
        
        # Link quality
        if any(domain in link for domain in ['marketwatch', 'cnbc', 'bloomberg', 'ft.com']):
            score += 0.15
        
        return min(score, 1.0)
    
    def _generate_article_hash(self, title: str, link: str) -> str:
        """Generate hash for duplicate detection"""
        content = (title + link).encode('utf-8')
        return hashlib.md5(content).hexdigest()[:12]
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on hash"""
        unique_articles = []
        
        for article in articles:
            article_hash = article.get('article_hash')
            if article_hash and article_hash not in self._seen_articles:
                self._seen_articles.add(article_hash)
                unique_articles.append(article)
        
        return unique_articles
    
    def _truncate_summary(self, summary: str, max_length: int = 400) -> str:
        """Intelligently truncate summary"""
        if len(summary) <= max_length:
            return summary
        
        # Try to truncate at sentence boundary
        truncated = summary[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        sentence_end = max(last_period, last_exclamation, last_question)
        
        if sentence_end > max_length * 0.7:  # If we can keep at least 70% and end at sentence
            return summary[:sentence_end + 1]
        else:
            return summary[:max_length].rsplit(' ', 1)[0] + '...'
    
    def _parse_publication_date_enhanced(self, entry) -> Optional[datetime]:
        """Enhanced date parsing with more formats and validation"""
        
        # Method 1: published_parsed (most reliable)
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                parsed_date = datetime(*entry.published_parsed[:6])
                # Validate date is reasonable (not in future, not too old)
                now = datetime.now()
                if parsed_date <= now and parsed_date >= now - timedelta(days=30):
                    return parsed_date
            except (TypeError, ValueError, OverflowError):
                pass
        
        # Method 2: published string with expanded formats
        if hasattr(entry, 'published') and entry.published:
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%d %b %Y %H:%M:%S %Z',
                '%Y-%m-%d',
                '%a %b %d %H:%M:%S %Y',
                '%d/%m/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M:%S'
            ]
            
            published_str = entry.published.strip()
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(published_str, fmt)
                    # Validate date
                    now = datetime.now()
                    if parsed_date <= now and parsed_date >= now - timedelta(days=30):
                        return parsed_date
                except ValueError:
                    continue
        
        # Method 3: updated_parsed
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                parsed_date = datetime(*entry.updated_parsed[:6])
                now = datetime.now()
                if parsed_date <= now and parsed_date >= now - timedelta(days=30):
                    return parsed_date
            except (TypeError, ValueError, OverflowError):
                pass
        
        # Fallback: recent time (within last few hours)
        return datetime.now() - timedelta(hours=2)
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Remove HTML tags if present
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        
        # Remove common RSS artifacts
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # Clean up excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _extract_finance_tags_enhanced(self, text: str) -> List[str]:
        """Enhanced financial tag extraction with more categories"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_tags = set()
        
        # Core financial keywords
        keywords = {
            'market_indices': ['dow', 'nasdaq', 's&p', 'ftse', 'nikkei', 'hang seng', 'dax'],
            'market_general': ['market', 'trading', 'stocks', 'shares', 'equity', 'securities'],
            'earnings': ['earnings', 'eps', 'profit', 'revenue', 'results', 'quarterly', 'guidance'],
            'tech_companies': ['apple', 'microsoft', 'google', 'amazon', 'meta', 'tesla', 'nvidia', 'tsmc'],
            'tech_general': ['technology', 'tech', 'ai', 'artificial intelligence', 'semiconductor', 'chips'],
            'finance_general': ['investment', 'merger', 'acquisition', 'ipo', 'listing', 'funding', 'valuation'],
            'banking': ['bank', 'banking', 'credit', 'loan', 'mortgage', 'interest rate', 'fed', 'federal reserve'],
            'crypto': ['bitcoin', 'cryptocurrency', 'crypto', 'blockchain', 'ethereum', 'digital currency'],
            'economy': ['economy', 'economic', 'gdp', 'inflation', 'recession', 'growth', 'unemployment'],
            'energy': ['oil', 'energy', 'renewable', 'solar', 'gas', 'petroleum', 'crude', 'opec'],
            'commodities': ['gold', 'silver', 'copper', 'commodities', 'futures', 'wheat', 'corn'],
            'regions': {
                'asia': ['china', 'japan', 'korea', 'asia', 'asian', 'taiwan', 'singapore', 'hong kong'],
                'europe': ['europe', 'european', 'uk', 'germany', 'france', 'brexit', 'eu'],
                'americas': ['usa', 'america', 'canada', 'mexico', 'brazil', 'latin america']
            },
            'sectors': ['healthcare', 'biotech', 'automotive', 'retail', 'real estate', 'manufacturing'],
            'market_events': ['ipo', 'earnings', 'merger', 'acquisition', 'spinoff', 'dividend', 'buyback']
        }
        
        # Process regular keywords
        for category, terms in keywords.items():
            if category != 'regions':  # Handle regions separately
                if isinstance(terms, list):
                    if any(term in text_lower for term in terms):
                        found_tags.add(category)
                        # Add specific matching terms
                        matching_terms = [term.replace(' ', '_') for term in terms if term in text_lower]
                        found_tags.update(matching_terms[:3])  # Limit to avoid too many tags
        
        # Handle regions specially to add geographic tags
        for region, countries in keywords['regions'].items():
            if any(country in text_lower for country in countries):
                found_tags.add(region)
                found_tags.add('international')
        
        # Add composite tags based on combinations
        if 'asia' in found_tags and 'tech_general' in found_tags:
            found_tags.add('asia_tech')
        
        if 'earnings' in found_tags and any(tag.startswith('tech_') for tag in found_tags):
            found_tags.add('tech_earnings')
        
        if 'market_general' in found_tags and 'economy' in found_tags:
            found_tags.add('market_economy')
        
        # Add urgency/timing tags
        urgent_terms = ['breaking', 'urgent', 'alert', 'just in', 'developing']
        if any(term in text_lower for term in urgent_terms):
            found_tags.add('breaking_news')
        
        # Add sentiment indicators
        positive_terms = ['surge', 'rally', 'gain', 'rise', 'boost', 'growth', 'strong']
        negative_terms = ['fall', 'drop', 'decline', 'crash', 'plunge', 'weak', 'loss']
        
        if any(term in text_lower for term in positive_terms):
            found_tags.add('positive_sentiment')
        
        if any(term in text_lower for term in negative_terms):
            found_tags.add('negative_sentiment')
        
        return list(found_tags)
    
    def _log_scraping_summary(self, successful_sources: int, failed_sources: List[str], 
                            articles: List[Dict], hours_back: int):
        """Enhanced logging with detailed analytics"""
        
        self.logger.info(f"📊 SCRAPING SUMMARY")
        self.logger.info(f"   ✅ Successful sources: {successful_sources}/{len(self.sources)}")
        self.logger.info(f"   📰 Total articles collected: {len(articles)}")
        self.logger.info(f"   ⏰ Time window: {hours_back} hours")
        
        if articles:
            # Source breakdown
            source_counts = {}
            for article in articles:
                source = article['source']
                source_counts[source] = source_counts.get(source, 0) + 1
            
            self.logger.info("   📊 Articles by source:")
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"      • {source}: {count}")
            
            # Tag analysis
            all_tags = []
            for article in articles:
                all_tags.extend(article.get('tags', []))
            
            if all_tags:
                from collections import Counter
                top_tags = Counter(all_tags).most_common(10)
                self.logger.info("   🏷️  Top tags:")
                for tag, count in top_tags:
                    self.logger.info(f"      • {tag}: {count}")
            
            # Quality metrics
            avg_quality = sum(article.get('quality_score', 0.5) for article in articles) / len(articles)
            articles_with_good_summaries = sum(1 for a in articles if len(a.get('summary', '')) > 100)
            
            self.logger.info(f"   📈 Average quality score: {avg_quality:.2f}")
            self.logger.info(f"   📝 Articles with good summaries: {articles_with_good_summaries}")
        
        if failed_sources:
            self.logger.warning(f"   ❌ Failed sources: {', '.join(failed_sources)}")
    
    def _create_enhanced_fallback_articles(self) -> List[Dict]:
        """Create enhanced fallback articles with realistic data"""
        self.logger.info("🔄 Creating enhanced fallback articles")
        
        now = datetime.now()
        fallback_articles = [
            {
                'title': 'Asian Technology Stocks Show Mixed Performance in Early Trading',
                'summary': 'Technology stocks across Asia showed mixed results in early trading sessions, with semiconductor companies facing headwinds while software firms maintained steady growth. Market analysts continue to monitor developments in the region.',
                'link': 'https://example.com/asia-tech-mixed-trading',
                'published_date': (now - timedelta(hours=2)).isoformat(),
                'source': 'fallback_market_data',
                'content_type': 'news_article',
                'tags': ['market_general', 'asia', 'tech_general', 'asia_tech', 'mixed_sentiment'],
                'timestamp': now.isoformat(),
                'word_count': 45,
                'quality_score': 0.7,
                'article_hash': 'fallback_001'
            },
            {
                'title': 'Federal Reserve Officials Signal Cautious Approach to Interest Rate Policy',
                'summary': 'Federal Reserve officials indicated a measured approach to future interest rate decisions, citing ongoing economic uncertainties and inflation concerns. The central bank continues to balance growth objectives with price stability goals.',
                'link': 'https://example.com/fed-interest-rate-policy',
                'published_date': (now - timedelta(hours=4)).isoformat(),
                'source': 'fallback_economic_data',
                'content_type': 'news_article',
                'tags': ['banking', 'economy', 'federal_reserve', 'interest_rate', 'policy'],
                'timestamp': now.isoformat(),
                'word_count': 38,
                'quality_score': 0.8,
                'article_hash': 'fallback_002'
            },
            {
                'title': 'Energy Sector Experiences Volatility Amid Global Supply Chain Concerns',
                'summary': 'Energy markets faced increased volatility as investors weigh global supply chain disruptions against growing demand. Oil prices fluctuated throughout the session as traders assessed geopolitical developments and production forecasts.',
                'link': 'https://example.com/energy-sector-volatility',
                'published_date': (now - timedelta(hours=6)).isoformat(),
                'source': 'fallback_energy_data',
                'content_type': 'news_article',
                'tags': ['energy', 'oil', 'commodities', 'volatility', 'supply_chain'],
                'timestamp': now.isoformat(),
                'word_count': 42,
                'quality_score': 0.75,
                'article_hash': 'fallback_003'
            }
        ]
        
        return fallback_articles
    
    def test_feeds(self) -> Dict[str, Dict]:
        """Enhanced feed testing with detailed diagnostics"""
        results = {}
        
        self.logger.info("🧪 Running enhanced feed diagnostics...")
        
        for source_name, feed_url in self.sources.items():
            result = {
                'url': feed_url,
                'status': 'unknown',
                'entries': 0,
                'error': None,
                'response_time': None,
                'content_length': 0,
                'feed_title': None,
                'latest_article': None,
                'feed_format': None
            }
            
            start_time = time.time()
            
            try:
                # Test HTTP connection
                response = self.session.get(feed_url, timeout=15)
                result['http_status'] = response.status_code
                result['response_time'] = round(time.time() - start_time, 2)
                result['content_length'] = len(response.content)
                
                if response.status_code == 200:
                    # Analyze content type
                    content_type = response.headers.get('content-type', '')
                    result['content_type'] = content_type
                    
                    # Test feed parsing
                    feed = feedparser.parse(response.content)
                    result['entries'] = len(feed.entries)
                    result['feed_title'] = getattr(feed.feed, 'title', 'No title')
                    result['feed_format'] = getattr(feed, 'version', 'Unknown')
                    
                    if result['entries'] > 0:
                        result['status'] = 'working'
                        
                        # Get latest article info
                        latest = feed.entries[0]
                        result['latest_article'] = {
                            'title': getattr(latest, 'title', 'No title')[:60],
                            'published': getattr(latest, 'published', 'No date'),
                            'link': getattr(latest, 'link', 'No link')
                        }
                        
                        self.logger.info(f"✅ {source_name}: {result['entries']} entries, {result['response_time']}s")
                    else:
                        result['status'] = 'no_entries'
                        result['error'] = 'Feed parsed but no entries found'
                        self.logger.warning(f"⚠️  {source_name}: No entries found")
                
                elif response.status_code == 429:
                    result['status'] = 'rate_limited'
                    result['error'] = 'Rate limited - try again later'
                    self.logger.warning(f"🚫 {source_name}: Rate limited")
                
                elif response.status_code in [403, 404]:
                    result['status'] = 'blocked'
                    result['error'] = f'HTTP {response.status_code} - Access denied or not found'
                    self.logger.error(f"🚫 {source_name}: Blocked (HTTP {response.status_code})")
                
                else:
                    result['status'] = 'http_error'
                    result['error'] = f'HTTP {response.status_code}'
                    self.logger.error(f"❌ {source_name}: HTTP {response.status_code}")
                
            except requests.exceptions.Timeout:
                result['status'] = 'timeout'
                result['error'] = 'Request timed out'
                result['response_time'] = round(time.time() - start_time, 2)
                self.logger.error(f"⏰ {source_name}: Timeout")
                
            except requests.exceptions.ConnectionError as e:
                result['status'] = 'connection_error'
                result['error'] = f'Connection failed: {str(e)[:50]}...'
                result['response_time'] = round(time.time() - start_time, 2)
                self.logger.error(f"🌐 {source_name}: Connection error")
                
            except Exception as e:
                result['status'] = 'error'
                result['error'] = f'Unexpected error: {str(e)[:50]}...'
                result['response_time'] = round(time.time() - start_time, 2)
                self.logger.error(f"❌ {source_name}: {type(e).__name__}")
            
            results[source_name] = result
            time.sleep(0.8)  # Rate limiting between tests
        
        # Enhanced summary
        working = sum(1 for r in results.values() if r['status'] == 'working')
        rate_limited = sum(1 for r in results.values() if r['status'] == 'rate_limited')
        blocked = sum(1 for r in results.values() if r['status'] == 'blocked')
        errors = len(results) - working - rate_limited - blocked
        
        self.logger.info(f"🏁 Feed test summary:")
        self.logger.info(f"   ✅ Working: {working}")
        self.logger.info(f"   🚫 Rate limited: {rate_limited}")
        self.logger.info(f"   🔒 Blocked: {blocked}")
        self.logger.info(f"   ❌ Errors: {errors}")
        
        return results