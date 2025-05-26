import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse

class WebDocumentLoader:
    """Loads and processes web documents and HTML content"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def load_webpage(self, url: str, content_selectors: Optional[List[str]] = None) -> Dict:
        """Load and extract content from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup, content_selectors)
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ""
            
            document = {
                'title': title_text,
                'content': content,
                'description': description,
                'source_url': url,
                'domain': urlparse(url).netloc,
                'content_type': 'web_document',
                'source': 'web_loader',
                'timestamp': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'tags': self._extract_tags_from_content(content)
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error loading webpage {url}: {e}")
            return {}
    
    def load_investor_relations_page(self, company_symbol: str, ir_url: str) -> List[Dict]:
        """Load investor relations page and extract relevant documents"""
        documents = []
        
        try:
            main_doc = self.load_webpage(ir_url)
            if main_doc:
                main_doc.update({
                    'symbol': company_symbol,
                    'content_type': 'investor_relations',
                    'tags': ['investor_relations', company_symbol.lower()]
                })
                documents.append(main_doc)
            
            # Look for earnings-related links
            response = requests.get(ir_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            earnings_links = self._find_earnings_links(soup, ir_url)
            
            for link_info in earnings_links[:3]:  # Limit to 3 most relevant
                earnings_doc = self.load_webpage(link_info['url'])
                if earnings_doc:
                    earnings_doc.update({
                        'symbol': company_symbol,
                        'content_type': 'earnings_material',
                        'document_type': link_info['type'],
                        'tags': ['earnings', 'investor_relations', company_symbol.lower()]
                    })
                    documents.append(earnings_doc)
                    
        except Exception as e:
            self.logger.error(f"Error loading IR page for {company_symbol}: {e}")
        
        return documents
    
    def _extract_main_content(self, soup: BeautifulSoup, selectors: Optional[List[str]]) -> str:
        """Extract main content using CSS selectors or fallback methods"""
        content_parts = []
        
        if selectors:
            # Use provided selectors
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    content_parts.append(element.get_text(strip=True))
        else:
            # Fallback: common content selectors
            common_selectors = [
                'article', 'main', '.content', '#content', 
                '.post-content', '.entry-content', '.article-body'
            ]
            
            for selector in common_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        content_parts.append(element.get_text(strip=True))
                    break
            
            # If no common selectors found, extract from body
            if not content_parts:
                body = soup.find('body')
                if body:
                    content_parts.append(body.get_text(strip=True))
        
        return '\n'.join(content_parts)
    
    def _find_earnings_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Find earnings-related links on the page"""
        earnings_keywords = [
            'earnings', 'quarterly', 'results', 'financial', 
            'transcript', 'webcast', 'presentation'
        ]
        
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True).lower()
            href = link['href']
            
            # Check if link text contains earnings keywords
            if any(keyword in link_text for keyword in earnings_keywords):
                full_url = urljoin(base_url, href)
                
                # Determine document type
                doc_type = 'earnings_general'
                if 'transcript' in link_text:
                    doc_type = 'earnings_transcript'
                elif 'presentation' in link_text:
                    doc_type = 'earnings_presentation'
                elif 'webcast' in link_text:
                    doc_type = 'earnings_webcast'
                
                links.append({
                    'url': full_url,
                    'text': link.get_text(strip=True),
                    'type': doc_type
                })
        
        return links
    
    def _extract_tags_from_content(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        content_lower = content.lower()
        
        # Financial keywords
        financial_keywords = [
            'revenue', 'earnings', 'profit', 'loss', 'growth',
            'market', 'stock', 'shares', 'dividend', 'guidance'
        ]
        
        # Technology keywords
        tech_keywords = [
            'technology', 'semiconductor', 'chip', 'ai', 'cloud',
            'software', 'hardware', 'innovation', 'digital'
        ]
        
        # Regional keywords
        regional_keywords = [
            'asia', 'china', 'taiwan', 'korea', 'japan', 'singapore'
        ]
        
        tags = []
        for keyword in financial_keywords + tech_keywords + regional_keywords:
            if keyword in content_lower:
                tags.append(keyword)
        
        return tags
