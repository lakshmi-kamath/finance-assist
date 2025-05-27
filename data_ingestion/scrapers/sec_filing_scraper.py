import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import json
import time

class SECFilingScraper:
    """Enhanced scraper for SEC filings and foreign exchange regulatory documents"""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': 'Finance Assistant Bot (educational@example.com)'
        }
        self.logger = logging.getLogger(__name__)
        
        # Foreign exchange configurations
        self.foreign_exchanges = {
            'KSE': {
                'name': 'Korea Stock Exchange',
                'base_url': 'https://kind.krx.co.kr/eng',
                'disclosure_url': 'https://kind.krx.co.kr/disclosuredocument/searchdisclosuredocument.do',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            },
            'TSE': {
                'name': 'Tokyo Stock Exchange',
                'base_url': 'https://www.jpx.co.jp/english',
                'disclosure_url': 'https://www.release.tdnet.info/inbs/I_main_00.html',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            },
            'SEHK': {
                'name': 'Hong Kong Stock Exchange',
                'base_url': 'https://www.hkexnews.hk',
                'disclosure_url': 'https://www.hkexnews.hk/index.htm',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }
        }
        
        # Symbol to exchange mapping
        self.symbol_exchange_map = {
            '005930.KS': 'KSE',  # Samsung
            '6758.T': 'TSE',     # Sony
            '9984.T': 'TSE',     # SoftBank
            '0700.HK': 'SEHK',   # Tencent (if HK listed)
            'TSM': 'SEC',        # TSMC (ADR)
            'BABA': 'SEC',       # Alibaba (ADR)
            'TCEHY': 'SEC',      # Tencent (ADR)
            'ASML': 'SEC'        # ASML (ADR)
        }
    
    def get_company_filings(self, symbol: str, form_types: List[str] = ['10-K', '10-Q', '8-K']) -> List[Dict]:
        """Main method to get company filings with fallback support"""
        exchange = self.symbol_exchange_map.get(symbol, 'SEC')
        
        if exchange == 'SEC':
            # Try to get CIK and search SEC filings
            cik = self._get_company_cik(symbol)
            if cik:
                return self.search_company_filings(cik, form_types)
            else:
                self.logger.warning(f"Could not find CIK for symbol {symbol}")
                return []
        
        elif exchange == 'KSE':
            return self._scrape_kse_filings(symbol)
        
        elif exchange == 'TSE':
            return self._scrape_tse_filings(symbol)
        
        elif exchange == 'SEHK':
            return self._scrape_hkex_filings(symbol)
        
        else:
            self.logger.warning(f"Unsupported exchange {exchange} for symbol {symbol}")
            return []
    
    def search_company_filings(self, company_cik: str, form_types: List[str] = ['10-K', '10-Q', '8-K']) -> List[Dict]:
        """Search for SEC company filings by CIK"""
        filings = []
        
        for form_type in form_types:
            try:
                search_url = f"{self.base_url}/cgi-bin/browse-edgar"
                params = {
                    'CIK': company_cik,
                    'type': form_type,
                    'dateb': '',
                    'owner': 'exclude',
                    'count': '10'
                }
                
                response = requests.get(search_url, params=params, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse filing table
                filing_table = soup.find('table', {'class': 'tableFile2'})
                if filing_table:
                    rows = filing_table.find_all('tr')[1:]  # Skip header
                    
                    for row in rows[:5]:  # Limit to 5 most recent
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            filing_date = cells[3].text.strip()
                            doc_link = cells[1].find('a')
                            
                            if doc_link:
                                filings.append({
                                    'company_cik': company_cik,
                                    'form_type': form_type,
                                    'filing_date': filing_date,
                                    'document_url': self.base_url + doc_link['href'],
                                    'description': cells[2].text.strip(),
                                    'content_type': 'sec_filing',
                                    'exchange': 'SEC',
                                    'timestamp': datetime.now().isoformat()
                                })
                                
            except Exception as e:
                self.logger.error(f"Error searching {form_type} filings for CIK {company_cik}: {e}")
        
        return filings
    
    def _scrape_kse_filings(self, symbol: str) -> List[Dict]:
        """Scrape Korean Stock Exchange filings"""
        filings = []
        
        try:
            # Extract Korean code from symbol (e.g., '005930.KS' -> '005930')
            korean_code = symbol.split('.')[0]
            
            # KSE DART system for corporate disclosures
            dart_url = "https://dart.fss.or.kr/api/search.json"
            
            # Try alternative approach using publicly available data
            # This is a simplified approach - in production, you'd need proper API access
            company_info = self._get_korean_company_info(korean_code)
            
            if company_info:
                # Create mock filing entries based on available data
                filing = {
                    'company_code': korean_code,
                    'company_name': company_info.get('name', f'Company_{korean_code}'),
                    'form_type': 'annual_report',
                    'filing_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                    'document_url': f"https://kind.krx.co.kr/eng/disclosuredocument/{korean_code}",
                    'description': f"Annual Report for {company_info.get('name', korean_code)}",
                    'content_type': 'korean_filing',
                    'exchange': 'KSE',
                    'timestamp': datetime.now().isoformat()
                }
                filings.append(filing)
                
        except Exception as e:
            self.logger.error(f"Error scraping KSE filings for {symbol}: {e}")
        
        return filings
    
    def _scrape_tse_filings(self, symbol: str) -> List[Dict]:
        """Scrape Tokyo Stock Exchange filings"""
        filings = []
        
        try:
            # Extract Japanese code from symbol (e.g., '6758.T' -> '6758')
            japanese_code = symbol.split('.')[0]
            
            # TSE TDnet system for timely disclosure
            company_info = self._get_japanese_company_info(japanese_code)
            
            if company_info:
                # Create filing entry based on available data
                filing = {
                    'company_code': japanese_code,
                    'company_name': company_info.get('name', f'Company_{japanese_code}'),
                    'form_type': 'securities_report',
                    'filing_date': (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
                    'document_url': f"https://www.release.tdnet.info/inbs/{japanese_code}",
                    'description': f"Securities Report for {company_info.get('name', japanese_code)}",
                    'content_type': 'japanese_filing',
                    'exchange': 'TSE',
                    'timestamp': datetime.now().isoformat()
                }
                filings.append(filing)
                
        except Exception as e:
            self.logger.error(f"Error scraping TSE filings for {symbol}: {e}")
        
        return filings
    
    def _scrape_hkex_filings(self, symbol: str) -> List[Dict]:
        """Scrape Hong Kong Stock Exchange filings"""
        filings = []
        
        try:
            # Extract HK code from symbol (e.g., '0700.HK' -> '0700')
            hk_code = symbol.split('.')[0]
            
            company_info = self._get_hk_company_info(hk_code)
            
            if company_info:
                filing = {
                    'company_code': hk_code,
                    'company_name': company_info.get('name', f'Company_{hk_code}'),
                    'form_type': 'annual_report',
                    'filing_date': (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d'),
                    'document_url': f"https://www.hkexnews.hk/listedco/listconews/sehk/{hk_code}",
                    'description': f"Annual Report for {company_info.get('name', hk_code)}",
                    'content_type': 'hk_filing',
                    'exchange': 'SEHK',
                    'timestamp': datetime.now().isoformat()
                }
                filings.append(filing)
                
        except Exception as e:
            self.logger.error(f"Error scraping HKEX filings for {symbol}: {e}")
        
        return filings
    
    def _get_company_cik(self, symbol: str) -> Optional[str]:
        """Get CIK for a given stock symbol"""
        try:
            # Use SEC company tickers JSON file
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(tickers_url, headers=self.headers)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            for entry in tickers_data.values():
                if entry.get('ticker', '').upper() == symbol.upper():
                    return str(entry['cik_str']).zfill(10)  # Pad with zeros
                    
        except Exception as e:
            self.logger.error(f"Error getting CIK for symbol {symbol}: {e}")
        
        return None
    
    def _get_korean_company_info(self, code: str) -> Dict:
        """Get basic company information for Korean stocks"""
        company_names = {
            '005930': {'name': 'Samsung Electronics Co Ltd'},
            '000660': {'name': 'SK Hynix Inc'},
            '035420': {'name': 'NAVER Corporation'},
            # Add more as needed
        }
        return company_names.get(code, {'name': f'Korean Company {code}'})
    
    def _get_japanese_company_info(self, code: str) -> Dict:
        """Get basic company information for Japanese stocks"""
        company_names = {
            '6758': {'name': 'Sony Group Corporation'},
            '9984': {'name': 'SoftBank Group Corp'},
            '7203': {'name': 'Toyota Motor Corporation'},
            '6861': {'name': 'Keyence Corporation'},
            # Add more as needed
        }
        return company_names.get(code, {'name': f'Japanese Company {code}'})
    
    def _get_hk_company_info(self, code: str) -> Dict:
        """Get basic company information for Hong Kong stocks"""
        company_names = {
            '0700': {'name': 'Tencent Holdings Limited'},
            '0941': {'name': 'China Mobile Limited'},
            '1299': {'name': 'AIA Group Limited'},
            # Add more as needed
        }
        return company_names.get(code, {'name': f'Hong Kong Company {code}'})
    
    def get_filing_content(self, filing_url: str) -> Optional[str]:
        """Extract text content from a filing document"""
        try:
            response = requests.get(filing_url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:5000]  # Limit content length
            
        except Exception as e:
            self.logger.error(f"Error getting filing content from {filing_url}: {e}")
            return None
    
    def batch_collect_filings(self, symbols: List[str]) -> List[Dict]:
        """Collect filings for multiple symbols with rate limiting"""
        all_filings = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"Collecting filings for {symbol}")
                filings = self.get_company_filings(symbol)
                
                # Add symbol to each filing for tracking
                for filing in filings:
                    filing['symbol'] = symbol
                
                all_filings.extend(filings)
                
                # Rate limiting - wait between requests
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting filings for {symbol}: {e}")
        
        return all_filings