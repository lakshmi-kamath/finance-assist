import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional
import logging
from datetime import datetime

class SECFilingScraper:
    """Scrapes SEC filings for financial documents"""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov"
        self.headers = {
            'User-Agent': 'Finance Assistant Bot (educational@example.com)'
        }
        self.logger = logging.getLogger(__name__)
    
    def search_company_filings(self, company_cik: str, form_types: List[str] = ['10-K', '10-Q', '8-K']) -> List[Dict]:
        """Search for company filings by CIK"""
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
                                    'timestamp': datetime.now().isoformat()
                                })
                                
            except Exception as e:
                self.logger.error(f"Error searching {form_type} filings for CIK {company_cik}: {e}")
        
        return filings
