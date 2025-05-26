import PyPDF2
import requests
from typing import Dict, List, Optional
from io import BytesIO
import logging
from datetime import datetime

class PDFDocumentLoader:
    """Loads and processes PDF documents from various sources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.headers = {
            'User-Agent': 'Finance Assistant PDF Loader (educational@example.com)'
        }
    
    def load_pdf_from_url(self, url: str, metadata: Optional[Dict] = None) -> Dict:
        """Load PDF document from URL and extract text"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            pdf_file = BytesIO(response.content)
            text_content = self._extract_text_from_pdf(pdf_file)
            
            document = {
                'content': text_content,
                'source_url': url,
                'content_type': 'pdf_document',
                'source': 'pdf_loader',
                'timestamp': datetime.now().isoformat(),
                'word_count': len(text_content.split()),
                'metadata': metadata or {}
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error loading PDF from {url}: {e}")
            return {}
    
    def load_sec_filing_pdf(self, filing_url: str, company_info: Dict) -> Dict:
        """Specifically load SEC filing PDFs with enhanced metadata"""
        document = self.load_pdf_from_url(filing_url)
        
        if document:
            # Enhanced metadata for SEC filings
            document.update({
                'content_type': 'sec_filing_pdf',
                'company_cik': company_info.get('cik', ''),
                'company_name': company_info.get('name', ''),
                'filing_type': company_info.get('form_type', ''),
                'filing_date': company_info.get('filing_date', ''),
                'tags': ['sec_filing', 'regulatory', company_info.get('form_type', '').lower()]
            })
            
            # Extract key sections for better searchability
            document['sections'] = self._extract_filing_sections(document['content'])
        
        return document
    
    def _extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        """Extract text content from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content.append(page.extract_text())
            
            return '\n'.join(text_content)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_filing_sections(self, text: str) -> Dict:
        """Extract key sections from SEC filings"""
        sections = {}
        
        # Common SEC filing sections
        section_patterns = {
            'business_overview': r'item\s+1\.?\s*business',
            'risk_factors': r'item\s+1a\.?\s*risk\s+factors',
            'financial_statements': r'item\s+8\.?\s*financial\s+statements',
            'management_discussion': r'item\s+7\.?\s*management.?s\s+discussion',
            'controls_procedures': r'item\s+9a\.?\s*controls\s+and\s+procedures'
        }
        
        text_lower = text.lower()
        
        for section_name, pattern in section_patterns.items():
            import re
            matches = re.search(pattern, text_lower)
            if matches:
                start_pos = matches.start()
                # Extract approximately 1000 characters from the section
                section_text = text[start_pos:start_pos + 1000]
                sections[section_name] = section_text.strip()
        
        return sections

# File: data_ingestion/documen