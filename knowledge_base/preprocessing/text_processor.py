import re
import nltk
from typing import List, Dict, Tuple
import logging
from datetime import datetime
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class FinancialTextProcessor:
    """Specialized text processor for financial documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stop_words = set(stopwords.words('english'))
        
        # Financial-specific stop words to remove
        self.financial_stop_words = {
            'company', 'corporation', 'inc', 'ltd', 'llc', 'corp',
            'said', 'says', 'according', 'reported', 'announced'
        }
        
        # Important financial terms to preserve
        self.preserve_terms = {
            'eps', 'pe', 'roi', 'roa', 'roe', 'ebitda', 'capex',
            'q1', 'q2', 'q3', 'q4', 'fy', 'yoy', 'qoq'
        }
    
    def clean_financial_text(self, text: str) -> str:
        """Clean and normalize financial text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve financial notation
        text = re.sub(r'[^\w\s\$\%\.\,\-\(\)]', ' ', text)
        
        # Normalize financial numbers
        text = self._normalize_financial_numbers(text)
        
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities like stock symbols, percentages, etc."""
        entities = {
            'stock_symbols': [],
            'percentages': [],
            'dollar_amounts': [],
            'financial_ratios': [],
            'dates': []
        }
        
        # Stock symbols (3-5 uppercase letters)
        stock_pattern = r'\b[A-Z]{2,5}\b'
        entities['stock_symbols'] = re.findall(stock_pattern, text)
        
        # Percentages
        percentage_pattern = r'\d+\.?\d*\s*%'
        entities['percentages'] = re.findall(percentage_pattern, text)
        
        # Dollar amounts
        dollar_pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?'
        entities['dollar_amounts'] = re.findall(dollar_pattern, text, re.IGNORECASE)
        
        # Financial ratios (P/E, EPS, etc.)
        ratio_pattern = r'\b(?:P/E|EPS|ROE|ROA|EBITDA)\s*:?\s*\d+\.?\d*'
        entities['financial_ratios'] = re.findall(ratio_pattern, text, re.IGNORECASE)
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        return entities
    
    def chunk_document(self, 
                      text: str, 
                      chunk_size: int = 512, 
                      overlap: int = 50,
                      preserve_sentences: bool = True) -> List[Dict]:
        """Split document into chunks with metadata"""
        if not text:
            return []
        
        chunks = []
        
        if preserve_sentences:
            sentences = sent_tokenize(text)
            current_chunk = ""
            current_word_count = 0
            
            for sentence in sentences:
                sentence_words = len(word_tokenize(sentence))
                
                if current_word_count + sentence_words <= chunk_size:
                    current_chunk += sentence + " "
                    current_word_count += sentence_words
                else:
                    if current_chunk:
                        chunk_info = {
                            'text': current_chunk.strip(),
                            'word_count': current_word_count,
                            'chunk_index': len(chunks),
                            'entities': self.extract_financial_entities(current_chunk),
                            'timestamp': datetime.now().isoformat()
                        }
                        chunks.append(chunk_info)
                    
                    # Start new chunk with overlap
                    if overlap > 0 and current_chunk:
                        overlap_text = " ".join(current_chunk.split()[-overlap:])
                        current_chunk = overlap_text + " " + sentence + " "
                        current_word_count = len(word_tokenize(current_chunk))
                    else:
                        current_chunk = sentence + " "
                        current_word_count = sentence_words
            
            # Add final chunk
            if current_chunk.strip():
                chunk_info = {
                    'text': current_chunk.strip(),
                    'word_count': current_word_count,
                    'chunk_index': len(chunks),
                    'entities': self.extract_financial_entities(current_chunk),
                    'timestamp': datetime.now().isoformat()
                }
                chunks.append(chunk_info)
        
        else:
            # Simple word-based chunking
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunk_info = {
                    'text': chunk_text,
                    'word_count': len(chunk_words),
                    'chunk_index': len(chunks),
                    'entities': self.extract_financial_entities(chunk_text),
                    'timestamp': datetime.now().isoformat()
                }
                chunks.append(chunk_info)
        
        return chunks
    
    def _normalize_financial_numbers(self, text: str) -> str:
        """Normalize financial number representations"""
        # Convert different billion/million formats
        text = re.sub(r'(\d+\.?\d*)\s*billion', r'\1B', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*million', r'\1M', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*trillion', r'\1T', text, flags=re.IGNORECASE)
        
        # Normalize percentage formats
        text = re.sub(r'(\d+\.?\d*)\s*percent', r'\1%', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key financial phrases using simple frequency analysis"""
        # Clean text
        cleaned_text = self.clean_financial_text(text)
        words = word_tokenize(cleaned_text.lower())
        
        # Remove stop words but preserve important financial terms
        filtered_words = []
        for word in words:
            if word not in self.stop_words or word in self.preserve_terms:
                if len(word) > 2 and word not in self.financial_stop_words:
                    filtered_words.append(word)
        
        # Simple n-gram extraction (2-3 word phrases)
        phrases = []
        for i in range(len(filtered_words) - 1):
            # 2-gram
            phrase = " ".join(filtered_words[i:i+2])
            phrases.append(phrase)
            
            # 3-gram
            if i < len(filtered_words) - 2:
                phrase_3 = " ".join(filtered_words[i:i+3])
                phrases.append(phrase_3)
        
        # Count phrase frequencies
        from collections import Counter
        phrase_counts = Counter(phrases)
        
        # Return most common phrases
        return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]
