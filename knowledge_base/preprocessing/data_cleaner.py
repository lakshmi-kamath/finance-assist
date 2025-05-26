import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re

class FinancialDataCleaner:
    """Cleans and validates financial data before storage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Data quality thresholds
        self.min_text_length = 50
        self.max_text_length = 50000
        self.required_fields = ['content_type', 'source', 'timestamp']
        
    def clean_document_batch(self, documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Clean a batch of documents and return cleaned docs + error log"""
        cleaned_documents = []
        errors = []
        
        for i, doc in enumerate(documents):
            try:
                cleaned_doc = self.clean_single_document(doc)
                if cleaned_doc:
                    cleaned_documents.append(cleaned_doc)
                else:
                    errors.append(f"Document {i}: Failed validation checks")
            except Exception as e:
                errors.append(f"Document {i}: Error during cleaning - {str(e)}")
                self.logger.error(f"Error cleaning document {i}: {e}")
        
        return cleaned_documents, errors
    
    def clean_single_document(self, document: Dict) -> Optional[Dict]:
        """Clean and validate a single document"""
        if not self._validate_required_fields(document):
            return None
        
        cleaned_doc = document.copy()
        
        # Clean text content
        if 'content' in cleaned_doc:
            cleaned_doc['content'] = self._clean_text_content(cleaned_doc['content'])
        
        if 'title' in cleaned_doc:
            cleaned_doc['title'] = self._clean_text_content(cleaned_doc['title'])
        
        if 'summary' in cleaned_doc:
            cleaned_doc['summary'] = self._clean_text_content(cleaned_doc['summary'])
        
        # Standardize timestamp format
        cleaned_doc['timestamp'] = self._standardize_timestamp(cleaned_doc['timestamp'])
        
        # Clean numerical fields
        cleaned_doc = self._clean_numerical_fields(cleaned_doc)
        
        # Validate content length
        if not self._validate_content_length(cleaned_doc):
            return None
        
        # Add data quality metrics
        cleaned_doc['data_quality'] = self._calculate_quality_score(cleaned_doc)
        
        # Remove duplicate or redundant fields
        cleaned_doc = self._remove_redundant_fields(cleaned_doc)
        
        return cleaned_doc
    
    def deduplicate_documents(self, documents: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """Remove duplicate documents based on content similarity"""
        if len(documents) <= 1:
            return documents
        
        unique_documents = []
        seen_hashes = set()
        
        for doc in documents:
            # Create content hash for exact duplicates
            content_hash = self._create_content_hash(doc)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_documents.append(doc)
            else:
                self.logger.info(f"Removed exact duplicate: {doc.get('title', 'Unknown')}")
        
        return unique_documents
    
    def _validate_required_fields(self, document: Dict) -> bool:
        """Validate that document has required fields"""
        for field in self.required_fields:
            if field not in document or not document[field]:
                self.logger.warning(f"Missing required field: {field}")
                return False
        return True
    
    def _clean_text_content(self, text: str) -> str:
        """Clean text content"""
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Clean up spaces around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        return text.strip()
    
    def _standardize_timestamp(self, timestamp: str) -> str:
        """Standardize timestamp format"""
        try:
            if isinstance(timestamp, str):
                # Try parsing common formats
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                    try:
                        dt = datetime.strptime(timestamp.split('.')[0], fmt)
                        return dt.isoformat()
                    except ValueError:
                        continue
                
                # If parsing fails, return current timestamp
                return datetime.now().isoformat()
            else:
                return datetime.now().isoformat()
        except Exception:
            return datetime.now().isoformat()
    
    def _clean_numerical_fields(self, document: Dict) -> Dict:
        """Clean and validate numerical fields"""
        numerical_fields = [
            'current_price', 'previous_close', 'market_cap', 'pe_ratio',
            'volume', 'reported_eps', 'estimated_eps', 'surprise_percentage'
        ]
        
        for field in numerical_fields:
            if field in document:
                try:
                    value = document[field]
                    if isinstance(value, str):
                        # Remove commas and dollar signs
                        value = re.sub(r'[$,]', '', value)
                        # Convert to float
                        document[field] = float(value) if value.replace('.', '').replace('-', '').isdigit() else 0.0
                    elif not isinstance(value, (int, float)):
                        document[field] = 0.0
                    else:
                        document[field] = float(value)
                except (ValueError, TypeError):
                    document[field] = 0.0
        
        return document
    
    def _validate_content_length(self, document: Dict) -> bool:
        """Validate content length is within acceptable range"""
        main_content = ""
        
        if 'content' in document:
            main_content = document['content']
        elif 'title' in document:
            main_content = document['title']
        elif 'summary' in document:
            main_content = document['summary']
        
        content_length = len(main_content)
        
        if content_length < self.min_text_length:
            self.logger.info(f"Document too short ({content_length} chars): {document.get('title', 'Unknown')}")
            return False
        
        if content_length > self.max_text_length:
            self.logger.info(f"Document too long ({content_length} chars), truncating: {document.get('title', 'Unknown')}")
            if 'content' in document:
                document['content'] = document['content'][:self.max_text_length]
        
        return True
    
    def _calculate_quality_score(self, document: Dict) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 0.0
        
        # Check for title
        if 'title' in document and len(document['title']) > 10:
            score += 0.2
        max_score += 0.2
        
        # Check for content quality
        if 'content' in document:
            content_len = len(document['content'])
            if content_len > 100:
                score += 0.3
            elif content_len > 50:
                score += 0.15
        max_score += 0.3
        
        # Check for metadata completeness
        metadata_fields = ['source', 'content_type', 'timestamp']
        present_fields = sum(1 for field in metadata_fields if field in document and document[field])
        score += (present_fields / len(metadata_fields)) * 0.2
        max_score += 0.2
        
        # Check for financial data
        financial_fields = ['symbol', 'current_price', 'market_cap']
        present_financial = sum(1 for field in financial_fields if field in document and document[field])
        if present_financial > 0:
            score += (present_financial / len(financial_fields)) * 0.15
        max_score += 0.15
        
        # Check for tags
        if 'tags' in document and isinstance(document['tags'], list) and len(document['tags']) > 0:
            score += 0.15
        max_score += 0.15
        
        return score / max_score if max_score > 0 else 0.0
    
    def _create_content_hash(self, document: Dict) -> str:
        """Create hash for duplicate detection"""
        import hashlib
        
        content_parts = []
        
        if 'title' in document:
            content_parts.append(document['title'])
        
        if 'content' in document:
            # Use first 500 characters for hashing
            content_parts.append(document['content'][:500])
        
        if 'symbol' in document:
            content_parts.append(str(document['symbol']))
        
        content_string = "|".join(content_parts)
        return hashlib.md5(content_string.encode()).hexdigest()
    
    def _remove_redundant_fields(self, document: Dict) -> Dict:
        """Remove redundant or empty fields"""
        redundant_fields = []
        
        for key, value in document.items():
            if value is None or value == "" or value == [] or value == {}:
                redundant_fields.append(key)
        
        for field in redundant_fields:
            if field not in self.required_fields:  # Don't remove required fields
                del document[field]
        
        return document