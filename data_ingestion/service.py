import os
import sys
import pickle
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from pathlib import Path
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Install with: pip install faiss-cpu sentence-transformers numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBIngestionManager:
    """Manages ingestion of API and scraper data into the vector database with duplicate detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.vector_db_path = config.get('vector_db_path', 'knowledge_base/vector_store/faiss.index')
        self.metadata_path = config.get('chunk_index_path', 'knowledge_base/vector_store/metadata.pkl')
        self.duplicate_hashes_path = config.get('duplicate_hashes_path', 'knowledge_base/vector_store/content_hashes.pkl')
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Load existing data
        self.index, self.metadata, self.content_hashes = self._load_existing_data()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'duplicates_skipped': 0,
            'new_chunks_added': 0,
            'api_chunks_added': 0,
            'scraper_chunks_added': 0,
            'errors': 0
        }
    
    def _load_existing_data(self) -> Tuple[faiss.Index, List[Dict], Set[str]]:
        """Load existing FAISS index, metadata, and content hashes"""
        # Load FAISS index
        if os.path.exists(self.vector_db_path):
            try:
                index = faiss.read_index(self.vector_db_path)
                logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}. Creating new index.")
                index = faiss.IndexFlatIP(self.embedding_dimension)
        else:
            logger.info("Creating new FAISS index")
            index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded {len(metadata)} metadata entries")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}. Starting with empty metadata.")
                metadata = []
        else:
            metadata = []
        
        # Load content hashes for duplicate detection
        if os.path.exists(self.duplicate_hashes_path):
            try:
                with open(self.duplicate_hashes_path, 'rb') as f:
                    content_hashes = pickle.load(f)
                logger.info(f"Loaded {len(content_hashes)} content hashes for duplicate detection")
            except Exception as e:
                logger.warning(f"Could not load content hashes: {e}. Starting with empty set.")
                content_hashes = set()
        else:
            content_hashes = set()
        
        return index, metadata, content_hashes
    
    def _generate_content_hash(self, content: str, source: str, timestamp: str) -> str:
        """Generate a hash for content to detect duplicates"""
        # Normalize content for consistent hashing
        normalized_content = content.strip().lower()
        hash_input = f"{normalized_content}|{source}|{timestamp[:10]}"  # Use date only, not full timestamp
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if content hash already exists"""
        return content_hash in self.content_hashes
    
    def _chunk_content(self, content: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + max_chunk_size
            
            # Try to end at a sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                sentence_ends = ['.', '!', '?', '\n']
                for i in range(min(100, end - start)):
                    char_idx = end - 1 - i
                    if content[char_idx] in sentence_ends and char_idx > start + max_chunk_size // 2:
                        end = char_idx + 1
                        break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + max_chunk_size - overlap, end)
            if start >= len(content):
                break
        
        return chunks
    
    def _extract_api_content(self, api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and structure content from API data"""
        extracted_content = []
        
        for task_name, task_data in api_data.items():
            try:
                data = task_data.get('data', {})
                metadata = task_data.get('metadata', {})
                
                # Extract different types of API data
                if 'company_overview' in task_name.lower():
                    content = self._format_company_overview(data)
                elif 'income_statement' in task_name.lower():
                    content = self._format_financial_statement(data, 'Income Statement')
                elif 'balance_sheet' in task_name.lower():
                    content = self._format_financial_statement(data, 'Balance Sheet')
                elif 'cash_flow' in task_name.lower():
                    content = self._format_financial_statement(data, 'Cash Flow')
                elif 'earnings' in task_name.lower():
                    content = self._format_earnings_data(data)
                elif 'time_series' in task_name.lower():
                    content = self._format_time_series_data(data)
                else:
                    # Generic formatting for other API data
                    content = self._format_generic_api_data(data, task_name)
                
                if content:
                    extracted_content.append({
                        'content': content,
                        'source': f"API_{task_name}",
                        'data_type': 'api',
                        'task_name': task_name,
                        'metadata': metadata,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error extracting API content from {task_name}: {e}")
                self.stats['errors'] += 1
        
        return extracted_content
    
    def _extract_scraper_content(self, scraper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and structure content from scraper data"""
        extracted_content = []
        
        for task_name, task_data in scraper_data.items():
            try:
                data = task_data.get('data', {})
                metadata = task_data.get('metadata', {})
                
                if isinstance(data, dict):
                    # Handle news articles
                    if 'articles' in data or 'news' in task_name.lower():
                        articles = data.get('articles', data.get('news', []))
                        for article in articles:
                            if isinstance(article, dict):
                                content = self._format_news_article(article)
                                if content:
                                    extracted_content.append({
                                        'content': content,
                                        'source': f"SCRAPER_{task_name}",
                                        'data_type': 'scraper',
                                        'task_name': task_name,
                                        'url': article.get('url', ''),
                                        'metadata': metadata,
                                        'timestamp': article.get('published_date', datetime.now(timezone.utc).isoformat())
                                    })
                    
                    # Handle SEC filings
                    elif 'sec' in task_name.lower() or 'filings' in task_name.lower():
                        if 'filings' in data:
                            for filing in data['filings']:
                                content = self._format_sec_filing(filing)
                                if content:
                                    extracted_content.append({
                                        'content': content,
                                        'source': f"SCRAPER_SEC_{filing.get('form_type', 'UNKNOWN')}",
                                        'data_type': 'scraper',
                                        'task_name': task_name,
                                        'filing_url': filing.get('filing_url', ''),
                                        'metadata': metadata,
                                        'timestamp': filing.get('filing_date', datetime.now(timezone.utc).isoformat())
                                    })
                    
                    # Handle generic scraped content
                    else:
                        content = self._format_generic_scraper_data(data, task_name)
                        if content:
                            extracted_content.append({
                                'content': content,
                                'source': f"SCRAPER_{task_name}",
                                'data_type': 'scraper',
                                'task_name': task_name,
                                'metadata': metadata,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                
                elif isinstance(data, list):
                    # Handle list of scraped items
                    for idx, item in enumerate(data):
                        content = self._format_generic_scraper_data(item, f"{task_name}_{idx}")
                        if content:
                            extracted_content.append({
                                'content': content,
                                'source': f"SCRAPER_{task_name}_{idx}",
                                'data_type': 'scraper',
                                'task_name': task_name,
                                'metadata': metadata,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                            
            except Exception as e:
                logger.error(f"Error extracting scraper content from {task_name}: {e}")
                self.stats['errors'] += 1
        
        return extracted_content
    
    def _format_company_overview(self, data: Dict[str, Any]) -> str:
        """Format company overview data"""
        if not data:
            return ""
        
        content_parts = []
        
        # Basic company info
        if 'Name' in data:
            content_parts.append(f"Company: {data['Name']}")
        if 'Symbol' in data:
            content_parts.append(f"Symbol: {data['Symbol']}")
        if 'Description' in data:
            content_parts.append(f"Description: {data['Description']}")
        
        # Financial metrics
        financial_fields = ['MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 
                          'BookValue', 'DividendPerShare', 'DividendYield', 'EPS']
        for field in financial_fields:
            if field in data and data[field] not in ['None', 'N/A', '']:
                content_parts.append(f"{field}: {data[field]}")
        
        # Industry and sector
        if 'Industry' in data:
            content_parts.append(f"Industry: {data['Industry']}")
        if 'Sector' in data:
            content_parts.append(f"Sector: {data['Sector']}")
        
        return "\n".join(content_parts)
    
    def _format_financial_statement(self, data: Dict[str, Any], statement_type: str) -> str:
        """Format financial statement data"""
        if not data:
            return ""
        
        content_parts = [f"{statement_type} Data:"]
        
        # Handle annual and quarterly reports
        for report_type in ['annualReports', 'quarterlyReports']:
            if report_type in data:
                reports = data[report_type][:3]  # Latest 3 reports
                for report in reports:
                    if isinstance(report, dict):
                        fiscal_date = report.get('fiscalDateEnding', 'Unknown')
                        content_parts.append(f"\n{report_type.replace('Reports', '')} - {fiscal_date}:")
                        
                        # Add key financial metrics
                        for key, value in report.items():
                            if key != 'fiscalDateEnding' and value not in ['None', 'N/A', '']:
                                content_parts.append(f"  {key}: {value}")
        
        return "\n".join(content_parts)
    
    def _format_earnings_data(self, data: Dict[str, Any]) -> str:
        """Format earnings data"""
        if not data:
            return ""
        
        content_parts = ["Earnings Data:"]
        
        if 'quarterlyEarnings' in data:
            earnings = data['quarterlyEarnings'][:4]  # Latest 4 quarters
            for earning in earnings:
                if isinstance(earning, dict):
                    fiscal_date = earning.get('fiscalDateEnding', 'Unknown')
                    reported_eps = earning.get('reportedEPS', 'N/A')
                    estimated_eps = earning.get('estimatedEPS', 'N/A')
                    surprise = earning.get('surprise', 'N/A')
                    
                    content_parts.append(f"\nQuarter ending {fiscal_date}:")
                    content_parts.append(f"  Reported EPS: {reported_eps}")
                    content_parts.append(f"  Estimated EPS: {estimated_eps}")
                    content_parts.append(f"  Surprise: {surprise}")
        
        return "\n".join(content_parts)
    
    def _format_time_series_data(self, data: Dict[str, Any]) -> str:
        """Format time series data"""
        if not data:
            return ""
        
        content_parts = ["Time Series Data:"]
        
        # Handle different time series formats
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'daily' in key.lower() or 'monthly' in key.lower():
                time_series_key = key
                break
        
        if time_series_key and isinstance(data[time_series_key], dict):
            # Get recent data points (latest 5)
            recent_dates = sorted(data[time_series_key].keys(), reverse=True)[:5]
            
            for date in recent_dates:
                date_data = data[time_series_key][date]
                content_parts.append(f"\n{date}:")
                for metric, value in date_data.items():
                    content_parts.append(f"  {metric}: {value}")
        
        return "\n".join(content_parts)
    
    def _format_news_article(self, article: Dict[str, Any]) -> str:
        """Format news article data"""
        content_parts = []
        
        if 'title' in article:
            content_parts.append(f"Title: {article['title']}")
        if 'summary' in article:
            content_parts.append(f"Summary: {article['summary']}")
        elif 'description' in article:
            content_parts.append(f"Description: {article['description']}")
        
        if 'content' in article and article['content']:
            content_parts.append(f"Content: {article['content']}")
        
        if 'source' in article:
            content_parts.append(f"Source: {article['source']}")
        
        if 'published_date' in article:
            content_parts.append(f"Published: {article['published_date']}")
        
        return "\n".join(content_parts)
    
    def _format_sec_filing(self, filing: Dict[str, Any]) -> str:
        """Format SEC filing data"""
        content_parts = []
        
        if 'form_type' in filing:
            content_parts.append(f"Form Type: {filing['form_type']}")
        if 'filing_date' in filing:
            content_parts.append(f"Filing Date: {filing['filing_date']}")
        if 'company_name' in filing:
            content_parts.append(f"Company: {filing['company_name']}")
        if 'description' in filing:
            content_parts.append(f"Description: {filing['description']}")
        if 'content' in filing and filing['content']:
            content_parts.append(f"Content: {filing['content']}")
        
        return "\n".join(content_parts)
    
    def _format_generic_api_data(self, data: Any, task_name: str) -> str:
        """Format generic API data"""
        if isinstance(data, dict):
            content_parts = [f"API Data from {task_name}:"]
            for key, value in data.items():
                if isinstance(value, (str, int, float)) and str(value) not in ['None', 'N/A', '']:
                    content_parts.append(f"{key}: {value}")
            return "\n".join(content_parts)
        elif isinstance(data, str):
            return f"API Data from {task_name}: {data}"
        else:
            return f"API Data from {task_name}: {str(data)}"
    
    def _format_generic_scraper_data(self, data: Any, task_name: str) -> str:
        """Format generic scraper data"""
        if isinstance(data, dict):
            content_parts = [f"Scraped Data from {task_name}:"]
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 0:
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, (int, float, bool)):
                    content_parts.append(f"{key}: {value}")
            return "\n".join(content_parts)
        elif isinstance(data, str):
            return f"Scraped Data from {task_name}: {data}"
        else:
            return f"Scraped Data from {task_name}: {str(data)}"
    
    async def ingest_orchestrator_results(self, orchestrator_results: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to ingest results from orchestrator agent"""
        logger.info("Starting ingestion of orchestrator results")
        
        # Extract API data
        api_data = orchestrator_results.get('api_data', {})
        if api_data:
            logger.info(f"Processing {len(api_data)} API data sources")
            api_content = self._extract_api_content(api_data)
            await self._process_content_batch(api_content, 'api')
        
        # Extract scraper data
        scraper_data = orchestrator_results.get('scraping_data', {})
        if scraper_data:
            logger.info(f"Processing {len(scraper_data)} scraper data sources")
            scraper_content = self._extract_scraper_content(scraper_data)
            await self._process_content_batch(scraper_content, 'scraper')
        
        # Save updated data
        self._save_data()
        
        # Return ingestion statistics
        return {
            'success': True,
            'statistics': self.stats,
            'total_vectors_in_db': self.index.ntotal,
            'total_metadata_entries': len(self.metadata),
            'total_content_hashes': len(self.content_hashes)
        }
    
    async def _process_content_batch(self, content_list: List[Dict[str, Any]], data_type: str):
        """Process a batch of content for ingestion"""
        for content_item in content_list:
            try:
                await self._process_single_content(content_item, data_type)
            except Exception as e:
                logger.error(f"Error processing {data_type} content: {e}")
                self.stats['errors'] += 1
    
    async def _process_single_content(self, content_item: Dict[str, Any], data_type: str):
        """Process a single content item"""
        content = content_item['content']
        source = content_item['source']
        timestamp = content_item['timestamp']
        
        # Generate content hash for duplicate detection
        content_hash = self._generate_content_hash(content, source, timestamp)
        
        # Check for duplicates
        if self._is_duplicate(content_hash):
            logger.debug(f"Skipping duplicate content from {source}")
            self.stats['duplicates_skipped'] += 1
            return
        
        # Chunk the content
        chunks = self._chunk_content(content)
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            try:
                # Generate embedding
                embedding = self.embedding_model.encode([chunk])
                embedding = embedding.astype(np.float32)
                
                # Normalize for cosine similarity (FAISS IndexFlatIP expects normalized vectors)
                faiss.normalize_L2(embedding)
                
                # Add to FAISS index
                self.index.add(embedding)
                
                # Create metadata
                chunk_metadata = {
                    'content': chunk,
                    'source': source,
                    'data_type': data_type,
                    'timestamp': timestamp,
                    'content_hash': content_hash,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'task_name': content_item.get('task_name', ''),
                    'url': content_item.get('url', ''),
                    'filing_url': content_item.get('filing_url', ''),
                    'metadata': content_item.get('metadata', {}),
                    'ingestion_timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Add to metadata list
                self.metadata.append(chunk_metadata)
                
                # Add content hash to set
                self.content_hashes.add(content_hash)
                
                # Update statistics
                self.stats['new_chunks_added'] += 1
                if data_type == 'api':
                    self.stats['api_chunks_added'] += 1
                elif data_type == 'scraper':
                    self.stats['scraper_chunks_added'] += 1
                
                logger.debug(f"Added chunk {chunk_idx + 1}/{len(chunks)} from {source}")
                
            except Exception as e:
                logger.error(f"Error processing chunk from {source}: {e}")
                self.stats['errors'] += 1
        
        self.stats['total_processed'] += 1
    
    def _save_data(self):
        """Save updated FAISS index, metadata, and content hashes"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.vector_db_path)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {self.vector_db_path}")
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved {len(self.metadata)} metadata entries to {self.metadata_path}")
            
            # Save content hashes
            with open(self.duplicate_hashes_path, 'wb') as f:
                pickle.dump(self.content_hashes, f)
            logger.info(f"Saved {len(self.content_hashes)} content hashes to {self.duplicate_hashes_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current ingestion statistics"""
        return {
            **self.stats,
            'total_vectors_in_db': self.index.ntotal,
            'total_metadata_entries': len(self.metadata),
            'total_content_hashes': len(self.content_hashes),
            'embedding_model': self.embedding_model_name,
            'embedding_dimension': self.embedding_dimension
        }


# Utility functions
async def ingest_orchestrator_results(orchestrator_results: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Utility function to ingest orchestrator results"""
    default_config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_db_path': 'knowledge_base/vector_store/faiss.index',
        'chunk_index_path': 'knowledge_base/vector_store/metadata.pkl',
        'duplicate_hashes_path': 'knowledge_base/vector_store/content_hashes.pkl'
    }
    
    if config:
        default_config.update(config)
    
    ingestion_manager = VectorDBIngestionManager(default_config)
    return await ingestion_manager.ingest_orchestrator_results(orchestrator_results)


def create_enhanced_orchestrator_node(original_node_func):
    """Decorator to enhance orchestrator nodes with automatic ingestion"""
    async def enhanced_node(state):
        # Execute original node
        result_state = await original_node_func(state)
        
        # Check if this is the final combine_results node
        node_name = original_node_func.__name__
        if 'combine_results' in node_name or 'final' in node_name:
            try:
                # Ingest the results into vector DB
                final_results = result_state.get('final_results', {})
                if final_results:
                    ingestion_config = {
                        'vector_db_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/faiss.index',
                        'chunk_index_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/metadata.pkl',
                        'duplicate_hashes_path': '/Users/lakshmikamath/Desktop/finance-assist/knowledge_base/vector_store/content_hashes.pkl'
                    }
                    
                    ingestion_result = await ingest_orchestrator_results(final_results, ingestion_config)
                    
                    # Add ingestion stats to the final results
                    if 'metadata' not in result_state:
                        result_state['metadata'] = {}
                    result_state['metadata']['ingestion_stats'] = ingestion_result
                    
                    logger.info(f"Successfully ingested orchestrator results: {ingestion_result['statistics']}")
            except Exception as e:
                logger.error(f"Error during automatic ingestion: {e}")
                # Don't fail the node, just log the error
        
        return result_state
    
    return enhanced_node

