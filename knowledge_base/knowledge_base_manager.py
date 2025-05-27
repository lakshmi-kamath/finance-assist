from typing import List, Dict
import asyncio
import logging
import os
from datetime import datetime
import schedule
import time
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .text_chunker import DocumentChunker
from .data_ingestion_service import DataIngestionService
from .query_processor import QueryProcessor

class KnowledgeBaseManager:
    """Manages the knowledge base pipeline and updates."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embedding_service = EmbeddingService(config['vector_store']['embedding_model'])
        self.vector_store = VectorStore(
            dimension=config['vector_store']['dimension'],
            index_type=config['vector_store']['index_type']
        )
        self.chunker = DocumentChunker(
            chunk_size=config['processing']['chunk_size'],
            chunk_overlap=config['processing']['chunk_overlap']
        )
        self.data_ingestion = DataIngestionService(config)
        self.query_processor = QueryProcessor()
        
        self._ensure_directories_exist()
        self._load_vector_store()
        
        self.pipeline_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'documents_processed': 0,
            'last_run_timestamp': None,
            'data_source_stats': {}
        }
    def _is_duplicate(self, doc: Dict, existing_docs: List[Dict]) -> bool:
        """Check if a document is a duplicate based on key fields."""
        for existing in existing_docs:
            if (doc.get('title') == existing.get('title') and
                doc.get('symbol') == existing.get('symbol') and
                doc.get('content_type') == existing.get('content_type')):
                return True
        return False
    
    def _ensure_directories_exist(self):
        """Ensure required directories exist."""
        directories = ['knowledge_base', 'knowledge_base/vector_store', 'logs', 'data', 'temp']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_vector_store(self):
        """Load existing vector store if available."""
        try:
            index_path = self.config['vector_store']['index_path']
            metadata_path = self.config['vector_store']['metadata_path']
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.vector_store.load(index_path, metadata_path)
                self.logger.info("Loaded existing FAISS index")
            else:
                self.logger.info("Starting with empty FAISS index")
        except Exception as e:
            self.logger.warning(f"Could not load FAISS index: {e}")
    
    async def run_pipeline(self) -> Dict:
        """Run the full data ingestion and indexing pipeline."""
        self.pipeline_stats['total_runs'] += 1
        results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'started',
            'documents_added': 0,
            'errors': [],
            'data_sources': {}
        }
        
        try:
            all_documents = []
            for collect_method, source_name in [
                (self.data_ingestion.collect_economic_data, 'economic_indicators'),
                (self.data_ingestion.collect_market_data, 'market_data'),
                (self.data_ingestion.collect_news_data, 'news_articles'),
                (self.data_ingestion.collect_earnings_data, 'earnings_data'),
                (self.data_ingestion.collect_regulatory_filings, 'regulatory_filings')
            ]:
                try:
                    docs = await collect_method()
                    all_documents.extend(docs)
                    results['data_sources'][source_name] = len(docs)
                except Exception as e:
                    results['errors'].append(f"{source_name} error: {str(e)}")
                    self.logger.error(f"{source_name} collection failed: {e}")
            
            if all_documents:
                # Deduplicate documents
                unique_documents = []
                for doc in all_documents:
                    if not self._is_duplicate(doc, self.vector_store.documents):
                        unique_documents.append(doc)
                
                processed_docs = self.chunker.chunk_documents(unique_documents)
                texts = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in processed_docs]
                embeddings = self.embedding_service.generate_embeddings(texts)
                
                for doc, embedding in zip(processed_docs, embeddings):
                    doc['embedding'] = embedding.tolist()
                
                self.vector_store.add_documents(processed_docs)
                
                try:
                    self.vector_store.save(
                        self.config['vector_store']['index_path'],
                        self.config['vector_store']['metadata_path']
                    )
                    self.logger.info("Saved FAISS index")
                except Exception as e:
                    results['errors'].append(f"Index save error: {str(e)}")
                    self.logger.error(f"Failed to save FAISS index: {e}")
                
                results['documents_added'] = len(all_documents)
                results['chunks_created'] = len(processed_docs)
                results['vector_store_size'] = self.vector_store.get_document_count()
            
            for source, count in results['data_sources'].items():
                if source not in self.pipeline_stats['data_source_stats']:
                    self.pipeline_stats['data_source_stats'][source] = {'total': 0, 'runs': 0}
                self.pipeline_stats['data_source_stats'][source]['total'] += count
                self.pipeline_stats['data_source_stats'][source]['runs'] += 1
            
            results['status'] = 'completed' if not results['errors'] else 'completed_with_errors'
            self.pipeline_stats['successful_runs' if not results['errors'] else 'failed_runs'] += 1
            self.pipeline_stats['documents_processed'] += len(all_documents)
        
        except Exception as e:
            results['status'] = 'failed'
            results['errors'].append(f"Pipeline critical error: {str(e)}")
            self.pipeline_stats['failed_runs'] += 1
            self.logger.error(f"Pipeline critical failure: {e}")
        
        self.pipeline_stats['last_run_timestamp'] = results['timestamp']
        return results
    
    async def search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search with enhanced query."""
        enhanced_query = self.query_processor.enhance_query(query)
        query_embedding = self.embedding_service.generate_query_embedding(enhanced_query)
        results = self.vector_store.search(query_embedding, k)
        return results
    
    def schedule_updates(self):
        """Schedule regular pipeline updates."""
        schedule.every(4).hours.do(lambda: asyncio.run(self.run_pipeline()))
        schedule.every().day.at("07:00").do(lambda: asyncio.run(self.run_pipeline()))
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        success_rate = (self.pipeline_stats['successful_runs'] / max(self.pipeline_stats['total_runs'], 1)) * 100
        return {
            'total_runs': self.pipeline_stats['total_runs'],
            'successful_runs': self.pipeline_stats['successful_runs'],
            'failed_runs': self.pipeline_stats['failed_runs'],
            'success_rate_percent': round(success_rate, 2),
            'total_documents_processed': self.pipeline_stats['documents_processed'],
            'last_run_timestamp': self.pipeline_stats['last_run_timestamp'],
            'data_source_statistics': self.pipeline_stats['data_source_stats']
        }
    
    def check_health(self) -> Dict:
        """Check vector store health."""
        try:
            doc_count = self.vector_store.get_document_count()
            test_query = "earnings report technology"
            test_embedding = self.embedding_service.generate_query_embedding(test_query)
            test_results = self.vector_store.search(test_embedding, k=3)
            return {
                'status': 'healthy',
                'document_count': doc_count,
                'test_search_results': len(test_results),
                'index_type': self.vector_store.index_type,
                'dimension': self.vector_store.dimension
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'document_count': 0
            }