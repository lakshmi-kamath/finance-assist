import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
os.makedirs('logs', exist_ok=True)

# Initialize logging for console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/embedding_agent.log')
    ]
)
import numpy as np
import json
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path

from agents.base_agent import BaseAgent, Task, AgentResult, TaskPriority

# For text processing and embeddings
import tiktoken
from sentence_transformers import SentenceTransformer
import faiss

# For content chunking
import re
from collections import defaultdict

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    source: str = ""
    doc_type: str = ""
    relevance_score: float = 1.0
    freshness_score: float = 1.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class EmbeddingAgent(BaseAgent):
    """Agent responsible for creating and managing document embeddings"""
    
    def __init__(self, agent_id: str = "embedding_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Model configuration
        self.model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
        self.max_tokens = self.config.get('max_tokens', 8192)
        
        # Storage paths
        self.storage_path = Path(self.config.get('storage_path', './data/embeddings'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
        self.tokenizer = None
        self.vector_store = None
        self.chunk_cache = {}
        
        self._initialize_models()
    
    def _define_capabilities(self) -> List[str]:
        """Define what this agent can do"""
        return [
            'create_embeddings',
            'process_documents',
            'chunk_text',
            'update_vector_store',
            'get_similar_chunks',
            'calculate_freshness_score',
            'calculate_relevance_score',
            'clean_old_embeddings',
            'get_embedding_stats',
            'rebuild_index'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['sentence_transformers', 'faiss', 'tiktoken']
    
    def _initialize_models(self):
        """Initialize embedding model and tokenizer"""
        try:
            # Initialize sentence transformer
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
            
            # Initialize tokenizer for token counting
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Load existing vector store if available
            self._load_vector_store()
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_vector_store(self):
        """Load existing FAISS index and metadata"""
        index_path = "knowledge_base/vector_store/faiss.index"
        metadata_path = "knowledge_base/vector_store/metadata.pkl"
        
        try:
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.vector_store = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.chunk_cache = pickle.load(f)
                
                self.logger.info(f"Loaded vector store with {self.vector_store.ntotal} embeddings")
            else:
                # Create new empty index
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.vector_store = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
                self.chunk_cache = {}
                self.logger.info("Created new empty vector store")
                
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            # Fallback to empty store
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.vector_store = faiss.IndexFlatIP(embedding_dim)
            self.chunk_cache = {}
    
    def _save_vector_store(self):
        """Save FAISS index and metadata"""
        try:
            index_path = "knowledge_base/vector_store/faiss.index"
            metadata_path = "knowledge_base/vector_store/metadata.pkl"
            
            # Save FAISS index
            faiss.write_index(self.vector_store, str(index_path))
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunk_cache, f)
            
            self.logger.info(f"Saved vector store with {self.vector_store.ntotal} embeddings")
            
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
    
    async def execute_task(self, task: Task) -> AgentResult:
        """Execute embedding-related tasks"""
        task_type = task.type
        parameters = task.parameters
        
        try:
            if task_type == 'create_embeddings':
                return await self._create_embeddings(parameters)
            
            elif task_type == 'process_documents':
                return await self._process_documents(parameters)
            
            elif task_type == 'chunk_text':
                return await self._chunk_text(parameters)
            
            elif task_type == 'update_vector_store':
                return await self._update_vector_store(parameters)
            
            elif task_type == 'get_similar_chunks':
                return await self._get_similar_chunks(parameters)
            
            elif task_type == 'calculate_freshness_score':
                return await self._calculate_freshness_score(parameters)
            
            elif task_type == 'calculate_relevance_score':
                return await self._calculate_relevance_score(parameters)
            
            elif task_type == 'clean_old_embeddings':
                return await self._clean_old_embeddings(parameters)
            
            elif task_type == 'get_embedding_stats':
                return await self._get_embedding_stats(parameters)
            
            elif task_type == 'rebuild_index':
                return await self._rebuild_index(parameters)
            
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown task type: {task_type}"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing task {task_type}: {e}")
            return AgentResult(
                success=False,
                error=str(e)
            )
    
    async def _create_embeddings(self, parameters: Dict[str, Any]) -> AgentResult:
        """Create embeddings for text chunks"""
        texts = parameters.get('texts', [])
        batch_size = parameters.get('batch_size', 32)
        
        if not texts:
            return AgentResult(success=False, error="No texts provided")
        
        try:
            embeddings = []
            
            # Process in batches for efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    normalize_embeddings=True,  # For cosine similarity
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            return AgentResult(
                success=True,
                data=embeddings,
                metadata={
                    'texts_processed': len(texts),
                    'embedding_dimension': len(embeddings[0]) if embeddings else 0,
                    'batch_size': batch_size
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _process_documents(self, parameters: Dict[str, Any]) -> AgentResult:
        """Process raw documents into chunks with embeddings"""
        documents = parameters.get('documents', [])
        doc_type = parameters.get('doc_type', 'general')
        source = parameters.get('source', 'unknown')
        
        if not documents:
            return AgentResult(success=False, error="No documents provided")
        
        processed_chunks = []
        
        for doc_idx, document in enumerate(documents):
            try:
                # Extract text content
                if isinstance(document, dict):
                    text_content = document.get('content', document.get('text', ''))
                    doc_metadata = document.get('metadata', {})
                    doc_source = document.get('source', source)
                else:
                    text_content = str(document)
                    doc_metadata = {}
                    doc_source = source
                
                if not text_content.strip():
                    continue
                
                # Chunk the text
                chunks = self._smart_chunk_text(text_content)
                
                # Create embeddings for chunks
                chunk_texts = [chunk['content'] for chunk in chunks]
                embeddings_result = await self._create_embeddings({'texts': chunk_texts})
                
                if not embeddings_result.success:
                    self.logger.error(f"Failed to create embeddings for document {doc_idx}")
                    continue
                
                embeddings = embeddings_result.data
                
                # Create DocumentChunk objects
                for chunk_idx, (chunk_info, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_id = self._generate_chunk_id(text_content, chunk_idx, doc_source)
                    
                    # Calculate scores
                    freshness_score = self._calculate_freshness_score_value(doc_metadata)
                    relevance_score = self._calculate_relevance_score_value(chunk_info['content'], doc_type)
                    
                    doc_chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_info['content'],
                        embedding=embedding,
                        metadata={
                            **doc_metadata,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'doc_index': doc_idx,
                            'token_count': chunk_info['token_count'],
                            'start_pos': chunk_info['start_pos'],
                            'end_pos': chunk_info['end_pos']
                        },
                        source=doc_source,
                        doc_type=doc_type,
                        relevance_score=relevance_score,
                        freshness_score=freshness_score
                    )
                    
                    processed_chunks.append(doc_chunk)
                
            except Exception as e:
                self.logger.error(f"Error processing document {doc_idx}: {e}")
                continue
        
        return AgentResult(
            success=True,
            data=processed_chunks,
            metadata={
                'documents_processed': len(documents),
                'chunks_created': len(processed_chunks),
                'doc_type': doc_type,
                'source': source
            }
        )
    
    def _smart_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Intelligently chunk text based on content structure"""
        chunks = []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try to split by natural boundaries (paragraphs, sections)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.tokenizer.encode(potential_chunk))
            
            if token_count <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append({
                        'content': current_chunk,
                        'token_count': len(self.tokenizer.encode(current_chunk)),
                        'start_pos': current_start,
                        'end_pos': current_start + len(current_chunk)
                    })
                    current_start += len(current_chunk)
                
                # Handle oversized paragraphs
                if len(self.tokenizer.encode(paragraph)) > self.chunk_size:
                    # Split oversized paragraph by sentences
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        temp_potential = temp_chunk + ". " + sentence if temp_chunk else sentence
                        if len(self.tokenizer.encode(temp_potential)) <= self.chunk_size:
                            temp_chunk = temp_potential
                        else:
                            if temp_chunk:
                                chunks.append({
                                    'content': temp_chunk,
                                    'token_count': len(self.tokenizer.encode(temp_chunk)),
                                    'start_pos': current_start,
                                    'end_pos': current_start + len(temp_chunk)
                                })
                                current_start += len(temp_chunk)
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk,
                'token_count': len(self.tokenizer.encode(current_chunk)),
                'start_pos': current_start,
                'end_pos': current_start + len(current_chunk)
            })
        
        return chunks
    
    def _generate_chunk_id(self, content: str, chunk_idx: int, source: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{source}_{timestamp}_{content_hash}_{chunk_idx}"
    
    def _calculate_freshness_score_value(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on timestamp"""
        try:
            # Try to extract timestamp from metadata
            timestamp_str = metadata.get('timestamp', metadata.get('date', metadata.get('published_at')))
            
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    # Parse timestamp string
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now() - timedelta(days=30)  # Default to old
                else:
                    timestamp = timestamp_str
            else:
                timestamp = datetime.now()  # Assume fresh if no timestamp
            
            # Calculate age in hours
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            
            # Freshness score: 1.0 for <1 hour, decay exponentially
            if age_hours < 1:
                return 1.0
            elif age_hours < 24:
                return 0.8
            elif age_hours < 168:  # 1 week
                return 0.6
            elif age_hours < 720:  # 1 month
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            self.logger.debug(f"Error calculating freshness score: {e}")
            return 0.5  # Default score
    
    def _calculate_relevance_score_value(self, content: str, doc_type: str) -> float:
        """Calculate relevance score based on content and type"""
        try:
            score = 0.5  # Base score
            
            # Financial keywords boost
            financial_keywords = [
                'earnings', 'revenue', 'profit', 'loss', 'dividend', 'stock', 'market',
                'investment', 'portfolio', 'risk', 'return', 'volatility', 'trading',
                'financial', 'economic', 'gdp', 'inflation', 'interest', 'fed'
            ]
            
            content_lower = content.lower()
            keyword_matches = sum(1 for keyword in financial_keywords if keyword in content_lower)
            keyword_boost = min(keyword_matches * 0.1, 0.4)  # Max 0.4 boost
            
            # Document type boost
            type_boosts = {
                'earnings_report': 0.9,
                'sec_filing': 0.8,
                'financial_news': 0.7,
                'market_data': 0.8,
                'economic_indicator': 0.8,
                'general': 0.5
            }
            
            base_score = type_boosts.get(doc_type, 0.5)
            final_score = min(base_score + keyword_boost, 1.0)
            
            return final_score
            
        except Exception as e:
            self.logger.debug(f"Error calculating relevance score: {e}")
            return 0.5  # Default score
    
    async def _update_vector_store(self, parameters: Dict[str, Any]) -> AgentResult:
        """Update vector store with new chunks"""
        chunks = parameters.get('chunks', [])
        
        if not chunks:
            return AgentResult(success=False, error="No chunks provided")
        
        try:
            added_count = 0
            
            for chunk in chunks:
                if isinstance(chunk, dict):
                    # Convert dict to DocumentChunk
                    chunk = DocumentChunk(**chunk)
                
                if chunk.embedding is None:
                    # Create embedding if not present
                    embedding_result = await self._create_embeddings({'texts': [chunk.content]})
                    if embedding_result.success:
                        chunk.embedding = embedding_result.data[0]
                    else:
                        continue
                
                # Add to FAISS index
                embedding_vector = np.array([chunk.embedding], dtype=np.float32)
                self.vector_store.add(embedding_vector)
                
                # Store metadata
                index_id = self.vector_store.ntotal - 1
                self.chunk_cache[index_id] = chunk
                
                added_count += 1
            
            # Save the updated store
            self._save_vector_store()
            
            return AgentResult(
                success=True,
                data={'chunks_added': added_count},
                metadata={
                    'total_embeddings': self.vector_store.ntotal,
                    'chunks_added': added_count
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _get_similar_chunks(self, parameters: Dict[str, Any]) -> AgentResult:
        """Find similar chunks for a query"""
        query = parameters.get('query', '')
        top_k = parameters.get('top_k', 5)
        min_score = parameters.get('min_score', 0.3)
        include_metadata = parameters.get('include_metadata', True)
        
        if not query:
            return AgentResult(success=False, error="No query provided")
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)[0]
            
            # Search vector store
            search_k = min(top_k * 2, self.vector_store.ntotal)  # Search more, filter later
            if search_k == 0:
                return AgentResult(
                    success=True,
                    data=[],
                    metadata={'message': 'No embeddings in vector store'}
                )
            
            scores, indices = self.vector_store.search(
                np.array([query_embedding], dtype=np.float32),
                search_k
            )
            
            # Filter and format results
            similar_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < min_score:
                    continue
                
                chunk = self.chunk_cache.get(idx)
                if chunk is None:
                    continue
                
                result_item = {
                    'content': chunk.content,
                    'similarity_score': float(score),
                    'chunk_id': chunk.chunk_id,
                    'source': chunk.source,
                    'doc_type': chunk.doc_type,
                    'freshness_score': chunk.freshness_score,
                    'relevance_score': chunk.relevance_score,
                    'timestamp': chunk.timestamp.isoformat()
                }
                
                if include_metadata:
                    result_item['metadata'] = chunk.metadata
                
                similar_chunks.append(result_item)
            
            # Sort by combined score and limit
            for item in similar_chunks:
                # Combined score: similarity * relevance * freshness
                item['combined_score'] = (
                    item['similarity_score'] * 
                    item['relevance_score'] * 
                    (0.5 + 0.5 * item['freshness_score'])  # Freshness contributes 50%
                )
            
            similar_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
            similar_chunks = similar_chunks[:top_k]
            
            return AgentResult(
                success=True,
                data=similar_chunks,
                metadata={
                    'query': query,
                    'total_searched': search_k,
                    'results_returned': len(similar_chunks),
                    'min_score': min_score,
                    'max_similarity_score': max(item['similarity_score'] for item in similar_chunks) if similar_chunks else 0,
                    'max_combined_score': max(item['combined_score'] for item in similar_chunks) if similar_chunks else 0
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _calculate_freshness_score(self, parameters: Dict[str, Any]) -> AgentResult:
        """Calculate freshness score for metadata"""
        metadata = parameters.get('metadata', {})
        
        try:
            score = self._calculate_freshness_score_value(metadata)
            return AgentResult(
                success=True,
                data={'freshness_score': score},
                metadata=metadata
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _calculate_relevance_score(self, parameters: Dict[str, Any]) -> AgentResult:
        """Calculate relevance score for content"""
        content = parameters.get('content', '')
        doc_type = parameters.get('doc_type', 'general')
        
        try:
            score = self._calculate_relevance_score_value(content, doc_type)
            return AgentResult(
                success=True,
                data={'relevance_score': score},
                metadata={'content_length': len(content), 'doc_type': doc_type}
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _clean_old_embeddings(self, parameters: Dict[str, Any]) -> AgentResult:
        """Clean old embeddings based on age threshold"""
        max_age_days = parameters.get('max_age_days', 30)
        
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Find chunks to remove
            chunks_to_remove = []
            for idx, chunk in self.chunk_cache.items():
                if chunk.timestamp < cutoff_date:
                    chunks_to_remove.append(idx)
            
            # Remove from cache
            for idx in chunks_to_remove:
                del self.chunk_cache[idx]
            
            # Rebuild index without old chunks
            if chunks_to_remove:
                await self._rebuild_index({})
            
            return AgentResult(
                success=True,
                data={'removed_chunks': len(chunks_to_remove)},
                metadata={
                    'max_age_days': max_age_days,
                    'cutoff_date': cutoff_date.isoformat(),
                    'remaining_chunks': len(self.chunk_cache)
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _get_embedding_stats(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get statistics about the embedding store"""
        try:
            if not self.chunk_cache:
                return AgentResult(
                    success=True,
                    data={
                        'total_chunks': 0,
                        'total_embeddings': 0,
                        'storage_size_mb': 0
                    }
                )
            
            # Calculate statistics
            doc_types = defaultdict(int)
            sources = defaultdict(int)
            freshness_scores = []
            relevance_scores = []
            timestamps = []
            
            for chunk in self.chunk_cache.values():
                doc_types[chunk.doc_type] += 1
                sources[chunk.source] += 1
                freshness_scores.append(chunk.freshness_score)
                relevance_scores.append(chunk.relevance_score)
                timestamps.append(chunk.timestamp)
            
            # Storage size estimation
            try:
                storage_size = sum(
                    (self.storage_path / f).stat().st_size 
                    for f in os.listdir(self.storage_path)
                ) / (1024 * 1024)  # Convert to MB
            except:
                storage_size = 0
            
            stats = {
                'total_chunks': len(self.chunk_cache),
                'total_embeddings': self.vector_store.ntotal,
                'doc_types': dict(doc_types),
                'sources': dict(sources),
                'average_freshness_score': np.mean(freshness_scores) if freshness_scores else 0,
                'average_relevance_score': np.mean(relevance_scores) if relevance_scores else 0,
                'oldest_chunk': min(timestamps).isoformat() if timestamps else None,
                'newest_chunk': max(timestamps).isoformat() if timestamps else None,
                'storage_size_mb': round(storage_size, 2),
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension()
            }
            
            return AgentResult(
                success=True,
                data=stats,
                metadata={'generated_at': datetime.now().isoformat()}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _rebuild_index(self, parameters: Dict[str, Any]) -> AgentResult:
        """Rebuild FAISS index from scratch"""
        try:
            if not self.chunk_cache:
                return AgentResult(
                    success=True,
                    data={'message': 'No chunks to rebuild index from'}
                )
            
            # Create new index
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatIP(embedding_dim)
            
            # Add all embeddings
            new_cache = {}
            for old_idx, chunk in self.chunk_cache.items():
                if chunk.embedding is not None:
                    embedding_vector = np.array([chunk.embedding], dtype=np.float32)
                    new_index.add(embedding_vector)
                    new_idx = new_index.ntotal - 1
                    new_cache[new_idx] = chunk
            
            # Replace old index and cache
            self.vector_store = new_index
            self.chunk_cache = new_cache
            
            # Save updated store
            self._save_vector_store()
            
            return AgentResult(
                success=True,
                data={
                    'chunks_rebuilt': len(self.chunk_cache),
                    'total_embeddings': self.vector_store.ntotal
                },
                metadata={'rebuild_timestamp': datetime.now().isoformat()}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def health_check(self) -> Dict[str, Any]:
        """Enhanced health check including embedding model status"""
        base_health = super().health_check()
        
        # Add embedding-specific health info
        embedding_status = {
            'model_loaded': self.embedding_model is not None,
            'vector_store_loaded': self.vector_store is not None,
            'total_embeddings': self.vector_store.ntotal if self.vector_store else 0,
            'storage_accessible': self.storage_path.exists(),
            'model_name': self.model_name
        }
        
        base_health['embedding_system'] = embedding_status
        base_health['healthy'] = (
            base_health['healthy'] and 
            embedding_status['model_loaded'] and 
            embedding_status['vector_store_loaded']
        )
        
        return base_health


# Utility function to create configured embedding agent
def create_embedding_agent(config: Dict[str, Any] = None) -> EmbeddingAgent:
    """Factory function to create configured embedding agent"""
    default_config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 512,
        'chunk_overlap': 50,
        'max_tokens': 8192,
        'storage_path': './data/embeddings',
        'timeout_seconds': 600
    }
    
    if config:
        default_config.update(config)
    
    return EmbeddingAgent(config=default_config)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_embedding_agent():
        """Test the embedding agent functionality"""
        # Initialize logging for console and file output
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/embedding_agent.log')
            ]
        )
        
        # Create embedding agent with default configuration
        agent = create_embedding_agent()
        logging.info(f"Agent Status: {agent.get_status()}")
        
        # Check initial health
        logging.info("\n--- Testing Health Check ---")
        health_status = agent.health_check()
        logging.info(f"Health Check: {health_status}")
        
        # Test document processing
        logging.info("\n--- Testing Document Processing ---")
        sample_docs = [
            {
                'content': 'Samsung Electronics reported strong Q1 2025 earnings with a 12% increase in semiconductor revenue. The company’s annual report was filed with the KSE on 2025-02-26.',
                'metadata': {'timestamp': datetime.now().isoformat(), 'symbol': '005930.KS', 'source': 'regulatory_kse', 'doc_type': 'regulatory_filing'},
                'source': 'regulatory_kse'
            },
            {
                'content': 'TSMC announced a 15% revenue growth in Q2 2025, driven by demand for AI chips. The earnings call highlighted expansion in Arizona facilities.',
                'metadata': {'timestamp': (datetime.now() - timedelta(days=1)).isoformat(), 'symbol': 'TSM', 'source': 'earnings_report', 'doc_type': 'earnings_report'},
                'source': 'earnings_report'
            },
            {
                'content': 'Asian tech markets saw volatility due to U.S. interest rate hikes. Samsung and TSMC stocks were impacted, with Samsung down 1.46%.',
                'metadata': {'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(), 'source': 'yahoo_finance_news', 'doc_type': 'financial_news'},
                'source': 'yahoo_finance_news'
            }
        ]
        
        # Process documents
        process_task = Task(
            id="process_documents_task",  # Add a unique ID
            type='process_documents',
            parameters={
                'documents': sample_docs,
                'doc_type': 'mixed',
                'source': 'test_source'
            },
            priority=TaskPriority.HIGH
        )
        process_result = await agent.execute_task(process_task)
        if process_result.success:
            logging.info(f"Processed {len(process_result.data)} chunks from {len(sample_docs)} documents")
            logging.info(f"Processing metadata: {process_result.metadata}")
        else:
            logging.error(f"Document processing failed: {process_result.error}")
            return
    # Run the test
    asyncio.run(test_embedding_agent())