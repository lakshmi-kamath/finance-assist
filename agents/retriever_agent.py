import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add these new imports for embedding functionality
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

from agents.base_agent import BaseAgent, Task, AgentResult, TaskPriority

# Initialize logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/retriever_agent.log')
    ]
)

@dataclass
class RetrievalQuery:
    """Query structure for retrieval requests"""
    query_text: str
    context: Dict[str, Any]
    max_results: int = 10
    freshness_weight: float = 0.3
    relevance_weight: float = 0.4
    similarity_weight: float = 0.3
    min_similarity_score: float = 0.3
    min_combined_score: float = 0.4
    time_window_hours: Optional[int] = None
    required_tickers: Optional[List[str]] = None
    required_sectors: Optional[List[str]] = None
    required_doc_types: Optional[List[str]] = None
    required_sources: Optional[List[str]] = None

@dataclass
class RetrievalResult:
    """Enhanced result from retrieval operation"""
    chunks: List[Dict[str, Any]]
    confidence_score: float
    total_matches: int
    filtered_matches: int
    query_time_ms: float
    query_embedding_time_ms: float
    search_time_ms: float
    scoring_time_ms: float
    metadata: Dict[str, Any]

class RetrieverAgent(BaseAgent):
    """Advanced retrieval agent with integrated embedding capabilities"""
    
    def __init__(self, agent_id: str = "retriever_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Configuration
        self.default_freshness_weight = self.config.get('default_freshness_weight', 0.3)
        self.default_relevance_weight = self.config.get('default_relevance_weight', 0.4)
        self.default_similarity_weight = self.config.get('default_similarity_weight', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_search_results = self.config.get('max_search_results', 50)
        
        # Embedding configuration
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.vector_db_path = self.config.get('vector_db_path', '')
        self.chunk_index_path = self.config.get('chunk_index_path', '')
        
        # Initialize embedding components
        self.embedding_model = None
        self.vector_index = None
        self.chunk_metadata = []
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Source credibility mapping
        self.source_credibility = self._initialize_source_credibility()
        
        # Financial keywords for content analysis
        self.financial_keywords = self._initialize_financial_keywords()
        
        # Sector mappings
        self.sector_keywords = self._initialize_sector_keywords()
        
        # Initialize embedding components
        self._initialize_embedding_components()
    
    def _define_capabilities(self) -> List[str]:
        """Define what this agent can do"""
        return [
            'retrieve_similar_chunks',
            'advanced_search',
            'contextual_retrieval',
            'portfolio_analysis_retrieval',
            'earnings_analysis_retrieval',
            'market_risk_retrieval',
            'news_sentiment_retrieval',
            'calculate_confidence_score',
            'filter_results',
            'rank_results',
            'get_retrieval_stats',
            'add_documents',
            'build_vector_index',
            'load_vector_index',
            'save_vector_index'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['sentence_transformers', 'faiss', 'numpy']
    
    def _initialize_embedding_components(self):
        """Initialize embedding model and vector database"""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Try to load existing vector index
            if os.path.exists(self.vector_db_path) and os.path.exists(self.chunk_index_path):
                self.load_vector_index()
            else:
                # Initialize empty FAISS index
                self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.chunk_metadata = []
                
            self.logger.info(f"Initialized embedding components with model: {self.embedding_model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding components: {e}")
            self.embedding_model = None
            self.vector_index = None
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings.astype(np.float32)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        try:
            if not self.embedding_model or not self.vector_index:
                self.logger.error("Embedding components not initialized")
                return False
            
            # Extract text content and prepare metadata
            texts = []
            metadata_batch = []
            
            for doc in documents:
                content = doc.get('content', '')
                if not content:
                    continue
                
                texts.append(content)
                metadata_batch.append({
                    'chunk_id': doc.get('chunk_id', f"chunk_{len(self.chunk_metadata)}"),
                    'content': content,
                    'source': doc.get('source', 'unknown'),
                    'doc_type': doc.get('doc_type', 'unknown'),
                    'timestamp': doc.get('timestamp', datetime.now().isoformat()),
                    'metadata': doc.get('metadata', {})
                })
            
            if not texts:
                self.logger.warning("No valid texts to add")
                return False
            
            # Generate embeddings
            embeddings = self._embed_texts(texts)
            
            # Add to FAISS index
            self.vector_index.add(embeddings)
            
            # Add metadata
            self.chunk_metadata.extend(metadata_batch)
            
            self.logger.info(f"Added {len(texts)} documents to vector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 10, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            if not self.embedding_model or not self.vector_index:
                raise RuntimeError("Embedding components not initialized")
            
            if self.vector_index.ntotal == 0:
                self.logger.warning("Vector index is empty")
                return []
            
            # Generate query embedding
            query_embedding = self._embed_text(query).reshape(1, -1)
            
            # Search FAISS index
            scores, indices = self.vector_index.search(query_embedding, min(top_k, self.vector_index.ntotal))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score < min_score:  # FAISS returns -1 for invalid indices
                    continue
                
                if idx >= len(self.chunk_metadata):
                    continue
                
                chunk_data = self.chunk_metadata[idx].copy()
                chunk_data['similarity_score'] = float(score)
                
                # Calculate freshness score
                chunk_data['freshness_score'] = self._calculate_freshness_score(
                    chunk_data.get('timestamp', datetime.now().isoformat())
                )
                
                # Calculate basic relevance score (can be enhanced)
                chunk_data['relevance_score'] = float(score)  # Using similarity as base relevance
                
                results.append(chunk_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def _calculate_freshness_score(self, timestamp: str) -> float:
        """Calculate freshness score based on timestamp"""
        try:
            chunk_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            
            # Age in hours
            age_hours = (now - chunk_time).total_seconds() / 3600
            
            # Exponential decay: fresher content gets higher scores
            # Score = e^(-age_hours/24) so content is ~37% fresh after 24 hours
            freshness = np.exp(-age_hours / 24)
            return min(max(freshness, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Default neutral score
    
    def save_vector_index(self) -> bool:
        """Save vector index and metadata to disk"""
        try:
            if not self.vector_index or not self.chunk_metadata:
                self.logger.warning("No data to save")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.chunk_index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.vector_index, self.vector_db_path)
            
            # Save metadata
            with open(self.chunk_index_path, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            
            self.logger.info(f"Saved vector index with {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving vector index: {e}")
            return False
    
    def load_vector_index(self) -> bool:
        """Load vector index and metadata from disk"""
        try:
            if not os.path.exists(self.vector_db_path) or not os.path.exists(self.chunk_index_path):
                self.logger.warning("Vector index files not found")
                return False
            
            # Load FAISS index
            self.vector_index = faiss.read_index(self.vector_db_path)
            
            # Load metadata
            with open(self.chunk_index_path, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            self.logger.info(f"Loaded vector index with {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector index: {e}")
            return False
    
    def _initialize_source_credibility(self) -> Dict[str, float]:
        """Initialize source credibility scores"""
        return {
            'sec.gov': 1.0,
            'regulatory_sec': 1.0,
            'regulatory_kse': 0.95,
            'regulatory_hkex': 0.95,
            'earnings_report': 0.9,
            'bloomberg.com': 0.9,
            'bloomberg': 0.9,
            'reuters.com': 0.9,
            'reuters': 0.9,
            'marketwatch.com': 0.8,
            'marketwatch': 0.8,
            'yahoo_finance': 0.8,
            'cnbc.com': 0.8,
            'cnbc': 0.8,
            'financial_times': 0.85,
            'wsj.com': 0.85,
            'wall_street_journal': 0.85,
            'seekingalpha.com': 0.6,
            'seekingalpha': 0.6,
            'motley_fool': 0.5,
            'reddit': 0.3,
            'twitter': 0.2,
            'social_media': 0.2,
            'unknown': 0.5
        }
    
    def _initialize_financial_keywords(self) -> Dict[str, List[str]]:
        """Initialize financial keywords for different categories"""
        return {
            'earnings': [
                'earnings', 'revenue', 'profit', 'loss', 'eps', 'ebitda', 
                'guidance', 'outlook', 'quarterly', 'annual', 'beat', 'miss'
            ],
            'market': [
                'stock', 'market', 'trading', 'price', 'volume', 'volatility',
                'bullish', 'bearish', 'rally', 'correction', 'bubble', 'crash'
            ],
            'risk': [
                'risk', 'volatility', 'exposure', 'hedge', 'diversification',
                'correlation', 'beta', 'var', 'stress', 'scenario'
            ],
            'economic': [
                'gdp', 'inflation', 'interest', 'fed', 'monetary', 'fiscal',
                'unemployment', 'cpi', 'ppi', 'yield', 'curve'
            ],
            'investment': [
                'investment', 'portfolio', 'allocation', 'return', 'performance',
                'benchmark', 'alpha', 'sharpe', 'drawdown', 'rebalancing'
            ]
        }
    
    def _initialize_sector_keywords(self) -> Dict[str, List[str]]:
        """Initialize sector-specific keywords"""
        return {
            'Technology': [
                'semiconductor', 'chip', 'ai', 'artificial intelligence', 'cloud',
                'software', 'hardware', 'tsmc', 'samsung', 'nvidia', 'apple'
            ],
            'Healthcare': [
                'pharmaceutical', 'biotech', 'medical', 'drug', 'clinical',
                'fda', 'healthcare', 'hospital', 'therapy', 'vaccine'
            ],
            'Financial': [
                'bank', 'banking', 'insurance', 'credit', 'loan', 'mortgage',
                'fintech', 'payment', 'trading', 'brokerage'
            ],
            'Energy': [
                'oil', 'gas', 'renewable', 'solar', 'wind', 'nuclear',
                'petroleum', 'lng', 'pipeline', 'refinery'
            ],
            'Consumer': [
                'retail', 'consumer', 'brand', 'ecommerce', 'shopping',
                'discretionary', 'staples', 'luxury', 'fashion'
            ]
        }
    
    async def execute_task(self, task: Task) -> AgentResult:
        """Execute retrieval-related tasks"""
        task_type = task.type
        parameters = task.parameters
        
        try:
            if task_type == 'retrieve_similar_chunks':
                return await self._retrieve_similar_chunks(parameters)
            
            elif task_type == 'advanced_search':
                return await self._advanced_search(parameters)
            
            elif task_type == 'contextual_retrieval':
                return await self._contextual_retrieval(parameters)
            
            elif task_type == 'portfolio_analysis_retrieval':
                return await self._portfolio_analysis_retrieval(parameters)
            
            elif task_type == 'earnings_analysis_retrieval':
                return await self._earnings_analysis_retrieval(parameters)
            
            elif task_type == 'market_risk_retrieval':
                return await self._market_risk_retrieval(parameters)
            
            elif task_type == 'news_sentiment_retrieval':
                return await self._news_sentiment_retrieval(parameters)
            
            elif task_type == 'calculate_confidence_score':
                return await self._calculate_confidence_score(parameters)
            
            elif task_type == 'filter_results':
                return await self._filter_results(parameters)
            
            elif task_type == 'rank_results':
                return await self._rank_results(parameters)
            
            elif task_type == 'get_retrieval_stats':
                return await self._get_retrieval_stats(parameters)
            
            elif task_type == 'add_documents':
                success = self.add_documents(parameters.get('documents', []))
                return AgentResult(success=success, data={'documents_added': success})
            
            elif task_type == 'build_vector_index':
                # Alias for add_documents for backward compatibility
                success = self.add_documents(parameters.get('documents', []))
                return AgentResult(success=success, data={'index_built': success})
            
            elif task_type == 'load_vector_index':
                success = self.load_vector_index()
                return AgentResult(success=success, data={'index_loaded': success})
            
            elif task_type == 'save_vector_index':
                success = self.save_vector_index()
                return AgentResult(success=success, data={'index_saved': success})
            
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
    
    async def _retrieve_similar_chunks(self, parameters: Dict[str, Any]) -> AgentResult:
        """Main retrieval method using integrated embedding capabilities"""
        query_text = parameters.get('query', '')
        context = parameters.get('context', {})
        max_results = parameters.get('max_results', 10)
        
        if not query_text:
            return AgentResult(success=False, error="No query text provided")
        
        start_time = datetime.now()
        
        try:
            # Create retrieval query object
            retrieval_query = RetrievalQuery(
                query_text=query_text,
                context=context,
                max_results=max_results,
                **{k: v for k, v in parameters.items() if k in [
                    'freshness_weight', 'relevance_weight', 'similarity_weight',
                    'min_similarity_score', 'min_combined_score', 'time_window_hours',
                    'required_tickers', 'required_sectors', 'required_doc_types', 'required_sources'
                ]}
            )
            
            # Get similar chunks using integrated search
            embedding_start = datetime.now()
            raw_chunks = self.search_similar_chunks(
                query_text,
                top_k=self.max_search_results,
                min_score=retrieval_query.min_similarity_score
            )
            embedding_time = (datetime.now() - embedding_start).total_seconds() * 1000
            
            if not raw_chunks:
                return AgentResult(
                    success=True,
                    data=RetrievalResult(
                        chunks=[],
                        confidence_score=0.0,
                        total_matches=0,
                        filtered_matches=0,
                        query_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        query_embedding_time_ms=embedding_time,
                        search_time_ms=0,
                        scoring_time_ms=0,
                        metadata={'query': query_text, 'context': context}
                    ).__dict__
                )
            
            search_time_start = datetime.now()
            search_time = (datetime.now() - search_time_start).total_seconds() * 1000
            
            # Apply advanced filtering and scoring
            scoring_start = datetime.now()
            filtered_chunks = self._apply_advanced_filters(raw_chunks, retrieval_query)
            enhanced_chunks = await self._enhance_chunk_scoring(filtered_chunks, retrieval_query)
            final_chunks = self._rank_and_limit_results(enhanced_chunks, retrieval_query)
            scoring_time = (datetime.now() - scoring_start).total_seconds() * 1000
            
            # Calculate confidence
            confidence = self._calculate_retrieval_confidence(final_chunks, retrieval_query)
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = RetrievalResult(
                chunks=final_chunks,
                confidence_score=confidence,
                total_matches=len(raw_chunks),
                filtered_matches=len(filtered_chunks),
                query_time_ms=total_time,
                query_embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                scoring_time_ms=scoring_time,
                metadata={
                    'query': query_text,
                    'context': context,
                    'retrieval_params': asdict(retrieval_query)
                }
            )
            
            return AgentResult(
                success=True,
                data=asdict(result),
                metadata=result.metadata
            )
            
        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Error in retrieval: {e}")
            return AgentResult(
                success=False,
                error=str(e),
                metadata={'query_time_ms': total_time}
            )
    
    # Keep all the existing filtering, scoring, and ranking methods unchanged
    # ... (rest of the methods remain the same)
    
    def _apply_advanced_filters(self, chunks: List[Dict[str, Any]], 
                              query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Apply advanced filtering based on query parameters"""
        filtered = []
        
        for chunk in chunks:
            # Time window filter
            if query.time_window_hours:
                chunk_time = datetime.fromisoformat(chunk['timestamp'])
                cutoff_time = datetime.now() - timedelta(hours=query.time_window_hours)
                if chunk_time < cutoff_time:
                    continue
            
            # Ticker filter
            if query.required_tickers:
                chunk_metadata = chunk.get('metadata', {})
                chunk_ticker = chunk_metadata.get('symbol') or chunk_metadata.get('ticker')
                if chunk_ticker and chunk_ticker not in query.required_tickers:
                    continue
            
            # Sector filter
            if query.required_sectors:
                chunk_sector = self._extract_sector_from_chunk(chunk)
                if chunk_sector and chunk_sector not in query.required_sectors:
                    continue
            
            # Document type filter
            if query.required_doc_types:
                if chunk['doc_type'] not in query.required_doc_types:
                    continue
            
            # Source filter
            if query.required_sources:
                if chunk['source'] not in query.required_sources:
                    continue
            
            # Minimum similarity score filter
            if chunk['similarity_score'] < query.min_similarity_score:
                continue
            
            filtered.append(chunk)
        
        return filtered
    
    def _extract_sector_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract sector information from chunk"""
        # Check metadata first
        metadata = chunk.get('metadata', {})
        if 'sector' in metadata:
            return metadata['sector']
        
        # Try to infer from content using keywords
        content = chunk['content'].lower()
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    return sector
        
        return None
    
    async def _enhance_chunk_scoring(self, chunks: List[Dict[str, Any]], 
                                   query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Enhance chunks with advanced scoring"""
        enhanced_chunks = []
        
        for chunk in chunks:
            enhanced_chunk = chunk.copy()
            
            # Calculate context relevance score
            context_score = self._calculate_context_relevance(chunk, query.context)
            enhanced_chunk['context_relevance_score'] = context_score
            
            # Calculate source credibility score
            credibility_score = self._get_source_credibility_score(chunk['source'])
            enhanced_chunk['source_credibility_score'] = credibility_score
            
            # Calculate content quality score
            quality_score = self._calculate_content_quality_score(chunk)
            enhanced_chunk['content_quality_score'] = quality_score
            
            # Calculate keyword relevance score
            keyword_score = self._calculate_keyword_relevance(chunk, query.query_text)
            enhanced_chunk['keyword_relevance_score'] = keyword_score
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(enhanced_chunk, query)
            enhanced_chunk['composite_score'] = composite_score
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _calculate_context_relevance(self, chunk: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how well chunk matches the query context"""
        relevance = 0.0
        
        chunk_metadata = chunk.get('metadata', {})
        
        # Portfolio ticker matching
        if 'portfolio_tickers' in context:
            chunk_ticker = chunk_metadata.get('symbol') or chunk_metadata.get('ticker')
            if chunk_ticker and chunk_ticker in context['portfolio_tickers']:
                relevance += 0.4
        
        # Sector matching
        if 'focus_sectors' in context:
            chunk_sector = self._extract_sector_from_chunk(chunk)
            if chunk_sector and chunk_sector in context['focus_sectors']:
                relevance += 0.3
        
        # Document type preference
        if 'preferred_doc_types' in context:
            if chunk['doc_type'] in context['preferred_doc_types']:
                relevance += 0.2
        
        # Time preference (recent vs historical)
        if 'time_preference' in context:
            chunk_age_hours = (datetime.now() - datetime.fromisoformat(chunk['timestamp'])).total_seconds() / 3600
            if context['time_preference'] == 'recent' and chunk_age_hours < 24:
                relevance += 0.1
            elif context['time_preference'] == 'historical' and chunk_age_hours > 168:  # 1 week
                relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _get_source_credibility_score(self, source: str) -> float:
        """Get credibility score for a source"""
        source_lower = source.lower()
        
        # Check exact matches first
        if source_lower in self.source_credibility:
            return self.source_credibility[source_lower]
        
        # Check partial matches
        for source_key, score in self.source_credibility.items():
            if source_key in source_lower:
                return score
        
        return self.source_credibility['unknown']
    
    def _calculate_content_quality_score(self, chunk: Dict[str, Any]) -> float:
        """Calculate content quality based on various factors"""
        content = chunk['content']
        
        # Length factor (not too short, not too long)
        length_score = 0.5
        if 100 <= len(content) <= 2000:
            length_score = 1.0
        elif 50 <= len(content) < 100 or 2000 < len(content) <= 5000:
            length_score = 0.8
        
        # Structure factor (presence of sentences, punctuation)
        structure_score = 0.5
        sentences = content.count('.') + content.count('!') + content.count('?')
        if sentences >= 2:
            structure_score = min(sentences / 10, 1.0)
        
        # Information density (presence of numbers, financial terms)
        density_score = 0.5
        numbers = len([word for word in content.split() if any(char.isdigit() for char in word)])
        if numbers > 0:
            density_score = min(numbers / 20, 1.0)
        
        return (length_score + structure_score + density_score) / 3
    
    def _calculate_keyword_relevance(self, chunk: Dict[str, Any], query: str) -> float:
        """Calculate keyword relevance between chunk and query"""
        content = chunk['content'].lower()
        query_words = set(query.lower().split())
        
        # Direct word matches
        content_words = set(content.split())
        direct_matches = len(query_words.intersection(content_words))
        direct_score = min(direct_matches / max(len(query_words), 1), 1.0)
        
        # Financial keyword category matches
        category_score = 0.0
        for category, keywords in self.financial_keywords.items():
            query_category_matches = sum(1 for word in query_words if word in keywords)
            content_category_matches = sum(1 for keyword in keywords if keyword in content)
            
            if query_category_matches > 0 and content_category_matches > 0:
                category_score += 0.2
        
        category_score = min(category_score, 1.0)
        
        return (direct_score * 0.7 + category_score * 0.3)
    
    
    def _calculate_composite_score(self, chunk: Dict[str, Any], query: RetrievalQuery) -> float:
        """Calculate composite score using weighted combination of all factors"""
        similarity_score = chunk.get('similarity_score', 0.0)
        freshness_score = chunk.get('freshness_score', 0.0)
        relevance_score = chunk.get('relevance_score', 0.0)
        context_relevance = chunk.get('context_relevance_score', 0.0)
        credibility_score = chunk.get('source_credibility_score', 0.5)
        quality_score = chunk.get('content_quality_score', 0.5)
        keyword_score = chunk.get('keyword_relevance_score', 0.0)
        
        # Weighted combination
        composite = (
            similarity_score * query.similarity_weight +
            freshness_score * query.freshness_weight +
            relevance_score * query.relevance_weight +
            context_relevance * 0.15 +
            credibility_score * 0.1 +
            quality_score * 0.05 +
            keyword_score * 0.1
        )
        
        return min(composite, 1.0)
    
    def _rank_and_limit_results(self, chunks: List[Dict[str, Any]], 
                               query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Rank chunks by composite score and limit results"""
        # Filter by minimum combined score
        qualified_chunks = [
            chunk for chunk in chunks 
            if chunk.get('composite_score', 0.0) >= query.min_combined_score
        ]
        
        # Sort by composite score (descending)
        qualified_chunks.sort(key=lambda x: x.get('composite_score', 0.0), reverse=True)
        
        # Limit results
        return qualified_chunks[:query.max_results]
    
    def _calculate_retrieval_confidence(self, chunks: List[Dict[str, Any]], 
                                      query: RetrievalQuery) -> float:
        """Calculate overall confidence in retrieval results"""
        if not chunks:
            return 0.0
        
        # Average composite score
        avg_score = sum(chunk.get('composite_score', 0.0) for chunk in chunks) / len(chunks)
        
        # Score variance (lower variance = higher confidence)
        scores = [chunk.get('composite_score', 0.0) for chunk in chunks]
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        variance_penalty = min(score_variance * 2, 0.3)  # Cap penalty at 0.3
        
        # Result count factor (more results can indicate better coverage)
        count_factor = min(len(chunks) / query.max_results, 1.0) * 0.1
        
        # Source diversity factor
        unique_sources = len(set(chunk.get('source', 'unknown') for chunk in chunks))
        diversity_factor = min(unique_sources / max(len(chunks), 1), 1.0) * 0.1
        
        confidence = avg_score - variance_penalty + count_factor + diversity_factor
        return min(max(confidence, 0.0), 1.0)
    
    async def _advanced_search(self, parameters: Dict[str, Any]) -> AgentResult:
        """Advanced search with multiple query expansion techniques"""
        query_text = parameters.get('query', '')
        
        if not query_text:
            return AgentResult(success=False, error="No query text provided")
        
        try:
            # Expand query with synonyms and related terms
            expanded_queries = self._expand_query(query_text)
            
            all_results = []
            for expanded_query in expanded_queries:
                # Use the main retrieval method
                result = await self._retrieve_similar_chunks({
                    **parameters,
                    'query': expanded_query,
                    'max_results': parameters.get('max_results', 10) // len(expanded_queries) + 1
                })
                
                if result.success and result.data.get('chunks'):
                    all_results.extend(result.data['chunks'])
            
            # Deduplicate and re-rank
            deduplicated = self._deduplicate_chunks(all_results)
            
            # Create final retrieval query for ranking
            retrieval_query = RetrievalQuery(
                query_text=query_text,
                context=parameters.get('context', {}),
                max_results=parameters.get('max_results', 10)
            )
            
            final_results = self._rank_and_limit_results(deduplicated, retrieval_query)
            confidence = self._calculate_retrieval_confidence(final_results, retrieval_query)
            
            return AgentResult(
                success=True,
                data={
                    'chunks': final_results,
                    'confidence_score': confidence,
                    'total_matches': len(all_results),
                    'deduplicated_matches': len(deduplicated),
                    'expanded_queries': expanded_queries
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in advanced search: {e}")
            return AgentResult(success=False, error=str(e))
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with financial synonyms and related terms"""
        expanded = [query]  # Always include original query
        
        query_lower = query.lower()
        
        # Financial term expansions
        expansions = {
            'earnings': ['earnings', 'revenue', 'profit', 'income statement'],
            'risk': ['risk', 'volatility', 'exposure', 'uncertainty'],
            'performance': ['performance', 'returns', 'growth', 'results'],
            'market': ['market', 'trading', 'stock', 'equity'],
            'analysis': ['analysis', 'research', 'report', 'study'],
            'outlook': ['outlook', 'forecast', 'guidance', 'projection']
        }
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    if synonym != term and synonym not in query_lower:
                        expanded.append(query.replace(term, synonym))
                        break  # Add only one expansion per term to avoid explosion
        
        return expanded[:3]  # Limit to 3 expansions to manage performance
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on content similarity"""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_contents = set()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Simple deduplication based on content hash
            content_hash = hash(content[:200])  # Use first 200 chars for hash
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    async def _contextual_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Retrieval optimized for specific context"""
        context_type = parameters.get('context_type', 'general')
        
        # Adjust weights based on context type
        context_configs = {
            'earnings_analysis': {
                'freshness_weight': 0.4,
                'relevance_weight': 0.4,
                'similarity_weight': 0.2,
                'required_doc_types': ['earnings_report', 'financial_statement'],
                'min_similarity_score': 0.4
            },
            'market_sentiment': {
                'freshness_weight': 0.5,
                'relevance_weight': 0.3,
                'similarity_weight': 0.2,
                'required_doc_types': ['news', 'social_media', 'analyst_report'],
                'time_window_hours': 72
            },
            'risk_assessment': {
                'freshness_weight': 0.2,
                'relevance_weight': 0.5,
                'similarity_weight': 0.3,
                'required_doc_types': ['regulatory', 'risk_report', 'financial_statement'],
                'min_combined_score': 0.5
            },
            'portfolio_optimization': {
                'freshness_weight': 0.3,
                'relevance_weight': 0.4,
                'similarity_weight': 0.3,
                'required_doc_types': ['performance_report', 'market_data', 'research_report']
            }
        }
        
        # Apply context-specific configuration
        config = context_configs.get(context_type, {})
        enhanced_params = {**parameters, **config}
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _portfolio_analysis_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for portfolio analysis"""
        portfolio_tickers = parameters.get('tickers', [])
        
        if not portfolio_tickers:
            return AgentResult(success=False, error="No portfolio tickers provided")
        
        enhanced_params = {
            **parameters,
            'context': {
                **parameters.get('context', {}),
                'portfolio_tickers': portfolio_tickers,
                'focus_sectors': parameters.get('sectors', [])
            },
            'required_tickers': portfolio_tickers,
            'freshness_weight': 0.3,
            'relevance_weight': 0.4,
            'similarity_weight': 0.3,
            'max_results': parameters.get('max_results', 15)
        }
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _earnings_analysis_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for earnings analysis"""
        ticker = parameters.get('ticker')
        
        enhanced_params = {
            **parameters,
            'context': {
                **parameters.get('context', {}),
                'time_preference': 'recent'
            },
            'required_tickers': [ticker] if ticker else None,
            'required_doc_types': ['earnings_report', 'financial_statement', 'earnings_call'],
            'time_window_hours': parameters.get('time_window_hours', 168),  # 1 week default
            'freshness_weight': 0.4,
            'relevance_weight': 0.4,
            'similarity_weight': 0.2
        }
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _market_risk_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for market risk analysis"""
        enhanced_params = {
            **parameters,
            'context': {
                **parameters.get('context', {}),
                'preferred_doc_types': ['risk_report', 'regulatory', 'market_analysis']
            },
            'required_doc_types': ['risk_report', 'regulatory', 'market_analysis', 'research_report'],
            'freshness_weight': 0.2,
            'relevance_weight': 0.5,
            'similarity_weight': 0.3,
            'min_combined_score': 0.5
        }
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _news_sentiment_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for news sentiment analysis"""
        enhanced_params = {
            **parameters,
            'context': {
                **parameters.get('context', {}),
                'time_preference': 'recent'
            },
            'required_doc_types': ['news', 'press_release', 'analyst_report'],
            'time_window_hours': parameters.get('time_window_hours', 48),  # 2 days default
            'freshness_weight': 0.5,
            'relevance_weight': 0.3,
            'similarity_weight': 0.2,
            'max_results': parameters.get('max_results', 20)
        }
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _calculate_confidence_score(self, parameters: Dict[str, Any]) -> AgentResult:
        """Calculate confidence score for given results"""
        chunks = parameters.get('chunks', [])
        query = parameters.get('query', '')
        
        if not chunks or not query:
            return AgentResult(success=False, error="Missing chunks or query")
        
        try:
            retrieval_query = RetrievalQuery(query_text=query, context={})
            confidence = self._calculate_retrieval_confidence(chunks, retrieval_query)
            
            return AgentResult(
                success=True,
                data={'confidence_score': confidence}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _filter_results(self, parameters: Dict[str, Any]) -> AgentResult:
        """Filter results based on criteria"""
        chunks = parameters.get('chunks', [])
        filters = parameters.get('filters', {})
        
        try:
            filtered_chunks = []
            
            for chunk in chunks:
                include = True
                
                # Apply filters
                if 'min_score' in filters:
                    if chunk.get('composite_score', 0.0) < filters['min_score']:
                        include = False
                
                if 'sources' in filters:
                    if chunk.get('source') not in filters['sources']:
                        include = False
                
                if 'doc_types' in filters:
                    if chunk.get('doc_type') not in filters['doc_types']:
                        include = False
                
                if 'tickers' in filters:
                    chunk_ticker = chunk.get('metadata', {}).get('ticker')
                    if chunk_ticker not in filters['tickers']:
                        include = False
                
                if 'time_range' in filters:
                    chunk_time = datetime.fromisoformat(chunk['timestamp'])
                    start_time = datetime.fromisoformat(filters['time_range']['start'])
                    end_time = datetime.fromisoformat(filters['time_range']['end'])
                    if not (start_time <= chunk_time <= end_time):
                        include = False
                
                if include:
                    filtered_chunks.append(chunk)
            
            return AgentResult(
                success=True,
                data={
                    'filtered_chunks': filtered_chunks,
                    'original_count': len(chunks),
                    'filtered_count': len(filtered_chunks)
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _rank_results(self, parameters: Dict[str, Any]) -> AgentResult:
        """Rank results by specified criteria"""
        chunks = parameters.get('chunks', [])
        ranking_method = parameters.get('method', 'composite_score')
        reverse = parameters.get('reverse', True)
        
        try:
            if ranking_method == 'composite_score':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('composite_score', 0.0),
                    reverse=reverse
                )
            elif ranking_method == 'freshness':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('freshness_score', 0.0),
                    reverse=reverse
                )
            elif ranking_method == 'similarity':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('similarity_score', 0.0),
                    reverse=reverse
                )
            elif ranking_method == 'timestamp':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('timestamp', ''),
                    reverse=reverse
                )
            else:
                return AgentResult(success=False, error=f"Unknown ranking method: {ranking_method}")
            
            return AgentResult(
                success=True,
                data={'ranked_chunks': ranked_chunks}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _get_retrieval_stats(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get statistics about the retrieval system"""
        try:
            stats = {
                'vector_index_size': self.vector_index.ntotal if self.vector_index else 0,
                'chunk_metadata_count': len(self.chunk_metadata),
                'embedding_model': self.embedding_model_name,
                'embedding_dimension': self.embedding_dim,
                'vector_db_path': self.vector_db_path,
                'chunk_index_path': self.chunk_index_path,
                'source_credibility_entries': len(self.source_credibility),
                'financial_keyword_categories': len(self.financial_keywords),
                'sector_keyword_categories': len(self.sector_keywords)
            }
            
            # Calculate source distribution if requested
            if parameters.get('include_source_distribution', False):
                source_counts = defaultdict(int)
                for chunk in self.chunk_metadata:
                    source_counts[chunk.get('source', 'unknown')] += 1
                stats['source_distribution'] = dict(source_counts)
            
            # Calculate document type distribution if requested
            if parameters.get('include_doc_type_distribution', False):
                doc_type_counts = defaultdict(int)
                for chunk in self.chunk_metadata:
                    doc_type_counts[chunk.get('doc_type', 'unknown')] += 1
                stats['doc_type_distribution'] = dict(doc_type_counts)
            
            return AgentResult(success=True, data=stats)
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the retriever agent"""
        return {
            'agent_id': self.agent_id,
            'status': 'active' if self.embedding_model and self.vector_index else 'inactive',
            'embedding_model': self.embedding_model_name,
            'vector_index_size': self.vector_index.ntotal if self.vector_index else 0,
            'chunk_count': len(self.chunk_metadata),
            'capabilities': self._define_capabilities(),
            'dependencies': self._define_dependencies()
        }
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self.embedding_model:
                return False
            
            if not self.vector_index:
                return False
            
            # Test embedding generation
            test_embedding = self._embed_text("test")
            if test_embedding is None or len(test_embedding) != self.embedding_dim:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
# Example usage and testing functions

async def example_usage():
    """Example usage of the RetrieverAgent"""
    
    # Initialize the agent with proper configuration
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_db_path': 'knowledge_base/vector_store',
        'chunk_index_path': 'knowledge_base/vector_store/metadata.pkl',
        'confidence_threshold': 0.7,
        'max_search_results': 50,
        'default_freshness_weight': 0.3,
        'default_relevance_weight': 0.4,
        'default_similarity_weight': 0.3
    }
    
    retriever = RetrieverAgent('retriever_agent', config)
    
    # Wait a moment for initialization
    await asyncio.sleep(0.1)
    
    print("RetrieverAgent Status:")
    status = retriever.get_status()
    print(json.dumps(status, indent=2))
    
    # Example sample documents to add to the vector database
    sample_documents = [
        {
            'chunk_id': 'apple_earnings_q4_2024',
            'content': 'Apple Inc. reported strong quarterly earnings for Q4 2024 with revenue of $119.58 billion, beating analyst expectations. iPhone sales showed resilience despite market headwinds, while Services revenue grew 16% year-over-year to $22.31 billion.',
            'source': 'sec.gov',
            'doc_type': 'earnings_report',
            'timestamp': '2024-11-01T16:30:00',
            'metadata': {'ticker': 'AAPL', 'sector': 'Technology', 'quarter': 'Q4_2024'}
        },
        {
            'chunk_id': 'msft_cloud_growth_2024',
            'content': 'Microsoft Azure cloud services experienced accelerated growth in 2024, with revenue increasing 29% year-over-year. The company attributed this growth to increased enterprise adoption of AI-powered cloud solutions and hybrid work technologies.',
            'source': 'bloomberg.com',
            'doc_type': 'news',
            'timestamp': '2024-10-15T09:15:00',
            'metadata': {'ticker': 'MSFT', 'sector': 'Technology'}
        },
        {
            'chunk_id': 'tech_sector_outlook_2024',
            'content': 'Technology sector analysis indicates mixed performance in 2024. While AI and cloud computing sectors show strong growth, semiconductor companies face headwinds from supply chain constraints and geopolitical tensions affecting chip manufacturing.',
            'source': 'reuters.com',
            'doc_type': 'research_report',
            'timestamp': '2024-10-20T14:45:00',
            'metadata': {'sector': 'Technology'}
        },
        {
            'chunk_id': 'market_risk_assessment_2024',
            'content': 'Current market risk assessment highlights elevated volatility in technology stocks due to interest rate uncertainty and regulatory concerns. Portfolio diversification strategies should consider exposure to high-growth sectors while managing downside risk.',
            'source': 'financial_times',
            'doc_type': 'risk_report',
            'timestamp': '2024-10-25T11:30:00',
            'metadata': {'doc_category': 'risk_analysis'}
        }
    ]
    
    # Add sample documents to the vector database
    print("\nAdding sample documents to vector database...")
    add_docs_task = Task(
        id='add_documents_1',
        type='add_documents',
        parameters={'documents': sample_documents},
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(add_docs_task)
    print(f"Documents added successfully: {result.success}")
    if result.success:
        print(f"Added {len(sample_documents)} documents to the index")
    
    # Save the vector index
    save_task = Task(
        id='save_index_1',
        type='save_vector_index',
        parameters={},
        priority=TaskPriority.MEDIUM
    )
    
    save_result = await retriever.execute_task(save_task)
    print(f"Vector index saved: {save_result.success}")
    
    # Example 1: Basic retrieval
    print("\n" + "="*50)
    print("Example 1: Basic Retrieval")
    print("="*50)
    
    basic_task = Task(
        id='basic_retrieval_1',
        type='retrieve_similar_chunks',
        parameters={
            'query': 'Apple quarterly earnings performance',
            'max_results': 5,
            'freshness_weight': 0.4,
            'relevance_weight': 0.4,
            'similarity_weight': 0.2,
            'context': {
                'portfolio_tickers': ['AAPL'],
                'time_preference': 'recent'
            }
        },
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(basic_task)
    print(f"Success: {result.success}")
    if result.success:
        data = result.data
        print(f"Found {len(data['chunks'])} chunks")
        print(f"Confidence Score: {data['confidence_score']:.3f}")
        print(f"Query Time: {data['query_time_ms']:.2f}ms")
        
        for i, chunk in enumerate(data['chunks'][:2]):  # Show first 2 results
            print(f"\nResult {i+1}:")
            print(f"  Source: {chunk['source']}")
            print(f"  Doc Type: {chunk['doc_type']}")
            print(f"  Similarity Score: {chunk['similarity_score']:.3f}")
            print(f"  Composite Score: {chunk.get('composite_score', 'N/A')}")
            print(f"  Content: {chunk['content'][:150]}...")
    
    # Example 2: Portfolio analysis retrieval
    print("\n" + "="*50)
    print("Example 2: Portfolio Analysis Retrieval")
    print("="*50)
    
    portfolio_task = Task(
        id='portfolio_analysis_1',
        type='portfolio_analysis_retrieval',
        parameters={
            'query': 'technology sector performance analysis',
            'tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'sectors': ['Technology'],
            'max_results': 10
        },
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(portfolio_task)
    print(f"Success: {result.success}")
    if result.success:
        data = result.data
        print(f"Found {len(data['chunks'])} chunks for portfolio analysis")
        print(f"Confidence Score: {data['confidence_score']:.3f}")
        
        # Show sources found
        sources = set(chunk['source'] for chunk in data['chunks'])
        print(f"Sources: {', '.join(sources)}")
    
    # Example 3: Advanced search with query expansion
    print("\n" + "="*50)
    print("Example 3: Advanced Search")
    print("="*50)
    
    advanced_task = Task(
        id='advanced_search_1',
        type='advanced_search',
        parameters={
            'query': 'market risk technology sector',
            'max_results': 8,
            'context': {
                'focus_sectors': ['Technology'],
                'preferred_doc_types': ['risk_report', 'research_report']
            }
        },
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(advanced_task)
    print(f"Success: {result.success}")
    if result.success:
        data = result.data
        print(f"Found {len(data['chunks'])} chunks with advanced search")
        print(f"Confidence Score: {data['confidence_score']:.3f}")
        if 'expanded_queries' in data:
            print(f"Expanded Queries: {data['expanded_queries']}")
    
    # Example 4: Contextual retrieval for earnings analysis
    print("\n" + "="*50)
    print("Example 4: Contextual Retrieval - Earnings Analysis")
    print("="*50)
    
    contextual_task = Task(
        id='contextual_retrieval_1',
        type='contextual_retrieval',
        parameters={
            'query': 'quarterly earnings revenue growth',
            'context_type': 'earnings_analysis',
            'max_results': 5
        },
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(contextual_task)
    print(f"Success: {result.success}")
    if result.success:
        data = result.data
        print(f"Found {len(data['chunks'])} chunks for earnings analysis")
        print(f"Confidence Score: {data['confidence_score']:.3f}")
    
    # Example 5: Earnings-specific retrieval
    print("\n" + "="*50)
    print("Example 5: Earnings Analysis Retrieval")
    print("="*50)
    
    earnings_task = Task(
        id='earnings_analysis_1',
        type='earnings_analysis_retrieval',
        parameters={
            'query': 'Apple earnings revenue performance',
            'ticker': 'AAPL',
            'time_window_hours': 168,  # 1 week
            'max_results': 3
        },
        priority=TaskPriority.HIGH
    )
    
    result = await retriever.execute_task(earnings_task)
    print(f"Success: {result.success}")
    if result.success:
        data = result.data
        print(f"Found {len(data['chunks'])} earnings-related chunks")
        print(f"Confidence Score: {data['confidence_score']:.3f}")
    
    # Example 6: Filter and rank results
    print("\n" + "="*50)
    print("Example 6: Filter and Rank Results")
    print("="*50)
    
    # First get some results to filter
    search_result = await retriever.execute_task(Task(
        id='search_for_filter',
        type='retrieve_similar_chunks',
        parameters={
            'query': 'technology performance',
            'max_results': 10
        },
        priority=TaskPriority.MEDIUM
    ))
    
    if search_result.success and search_result.data['chunks']:
        # Filter the results
        filter_task = Task(
            id='filter_results_1',
            type='filter_results',
            parameters={
                'chunks': search_result.data['chunks'],
                'filters': {
                    'min_score': 0.3,
                    'sources': ['sec.gov', 'bloomberg.com', 'reuters.com', 'financial_times'],
                    'doc_types': ['earnings_report', 'news', 'research_report']
                }
            },
            priority=TaskPriority.LOW
        )
        
        filter_result = await retriever.execute_task(filter_task)
        print(f"Filter Success: {filter_result.success}")
        if filter_result.success:
            filter_data = filter_result.data
            print(f"Filtered {filter_data['original_count']} -> {filter_data['filtered_count']} chunks")
            
            # Rank the filtered results
            rank_task = Task(
                id='rank_results_1',
                type='rank_results',
                parameters={
                    'chunks': filter_data['filtered_chunks'],
                    'method': 'composite_score',
                    'reverse': True
                },
                priority=TaskPriority.LOW
            )
            
            rank_result = await retriever.execute_task(rank_task)
            print(f"Ranking Success: {rank_result.success}")
            if rank_result.success:
                ranked_chunks = rank_result.data['ranked_chunks']
                print(f"Ranking complete. Top result score: {ranked_chunks[0].get('composite_score', 'N/A')}")
    
    # Example 7: Get comprehensive retrieval statistics
    print("\n" + "="*50)
    print("Example 7: Retrieval Statistics")
    print("="*50)
    
    stats_task = Task(
        id='get_stats_1',
        type='get_retrieval_stats',
        parameters={
            'include_source_distribution': True,
            'include_doc_type_distribution': True
        },
        priority=TaskPriority.LOW
    )
    
    result = await retriever.execute_task(stats_task)
    print(f"Success: {result.success}")
    if result.success:
        stats = result.data
        print(f"Vector Index Size: {stats['vector_index_size']}")
        print(f"Chunk Metadata Count: {stats['chunk_metadata_count']}")
        print(f"Embedding Model: {stats['embedding_model']}")
        print(f"Embedding Dimension: {stats['embedding_dimension']}")
        
        if 'source_distribution' in stats:
            print("Source Distribution:")
            for source, count in stats['source_distribution'].items():
                print(f"  {source}: {count}")
        
        if 'doc_type_distribution' in stats:
            print("Document Type Distribution:")
            for doc_type, count in stats['doc_type_distribution'].items():
                print(f"  {doc_type}: {count}")
    
    # Health check
    print("\n" + "="*50)
    print("Health Check")
    print("="*50)
    
    health_status = await retriever.health_check()
    print(f"System Health: {'HEALTHY' if health_status else 'UNHEALTHY'}")
    
    print("\nExample usage completed!")

if __name__ == "__main__":
    # Run example usage
    import asyncio
    import json
    import sys
    import os
    
    # Add project root to path (adjust as needed)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import necessary classes
    from agents.base_agent import Task, TaskPriority
    
    # Run the example
    asyncio.run(example_usage())