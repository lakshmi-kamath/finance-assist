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

from base_agent import BaseAgent, Task, AgentResult, TaskPriority

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
    """Advanced retrieval agent that works with the existing EmbeddingAgent"""
    
    def __init__(self, agent_id: str = "retriever_agent", config: Dict[str, Any] = None):
        super().__init__(agent_id, config)
        
        # Configuration
        self.embedding_agent_id = self.config.get('embedding_agent_id', 'embedding_agent')
        self.default_freshness_weight = self.config.get('default_freshness_weight', 0.3)
        self.default_relevance_weight = self.config.get('default_relevance_weight', 0.4)
        self.default_similarity_weight = self.config.get('default_similarity_weight', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_search_results = self.config.get('max_search_results', 50)
        
        # Source credibility mapping
        self.source_credibility = self._initialize_source_credibility()
        
        # Financial keywords for content analysis
        self.financial_keywords = self._initialize_financial_keywords()
        
        # Sector mappings
        self.sector_keywords = self._initialize_sector_keywords()
        
        # Reference to embedding agent (will be set externally)
        self.embedding_agent = None
    
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
            'get_retrieval_stats'
        ]
    
    def _define_dependencies(self) -> List[str]:
        """Define dependencies"""
        return ['embedding_agent', 'numpy']
    
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
    
    def set_embedding_agent(self, embedding_agent):
        """Set reference to embedding agent"""
        self.embedding_agent = embedding_agent
        self.logger.info("Embedding agent reference set")
    
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
        """Main retrieval method using the embedding agent"""
        if not self.embedding_agent:
            return AgentResult(success=False, error="Embedding agent not set")
        
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
            
            # Get similar chunks from embedding agent
            embedding_start = datetime.now()
            similar_chunks_task = Task(
                id=f"similar_chunks_{datetime.now().timestamp()}",
                type='get_similar_chunks',
                parameters={
                    'query': query_text,
                    'top_k': self.max_search_results,
                    'min_score': retrieval_query.min_similarity_score,
                    'include_metadata': True
                },
                priority=TaskPriority.HIGH
            )
            
            embedding_result = await self.embedding_agent.execute_task(similar_chunks_task)
            embedding_time = (datetime.now() - embedding_start).total_seconds() * 1000
            
            if not embedding_result.success:
                return AgentResult(
                    success=False,
                    error=f"Embedding search failed: {embedding_result.error}"
                )
            
            search_time_start = datetime.now()
            raw_chunks = embedding_result.data
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
                    'retrieval_params': asdict(retrieval_query),
                    'embedding_metadata': embedding_result.metadata
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
        """Calculate final composite score"""
        similarity_score = chunk['similarity_score']
        freshness_score = chunk['freshness_score']
        relevance_score = chunk['relevance_score']
        context_score = chunk['context_relevance_score']
        credibility_score = chunk['source_credibility_score']
        quality_score = chunk['content_quality_score']
        keyword_score = chunk['keyword_relevance_score']
        
        # Weighted combination
        composite = (
            similarity_score * query.similarity_weight +
            freshness_score * query.freshness_weight +
            relevance_score * query.relevance_weight +
            context_score * 0.15 +
            credibility_score * 0.1 +
            quality_score * 0.05 +
            keyword_score * 0.1
        )
        
        return min(composite, 1.0)
    
    def _rank_and_limit_results(self, chunks: List[Dict[str, Any]], 
                               query: RetrievalQuery) -> List[Dict[str, Any]]:
        """Rank results by composite score and apply final filtering"""
        # Filter by minimum combined score
        qualified_chunks = [
            chunk for chunk in chunks 
            if chunk['composite_score'] >= query.min_combined_score
        ]
        
        # Sort by composite score
        qualified_chunks.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Limit results
        return qualified_chunks[:query.max_results]
    
    def _calculate_retrieval_confidence(self, chunks: List[Dict[str, Any]], 
                                       query: RetrievalQuery) -> float:
        """Calculate overall confidence in retrieval results"""
        if not chunks:
            return 0.0
        
        # Base confidence from average composite score
        avg_score = np.mean([chunk['composite_score'] for chunk in chunks])
        
        # Diversity bonus (different sources and types)
        unique_sources = len(set(chunk['source'] for chunk in chunks))
        unique_types = len(set(chunk['doc_type'] for chunk in chunks))
        diversity_bonus = min((unique_sources + unique_types) / 10, 0.2)
        
        # Recency bonus (recent data available)
        recent_count = sum(
            1 for chunk in chunks 
            if (datetime.now() - datetime.fromisoformat(chunk['timestamp'])).hours < 24
        )
        recency_bonus = min(recent_count / len(chunks) * 0.1, 0.1)
        
        # Quality bonus (high credibility sources)
        high_credibility_count = sum(
            1 for chunk in chunks 
            if chunk['source_credibility_score'] >= 0.8
        )
        quality_bonus = min(high_credibility_count / len(chunks) * 0.1, 0.1)
        
        confidence = avg_score + diversity_bonus + recency_bonus + quality_bonus
        return min(confidence, 1.0)
    
    async def _advanced_search(self, parameters: Dict[str, Any]) -> AgentResult:
        """Advanced search with multiple queries and result fusion"""
        primary_query = parameters.get('primary_query', '')
        secondary_queries = parameters.get('secondary_queries', [])
        fusion_method = parameters.get('fusion_method', 'rank_fusion')
        
        if not primary_query:
            return AgentResult(success=False, error="No primary query provided")
        
        try:
            # Execute primary search
            primary_result = await self._retrieve_similar_chunks({
                **parameters,
                'query': primary_query
            })
            
            if not primary_result.success:
                return primary_result
            
            primary_data = primary_result.data
            all_results = [primary_data['chunks']]
            
            # Execute secondary searches
            for secondary_query in secondary_queries:
                secondary_result = await self._retrieve_similar_chunks({
                    **parameters,
                    'query': secondary_query,
                    'max_results': parameters.get('max_results', 10) // 2
                })
                
                if secondary_result.success:
                    all_results.append(secondary_result.data['chunks'])
            
            # Fuse results
            if fusion_method == 'rank_fusion':
                fused_chunks = self._rank_fusion(all_results)
            else:
                fused_chunks = self._score_fusion(all_results)
            
            # Limit final results
            max_results = parameters.get('max_results', 10)
            final_chunks = fused_chunks[:max_results]
            
            return AgentResult(
                success=True,
                data={
                    'chunks': final_chunks,
                    'fusion_method': fusion_method,
                    'queries_executed': len(all_results),
                    'total_candidates': sum(len(results) for results in all_results)
                },
                metadata={
                    'primary_query': primary_query,
                    'secondary_queries': secondary_queries,
                    'fusion_method': fusion_method
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _rank_fusion(self, result_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine results using rank-based fusion"""
        chunk_scores = defaultdict(float)
        chunk_data = {}
        
        for rank_weight, results in enumerate(result_lists, 1):
            for rank, chunk in enumerate(results):
                chunk_id = chunk['chunk_id']
                # Score based on rank position and list importance
                score = (1.0 / (rank + 1)) * (1.0 / rank_weight)
                chunk_scores[chunk_id] += score
                
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = chunk
        
        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [chunk_data[chunk_id] for chunk_id, _ in sorted_chunks]
    
    def _score_fusion(self, result_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combine results using score-based fusion"""
        chunk_scores = defaultdict(list)
        chunk_data = {}
        
        for results in result_lists:
            for chunk in results:
                chunk_id = chunk['chunk_id']
                chunk_scores[chunk_id].append(chunk['composite_score'])
                
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = chunk
        
        # Calculate combined scores (average)
        combined_scores = {
            chunk_id: np.mean(scores)
            for chunk_id, scores in chunk_scores.items()
        }
        
        # Sort by combined score
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [chunk_data[chunk_id] for chunk_id, _ in sorted_chunks]
    
    async def _contextual_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Retrieval with enhanced contextual understanding"""
        query = parameters.get('query', '')
        context_type = parameters.get('context_type', 'general')
        
        # Enhance query based on context type
        if context_type == 'earnings_analysis':
            return await self._earnings_analysis_retrieval(parameters)
        elif context_type == 'portfolio_analysis':
            return await self._portfolio_analysis_retrieval(parameters)
        elif context_type == 'market_risk':
            return await self._market_risk_retrieval(parameters)
        elif context_type == 'news_sentiment':
            return await self._news_sentiment_retrieval(parameters)
        else:
            return await self._retrieve_similar_chunks(parameters)
    
    async def _portfolio_analysis_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for portfolio analysis"""
        portfolio_tickers = parameters.get('portfolio_tickers', [])
        analysis_type = parameters.get('analysis_type', 'general')
        
        enhanced_params = parameters.copy()
        enhanced_params.update({
            'required_tickers': portfolio_tickers,
            'preferred_doc_types': ['earnings_report', 'market_data', 'financial_news'],
            'freshness_weight': 0.4,
            'relevance_weight': 0.4,
            'context': {
                'portfolio_tickers': portfolio_tickers,
                'analysis_type': analysis_type,
                'time_preference': 'recent'
            }
        })
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _earnings_analysis_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for earnings analysis"""
        enhanced_params = parameters.copy()
        enhanced_params.update({
            'required_doc_types': ['earnings_report', 'regulatory_sec', 'regulatory_kse'],
            'freshness_weight': 0.5,
            'time_window_hours': 720,  # 30 days
            'context': {
                'preferred_doc_types': ['earnings_report', 'regulatory_sec', 'regulatory_kse'],
                'analysis_type': 'earnings',
                'time_preference': 'recent'
            },
            'min_combined_score': 0.5
        })
        
        # Add earnings-specific keywords to query
        original_query = parameters.get('query', '')
        earnings_keywords = ' '.join(self.financial_keywords['earnings'][:5])
        enhanced_query = f"{original_query} {earnings_keywords}"
        enhanced_params['query'] = enhanced_query
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _market_risk_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for market risk analysis"""
        enhanced_params = parameters.copy()
        enhanced_params.update({
            'preferred_doc_types': ['market_data', 'financial_news', 'research_report'],
            'freshness_weight': 0.6,
            'time_window_hours': 168,  # 7 days
            'context': {
                'analysis_type': 'risk',
                'time_preference': 'recent',
                'focus_sectors': parameters.get('sectors', [])
            },
            'min_combined_score': 0.4
        })
        
        # Add risk-specific keywords to query
        original_query = parameters.get('query', '')
        risk_keywords = ' '.join(self.financial_keywords['risk'][:5])
        enhanced_query = f"{original_query} {risk_keywords}"
        enhanced_params['query'] = enhanced_query
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _news_sentiment_retrieval(self, parameters: Dict[str, Any]) -> AgentResult:
        """Specialized retrieval for news sentiment analysis"""
        enhanced_params = parameters.copy()
        enhanced_params.update({
            'required_doc_types': ['financial_news', 'market_commentary', 'social_media'],
            'freshness_weight': 0.7,
            'time_window_hours': 72,  # 3 days
            'context': {
                'analysis_type': 'sentiment',
                'time_preference': 'recent',
                'preferred_doc_types': ['financial_news', 'market_commentary']
            },
            'min_combined_score': 0.3,
            'max_results': parameters.get('max_results', 20)  # More results for sentiment
        })
        
        # Add market sentiment keywords to query
        original_query = parameters.get('query', '')
        market_keywords = ' '.join(self.financial_keywords['market'][:5])
        enhanced_query = f"{original_query} {market_keywords}"
        enhanced_params['query'] = enhanced_query
        
        return await self._retrieve_similar_chunks(enhanced_params)
    
    async def _calculate_confidence_score(self, parameters: Dict[str, Any]) -> AgentResult:
        """Calculate confidence score for given chunks"""
        chunks = parameters.get('chunks', [])
        query_params = parameters.get('query_params', {})
        
        if not chunks:
            return AgentResult(
                success=True,
                data={'confidence_score': 0.0, 'factors': {}}
            )
        
        try:
            # Create a mock query object for confidence calculation
            mock_query = RetrievalQuery(
                query_text=query_params.get('query', ''),
                context=query_params.get('context', {}),
                **{k: v for k, v in query_params.items() if k in [
                    'max_results', 'freshness_weight', 'relevance_weight', 
                    'similarity_weight', 'min_similarity_score', 'min_combined_score'
                ]}
            )
            
            confidence = self._calculate_retrieval_confidence(chunks, mock_query)
            
            # Calculate individual factors for transparency
            factors = {
                'avg_composite_score': np.mean([chunk.get('composite_score', 0) for chunk in chunks]),
                'source_diversity': len(set(chunk.get('source', 'unknown') for chunk in chunks)),
                'doc_type_diversity': len(set(chunk.get('doc_type', 'unknown') for chunk in chunks)),
                'avg_credibility': np.mean([chunk.get('source_credibility_score', 0.5) for chunk in chunks]),
                'recent_content_ratio': sum(
                    1 for chunk in chunks 
                    if self._is_recent_content(chunk, hours=24)
                ) / len(chunks)
            }
            
            return AgentResult(
                success=True,
                data={
                    'confidence_score': confidence,
                    'factors': factors,
                    'chunk_count': len(chunks)
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _is_recent_content(self, chunk: Dict[str, Any], hours: int = 24) -> bool:
        """Check if content is recent"""
        try:
            chunk_time = datetime.fromisoformat(chunk['timestamp'])
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return chunk_time >= cutoff_time
        except:
            return False
    
    async def _filter_results(self, parameters: Dict[str, Any]) -> AgentResult:
        """Filter chunks based on various criteria"""
        chunks = parameters.get('chunks', [])
        filters = parameters.get('filters', {})
        
        if not chunks:
            return AgentResult(success=True, data={'filtered_chunks': []})
        
        try:
            filtered_chunks = []
            
            for chunk in chunks:
                # Apply all filters
                if self._passes_filters(chunk, filters):
                    filtered_chunks.append(chunk)
            
            return AgentResult(
                success=True,
                data={
                    'filtered_chunks': filtered_chunks,
                    'original_count': len(chunks),
                    'filtered_count': len(filtered_chunks),
                    'filters_applied': filters
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def _passes_filters(self, chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if chunk passes all specified filters"""
        # Score filters
        if 'min_composite_score' in filters:
            if chunk.get('composite_score', 0) < filters['min_composite_score']:
                return False
        
        if 'min_similarity_score' in filters:
            if chunk.get('similarity_score', 0) < filters['min_similarity_score']:
                return False
        
        if 'min_credibility_score' in filters:
            if chunk.get('source_credibility_score', 0) < filters['min_credibility_score']:
                return False
        
        # Time filters
        if 'max_age_hours' in filters:
            if not self._is_recent_content(chunk, filters['max_age_hours']):
                return False
        
        # Content filters
        if 'required_keywords' in filters:
            content_lower = chunk.get('content', '').lower()
            for keyword in filters['required_keywords']:
                if keyword.lower() not in content_lower:
                    return False
        
        if 'excluded_keywords' in filters:
            content_lower = chunk.get('content', '').lower()
            for keyword in filters['excluded_keywords']:
                if keyword.lower() in content_lower:
                    return False
        
        # Metadata filters
        if 'required_sources' in filters:
            if chunk.get('source') not in filters['required_sources']:
                return False
        
        if 'excluded_sources' in filters:
            if chunk.get('source') in filters['excluded_sources']:
                return False
        
        if 'required_doc_types' in filters:
            if chunk.get('doc_type') not in filters['required_doc_types']:
                return False
        
        return True
    
    async def _rank_results(self, parameters: Dict[str, Any]) -> AgentResult:
        """Rank chunks by specified criteria"""
        chunks = parameters.get('chunks', [])
        ranking_method = parameters.get('ranking_method', 'composite_score')
        reverse = parameters.get('reverse', True)
        
        if not chunks:
            return AgentResult(success=True, data={'ranked_chunks': []})
        
        try:
            if ranking_method == 'composite_score':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('composite_score', 0),
                    reverse=reverse
                )
            elif ranking_method == 'similarity_score':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('similarity_score', 0),
                    reverse=reverse
                )
            elif ranking_method == 'freshness_score':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('freshness_score', 0),
                    reverse=reverse
                )
            elif ranking_method == 'credibility_score':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('source_credibility_score', 0),
                    reverse=reverse
                )
            elif ranking_method == 'timestamp':
                ranked_chunks = sorted(
                    chunks,
                    key=lambda x: x.get('timestamp', '1970-01-01T00:00:00'),
                    reverse=reverse
                )
            else:
                return AgentResult(
                    success=False,
                    error=f"Unknown ranking method: {ranking_method}"
                )
            
            return AgentResult(
                success=True,
                data={
                    'ranked_chunks': ranked_chunks,
                    'ranking_method': ranking_method,
                    'chunk_count': len(ranked_chunks)
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    async def _get_retrieval_stats(self, parameters: Dict[str, Any]) -> AgentResult:
        """Get statistics about retrieval performance and data"""
        try:
            # Get stats from embedding agent if available
            embedding_stats = {}
            if self.embedding_agent:
                stats_task = Task(
                    id=f"stats_{datetime.now().timestamp()}",
                    type='get_stats',
                    parameters={},
                    priority=TaskPriority.LOW
                )
                
                stats_result = await self.embedding_agent.execute_task(stats_task)
                if stats_result.success:
                    embedding_stats = stats_result.data
            
            # Calculate retriever-specific stats
            retriever_stats = {
                'source_credibility_mapping': len(self.source_credibility),
                'financial_keyword_categories': len(self.financial_keywords),
                'sector_mappings': len(self.sector_keywords),
                'default_weights': {
                    'freshness': self.default_freshness_weight,
                    'relevance': self.default_relevance_weight,
                    'similarity': self.default_similarity_weight
                },
                'confidence_threshold': self.confidence_threshold,
                'max_search_results': self.max_search_results
            }
            
            return AgentResult(
                success=True,
                data={
                    'retriever_stats': retriever_stats,
                    'embedding_stats': embedding_stats,
                    'capabilities': self.capabilities,
                    'dependencies': self.dependencies
                }
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get current retrieval configuration"""
        return {
            'embedding_agent_id': self.embedding_agent_id,
            'default_weights': {
                'freshness': self.default_freshness_weight,
                'relevance': self.default_relevance_weight,
                'similarity': self.default_similarity_weight
            },
            'confidence_threshold': self.confidence_threshold,
            'max_search_results': self.max_search_results,
            'source_credibility_levels': len(self.source_credibility),
            'financial_categories': list(self.financial_keywords.keys()),
            'supported_sectors': list(self.sector_keywords.keys())
        }
    
    def update_source_credibility(self, source: str, credibility: float):
        """Update credibility score for a source"""
        if 0.0 <= credibility <= 1.0:
            self.source_credibility[source.lower()] = credibility
            self.logger.info(f"Updated credibility for {source}: {credibility}")
        else:
            self.logger.warning(f"Invalid credibility score: {credibility}")
    
    def add_financial_keywords(self, category: str, keywords: List[str]):
        """Add keywords to a financial category"""
        if category not in self.financial_keywords:
            self.financial_keywords[category] = []
        
        self.financial_keywords[category].extend(keywords)
        self.logger.info(f"Added {len(keywords)} keywords to category {category}")
    
    def add_sector_keywords(self, sector: str, keywords: List[str]):
        """Add keywords to a sector mapping"""
        if sector not in self.sector_keywords:
            self.sector_keywords[sector] = []
        
        self.sector_keywords[sector].extend(keywords)
        self.logger.info(f"Added {len(keywords)} keywords to sector {sector}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the retriever agent"""
        health_status = {
            'agent_id': self.agent_id,
            'status': 'healthy',
            'embedding_agent_connected': self.embedding_agent is not None,
            'capabilities_count': len(self.capabilities),
            'source_credibility_entries': len(self.source_credibility),
            'financial_keyword_categories': len(self.financial_keywords),
            'sector_mappings': len(self.sector_keywords),
            'timestamp': datetime.now().isoformat()
        }
        
        # Test embedding agent connection if available
        if self.embedding_agent:
            try:
                test_task = Task(
                    id=f"health_check_{datetime.now().timestamp()}",
                    type='health_check',
                    parameters={},
                    priority=TaskPriority.LOW
                )
                
                result = await self.embedding_agent.execute_task(test_task)
                health_status['embedding_agent_healthy'] = result.success
                
            except Exception as e:
                health_status['embedding_agent_healthy'] = False
                health_status['embedding_agent_error'] = str(e)
        
        return health_status

# Example usage and testing functions
async def example_usage():
    """Example usage of the RetrieverAgent"""
    
    # Initialize the agent
    config = {
        'embedding_agent_id': 'embedding_agent',
        'confidence_threshold': 0.7,
        'max_search_results': 50
    }
    
    retriever = RetrieverAgent('retriever_agent', config)
    
    # Example 1: Basic retrieval
    basic_task = Task(
        id='basic_retrieval_1',
        type='retrieve_similar_chunks',
        parameters={
            'query': 'Apple quarterly earnings performance',
            'max_results': 10,
            'context': {
                'portfolio_tickers': ['AAPL'],
                'time_preference': 'recent'
            }
        },
        priority=TaskPriority.HIGH
    )
    
    print("Basic retrieval example:")
    # result = await retriever.execute_task(basic_task)
    # print(f"Success: {result.success}")
    
    # Example 2: Portfolio analysis retrieval
    portfolio_task = Task(
        id='portfolio_analysis_1',
        type='portfolio_analysis_retrieval',
        parameters={
            'query': 'technology sector performance analysis',
            'portfolio_tickers': ['AAPL', 'MSFT', 'GOOGL'],
            'analysis_type': 'performance',
            'max_results': 15
        },
        priority=TaskPriority.HIGH
    )
    
    print("\nPortfolio analysis retrieval example:")
    # result = await retriever.execute_task(portfolio_task)
    # print(f"Success: {result.success}")
    
    # Example 3: Advanced search with multiple queries
    advanced_task = Task(
        id='advanced_search_1',
        type='advanced_search',
        parameters={
            'primary_query': 'semiconductor industry outlook',
            'secondary_queries': ['chip shortage impact', 'AI chip demand'],
            'fusion_method': 'rank_fusion',
            'max_results': 12
        },
        priority=TaskPriority.HIGH
    )
    
    print("\nAdvanced search example:")
    # result = await retriever.execute_task(advanced_task)
    # print(f"Success: {result.success}")
    
    # Example 4: Get retrieval statistics
    stats_task = Task(
        id='get_stats_1',
        type='get_retrieval_stats',
        parameters={},
        priority=TaskPriority.LOW
    )
    
    print("\nRetrieval statistics example:")
    # result = await retriever.execute_task(stats_task)
    # print(f"Success: {result.success}")
    
    print("\nRetriever configuration:")
    config = retriever.get_retrieval_config()
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())