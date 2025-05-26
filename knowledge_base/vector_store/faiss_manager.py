import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import json
import os

class FAISSVectorStore:
    """Manages FAISS vector store for financial documents"""
    
    def __init__(self, 
                 embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_path: str = 'knowledge_base/vector_store/faiss_index',
                 metadata_path: str = 'knowledge_base/vector_store/metadata.json'):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100):
        """Add documents to the vector store"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract text content for embedding
            texts = [self._extract_text_content(doc) for doc in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            for j, doc in enumerate(batch):
                doc_metadata = {
                    'id': len(self.metadata),
                    'text': texts[j],
                    'source': doc.get('source', 'unknown'),
                    'content_type': doc.get('content_type', 'unknown'),
                    'timestamp': doc.get('timestamp', ''),
                    'symbol': doc.get('symbol', ''),
                    'tags': doc.get('tags', []),
                    'metadata': doc
                }
                self.metadata.append(doc_metadata)
        
        self._save_index()
    
    def search(self, 
           query: str, 
           k: int = 10, 
           filter_by: Optional[Dict] = None,
           score_threshold: float = 0.3) -> List[Tuple[Dict, float]]:  # Lowered threshold
        """Search for similar documents with enhanced query processing"""
        
        # Enhance query for better matching
        enhanced_query = self._enhance_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([enhanced_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search with larger initial results
        search_k = min(k * 3, len(self.metadata))  # Get more candidates
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:
                doc_metadata = self.metadata[idx]
                
                # Apply filters if specified
                if filter_by and not self._matches_filter(doc_metadata, filter_by):
                    continue
                
                results.append((doc_metadata, float(score)))
                
                if len(results) >= k:
                    break
        
        return results

    def _enhance_query(self, query: str) -> str:
        """Advanced query enhancement with semantic expansion"""
        enhanced_terms = [query]
        query_lower = query.lower()
        
        # Financial domain expansions
        financial_expansions = {
            'earnings': ['quarterly results', 'financial performance', 'EPS', 'revenue', 'profit', 'income statement'],
            'market': ['stock market', 'trading', 'shares', 'equity', 'securities', 'investment'],
            'tech': ['technology', 'semiconductor', 'innovation', 'digital', 'software', 'hardware'],
            'asia': ['asian markets', 'apac', 'pacific rim', 'far east', 'emerging markets'],
            'financial news': ['market updates', 'business news', 'economic news', 'corporate news', 'investor relations'],
            'growth': ['expansion', 'increase', 'improvement', 'development', 'progress'],
            'performance': ['results', 'metrics', 'KPIs', 'financials', 'outcomes']
        }
        
        # Add semantic expansions
        for term, expansions in financial_expansions.items():
            if term in query_lower:
                enhanced_terms.extend(expansions[:3])
        
        # Company-specific expansions
        company_expansions = {
            'tsmc': ['Taiwan Semiconductor', 'TSM', 'chip manufacturer', 'foundry'],
            'samsung': ['Samsung Electronics', '005930.KS', 'Korean tech', 'memory chips'],
            'alibaba': ['BABA', 'Chinese ecommerce', 'cloud computing'],
            'softbank': ['9984.T', 'Japanese conglomerate', 'Vision Fund']
        }
        
        for company, expansions in company_expansions.items():
            if company in query_lower:
                enhanced_terms.extend(expansions)
        
        return ' '.join(enhanced_terms)
        
    def _extract_text_content(self, doc: Dict) -> str:
        """Create comprehensive searchable text with multiple contexts"""
        contexts = []
        
        # Primary content with weighted importance
        if 'title' in doc:
            contexts.append(f"TITLE: {doc['title']}")  # Weighted marker
        
        if 'summary' in doc and doc['summary'].strip():
            contexts.append(f"SUMMARY: {doc['summary']}")
        
        if 'content' in doc and doc['content'].strip():
            contexts.append(f"CONTENT: {doc['content']}")
        
        # Enhanced metadata context
        content_type = doc.get('content_type', '')
        if content_type == 'market_data':
            contexts.append("CATEGORY: stock market data financial metrics trading information")
        elif content_type == 'earnings_data':
            contexts.append("CATEGORY: earnings report quarterly results financial performance revenue profit")
        elif content_type == 'news_article':
            contexts.append("CATEGORY: financial news market news business news economic updates")
        
        # Geographic and sector context
        symbol = doc.get('symbol', '')
        if symbol in ['TSM', '005930.KS', 'BABA', 'TCEHY', '6758.T', '9984.T']:
            contexts.append("REGION: Asia Asian markets APAC technology sector")
            
        # Company-specific context
        company_context = self._get_company_context(doc)
        if company_context:
            contexts.append(f"COMPANY: {company_context}")
        
        return ' '.join(contexts)

    def _get_company_context(self, doc: Dict) -> str:
        """Get rich company context for better matching"""
        symbol = doc.get('symbol', '')
        company_contexts = {
            'TSM': 'Taiwan Semiconductor Manufacturing Company TSMC chip foundry semiconductor leader',
            '005930.KS': 'Samsung Electronics Samsung Korean technology memory semiconductors smartphones',
            'BABA': 'Alibaba Group Chinese ecommerce cloud computing digital economy',
            'TCEHY': 'Tencent Holdings Chinese internet gaming social media WeChat',
            '6758.T': 'Sony Corporation Japanese electronics entertainment gaming PlayStation',
            '9984.T': 'SoftBank Group Japanese conglomerate Vision Fund telecommunications'
        }
        return company_contexts.get(symbol, '')
        
    def _matches_filter(self, doc_metadata: Dict, filter_by: Dict) -> bool:
        """Check if document matches filter criteria"""
        for key, value in filter_by.items():
            if key == 'content_type':
                if doc_metadata.get('content_type') != value:
                    return False
            elif key == 'source':
                if doc_metadata.get('source') != value:
                    return False
            elif key == 'tags':
                doc_tags = doc_metadata.get('tags', [])
                if not any(tag in doc_tags for tag in value):
                    return False
        return True
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(f"{self.index_path}.index"):
                self.index = faiss.read_index(f"{self.index_path}.index")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")

    def hybrid_search(self, 
                 query: str, 
                 k: int = 10,
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3) -> List[Tuple[Dict, float]]:
        """Combine semantic and keyword search for better results"""
        
        # Get semantic search results
        semantic_results = self.search(query, k=k*2)
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, k=k*2)
        
        # Combine and rerank results
        combined_scores = {}
        
        for doc, score in semantic_results:
            doc_id = doc['id']
            combined_scores[doc_id] = {
                'doc': doc,
                'semantic_score': score,
                'keyword_score': 0.0
            }
        
        for doc, score in keyword_results:
            doc_id = doc['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['keyword_score'] = score
            else:
                combined_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0.0,
                    'keyword_score': score
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            hybrid_score = (semantic_weight * scores['semantic_score'] + 
                        keyword_weight * scores['keyword_score'])
            final_results.append((scores['doc'], hybrid_score))
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def _keyword_search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Simple keyword-based search for hybrid approach"""
        query_terms = set(query.lower().split())
        results = []
        
        for doc in self.metadata:
            text_content = doc['text'].lower()
            # Calculate simple keyword overlap score
            doc_terms = set(text_content.split())
            overlap = len(query_terms.intersection(doc_terms))
            if overlap > 0:
                score = overlap / len(query_terms)
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def adaptive_search(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Adapt search strategy based on query type"""
        query_type = self._classify_query(query)
        
        if query_type == 'broad_topic':
            # Use hybrid search with lower semantic weight
            return self.hybrid_search(query, k, semantic_weight=0.5, keyword_weight=0.5)
        elif query_type == 'specific_company':
            # Use higher similarity threshold and company filtering
            return self.search(query, k, score_threshold=0.4)
        elif query_type == 'financial_metric':
            # Focus on earnings and market data
            filter_by = {'content_type': ['earnings_data', 'market_data']}
            return self.search(query, k, filter_by=filter_by)
        else:
            # Default semantic search
            return self.search(query, k)

    def _classify_query(self, query: str) -> str:
        """Classify query type for adaptive search"""
        query_lower = query.lower()
        
        # Check for specific companies
        companies = ['tsm', 'tsmc', 'samsung', 'alibaba', 'baba', 'sony', 'softbank']
        if any(company in query_lower for company in companies):
            return 'specific_company'
        
        # Check for financial metrics
        metrics = ['earnings', 'revenue', 'profit', 'eps', 'pe ratio', 'market cap']
        if any(metric in query_lower for metric in metrics):
            return 'financial_metric'
        
        # Check for broad topics
        broad_terms = ['financial news', 'market', 'asia tech', 'technology']
        if any(term in query_lower for term in broad_terms):
            return 'broad_topic'
        
        return 'general'