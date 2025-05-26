import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingUtilities:
    """Utility functions for embedding operations"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix for embeddings"""
        return cosine_similarity(embeddings)
    
    def find_duplicate_documents(self, 
                                embeddings: np.ndarray, 
                                metadata: List[Dict], 
                                threshold: float = 0.95) -> List[Tuple[int, int, float]]:
        """Find potential duplicate documents based on embedding similarity"""
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        duplicates = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = similarity_matrix[i][j]
                if similarity >= threshold:
                    duplicates.append((i, j, similarity))
        
        return duplicates
    
    def cluster_documents(self, 
                         embeddings: np.ndarray, 
                         n_clusters: int = 10,
                         method: str = 'kmeans') -> np.ndarray:
        """Cluster documents based on embeddings"""
        try:
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                return clusters
            else:
                self.logger.warning(f"Clustering method {method} not supported")
                return np.array([])
        except ImportError:
            self.logger.error("sklearn not available for clustering")
            return np.array([])
    
    def reduce_dimensionality(self, 
                            embeddings: np.ndarray, 
                            n_components: int = 50,
                            method: str = 'pca') -> np.ndarray:
        """Reduce embedding dimensionality for visualization or storage"""
        try:
            if method == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                reduced_embeddings = pca.fit_transform(embeddings)
                return reduced_embeddings
            elif method == 'umap':
                import umap
                reducer = umap.UMAP(n_components=n_components)
                reduced_embeddings = reducer.fit_transform(embeddings)
                return reduced_embeddings
            else:
                self.logger.warning(f"Dimensionality reduction method {method} not supported")
                return embeddings
        except ImportError as e:
            self.logger.error(f"Required library not available: {e}")
            return embeddings
    
    def semantic_search_rerank(self, 
                              query_embedding: np.ndarray,
                              candidate_embeddings: np.ndarray,
                              candidate_texts: List[str],
                              top_k: int = 10) -> List[Tuple[int, float]]:
        """Rerank search results using cross-encoder for better accuracy"""
        try:
            # Calculate initial similarity scores
            similarities = cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]
            
            # Get top candidates for reranking
            initial_candidates = np.argsort(similarities)[::-1][:top_k * 2]
            
            # For now, return initial ranking (can be enhanced with cross-encoder)
            reranked_results = [(idx, similarities[idx]) for idx in initial_candidates[:top_k]]
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search reranking: {e}")
            return []
    
    def create_query_expansion(self, query: str, expansion_terms: List[str]) -> str:
        """Expand query with related financial terms"""
        financial_synonyms = {
            'earnings': ['profits', 'income', 'results', 'financial performance'],
            'revenue': ['sales', 'income', 'turnover', 'top line'],
            'growth': ['expansion', 'increase', 'rise', 'improvement'],
            'asia': ['asian', 'apac', 'pacific', 'far east'],
            'tech': ['technology', 'technological', 'digital', 'innovation']
        }
        
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in financial_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        expanded_terms.extend(expansion_terms)
        return ' '.join(expanded_terms)
