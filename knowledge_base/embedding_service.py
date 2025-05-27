from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

class EmbeddingService:
    """Handles embedding generation for documents and queries."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
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
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return np.array([])