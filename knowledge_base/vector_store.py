import faiss
import numpy as np
import pickle
from typing import List, Dict
import logging

class VectorStore:
    """Manages FAISS vector store for document embeddings."""
    
    def __init__(self, dimension: int = 384, index_type: str = 'flat'):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.documents = []
        self.logger = logging.getLogger(__name__)
    
    def _create_index(self):
        """Create FAISS index based on type."""
        if self.index_type == 'flat':
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            return faiss.IndexFlatIP(self.dimension)
    
    def add_documents(self, processed_documents: List[Dict]):
        """Add processed documents with embeddings to vector store."""
        if not processed_documents:
            return
        
        embeddings = []
        for doc in processed_documents:
            embedding = np.array(doc['embedding'], dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        self.index.add(embeddings_array)
        
        for doc in processed_documents:
            doc_meta = doc.copy()
            doc_meta.pop('embedding', None)
            self.documents.append(doc_meta)
        
        self.logger.info(f"Added {len(processed_documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """Save index and metadata."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def load(self, index_path: str, metadata_path: str):
        """Load index and metadata."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.documents = pickle.load(f)
    
    def get_document_count(self) -> int:
        """Return the number of documents in the store."""
        return self.index.ntotal