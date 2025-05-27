from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents for vector storage."""
        chunked_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            if len(content) < self.chunk_size:
                chunked_docs.append(doc)
            else:
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_doc = doc.copy()
                    chunk_doc['content'] = chunk
                    chunk_doc['chunk_id'] = f"{doc.get('symbol', 'unknown')}_{i}"
                    chunk_doc['total_chunks'] = len(chunks)
                    chunk_doc['chunk_index'] = i
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs