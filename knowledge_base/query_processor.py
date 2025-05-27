from typing import List, Dict

class QueryProcessor:
    """Handles query enhancement for semantic search."""
    
    def __init__(self):
        self.financial_synonyms = {
            'earnings': ['quarterly results', 'financial performance', 'EPS', 'revenue', 'profit'],
            'market': ['stock market', 'trading', 'shares', 'equity'],
            'tech': ['technology', 'semiconductor', 'software', 'hardware'],
            'asia': ['asian markets', 'apac', 'pacific rim'],
            'growth': ['expansion', 'increase', 'improvement']
        }
        self.company_synonyms = {
            'tsmc': ['Taiwan Semiconductor', 'TSM', 'chip manufacturer'],
            'samsung': ['Samsung Electronics', '005930.KS', 'Korean tech'],
            'alibaba': ['BABA', 'Chinese ecommerce'],
            'softbank': ['9984.T', 'Japanese conglomerate']
        }
    
    def enhance_query(self, query: str) -> str:
        """Enhance query with financial and company-specific terms."""
        query_lower = query.lower()
        enhanced_terms = [query]
        
        # Add financial term expansions
        for term, synonyms in self.financial_synonyms.items():
            if term in query_lower:
                enhanced_terms.extend(synonyms[:2])
        
        # Add company-specific expansions
        for company, synonyms in self.company_synonyms.items():
            if company in query_lower:
                enhanced_terms.extend(synonyms[:2])
        
        return ' '.join(enhanced_terms)
    
    def classify_query(self, query: str) -> str:
        """Classify query type for adaptive search."""
        query_lower = query.lower()
        
        companies = ['tsmc', 'samsung', 'alibaba', 'sony', 'softbank']
        if any(company in query_lower for company in companies):
            return 'specific_company'
        
        metrics = ['earnings', 'revenue', 'profit', 'eps']
        if any(metric in query_lower for metric in metrics):
            return 'financial_metric'
        
        broad_terms = ['financial news', 'market', 'asia tech']
        if any(term in query_lower for term in broad_terms):
            return 'broad_topic'
        
        return 'general'
    
    async def search(self, query: str, k: int = 5) -> List[Dict]:
        enhanced_query = self.query_processor.enhance_query(query)
        self.logger.info(f"Enhanced query: {enhanced_query}")
        query_embedding = self.embedding_service.generate_query_embedding(enhanced_query)
        results = self.vector_store.search(query_embedding, k)
        return results