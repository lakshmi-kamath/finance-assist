import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

def get_knowledge_base_config() -> Dict:
    """Get knowledge base configuration"""
    return {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY', 'demo'),
        'fred_api_key': os.getenv('FRED_API_KEY', 'demo'),
        'vector_store': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_path': 'knowledge_base/vector_store/faiss_index',
            'metadata_path': 'knowledge_base/vector_store/metadata.json',
            'max_documents': 10000,
            'similarity_threshold': 0.5
        },
        'data_collection': {
            'news_sources': [
                'reuters_finance',
                'marketwatch', 
                'yahoo_finance_news'
            ],
            'asia_tech_symbols': [
                'TSM', '005930.KS', 'BABA', 'TCEHY', '6758.T', 'ASML', '9984.T'
            ],
            'update_frequency_hours': 4,
            'news_lookback_hours': 24
        },
        'processing': {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'batch_size': 100
        }
    }
