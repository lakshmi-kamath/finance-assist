"""
Test script to validate knowledge base functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from knowledge_base.config.kb_config import get_knowledge_base_config
from data_ingestion.pipeline_orchestrator import DataIngestionPipeline

async def test_knowledge_base():
    """Test the complete knowledge base system"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = get_knowledge_base_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize pipeline
        pipeline = DataIngestionPipeline(config)
        logger.info("Pipeline initialized successfully")
        
        # Test individual components
        logger.info("Testing Yahoo Finance collector...")
        yahoo_data = pipeline.yahoo_finance.get_stock_info(['AAPL', 'MSFT'])
        logger.info(f"Retrieved {len(yahoo_data)} stock records")
        
        logger.info("Testing news scraper...")
        news_data = pipeline.news_scraper.scrape_rss_feeds(hours_back=1)
        logger.info(f"Retrieved {len(news_data)} news articles")
        
        # Test vector store
        logger.info("Testing vector store...")
        test_documents = [
            {
                'title': 'Test Financial Document',
                'content': 'Apple Inc reported strong quarterly earnings with revenue growth of 15%',
                'content_type': 'test_doc',
                'source': 'test',
                'timestamp': '2024-01-01T00:00:00',
                'tags': ['test', 'earnings']
            }
        ]
        
        pipeline.vector_store.add_documents(test_documents)
        
        # Test search
        search_results = pipeline.vector_store.search("Apple earnings revenue", k=5)
        logger.info(f"Search returned {len(search_results)} results")
        
        if search_results:
            for doc, score in search_results:
                logger.info(f"  - {doc.get('title', 'Unknown')} (score: {score:.3f})")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_knowledge_base())