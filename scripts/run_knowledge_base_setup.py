import asyncio
import logging
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.config.kb_config import get_knowledge_base_config
from data_ingestion.pipeline_orchestrator import DataIngestionPipeline

async def main():
    """Initialize and populate the knowledge base"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_knowledge_base_config()
        
        # Initialize pipeline
        logger.info("Initializing data ingestion pipeline...")
        pipeline = DataIngestionPipeline(config)
        
        # Run initial data collection
        logger.info("Starting initial knowledge base population...")
        results = await pipeline.run_full_pipeline()
        
        logger.info(f"Knowledge base setup completed: {results}")
        
        # Enhanced testing with multiple search queries
        await test_search_functionality(pipeline, logger)
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise

async def test_search_functionality(pipeline, logger):
    """Test search functionality with multiple queries"""
    
    search_queries = [
        "TSMC earnings surprise semiconductor",
        "market data stock price", 
        "Asia tech companies",
        "financial news",
        "TSM quarterly results",
        "Samsung earnings"
    ]
    
    logger.info("="*50)
    logger.info("TESTING SEARCH FUNCTIONALITY")
    logger.info("="*50)
    
    for i, query in enumerate(search_queries, 1):
        logger.info(f"\n[Test {i}/{len(search_queries)}] Testing search: '{query}'")
        
        try:
            search_results = pipeline.vector_store.search(query, k=3)
            logger.info(f"Found {len(search_results)} results")
            
            if search_results:
                for j, (doc, score) in enumerate(search_results, 1):
                    # Enhanced title extraction
                    title = extract_document_title(doc)
                    content_type = doc.get('content_type', 'unknown')
                    source = doc.get('source', 'unknown')
                    
                    logger.info(f"  [{j}] {title}")
                    logger.info(f"      Score: {score:.3f} | Type: {content_type} | Source: {source}")
                    
                    # Show a snippet of content for debugging
                    content = doc.get('content', doc.get('text', ''))
                    if content:
                        snippet = content[:100].replace('\n', ' ').strip()
                        logger.info(f"      Content: {snippet}...")
            else:
                logger.warning(f"No results found for query: '{query}'")
                
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
        
        logger.info("-" * 30)

def extract_document_title(doc):
    """Extract title from document with multiple fallback options"""
    
    # Try multiple title extraction methods
    title_candidates = [
        doc.get('title'),
        doc.get('metadata', {}).get('title') if isinstance(doc.get('metadata'), dict) else None,
        f"{doc.get('symbol', '')} {doc.get('content_type', 'Document')}".strip(),
        f"{doc.get('source', 'Unknown')} Document"
    ]
    
    # Return first non-empty title
    for title in title_candidates:
        if title and title.strip():
            return title.strip()
    
    return "No Title Available"

async def run_diagnostic_checks(pipeline, logger):
    """Run diagnostic checks on the knowledge base"""
    
    logger.info("\n" + "="*50)
    logger.info("RUNNING DIAGNOSTIC CHECKS")
    logger.info("="*50)
    
    try:
        # Check vector store status
        logger.info("Checking vector store status...")
        
        # Get total document count (if the vector store supports it)
        try:
            total_docs = getattr(pipeline.vector_store, 'get_document_count', lambda: 'Unknown')()
            logger.info(f"Total documents in vector store: {total_docs}")
        except:
            logger.info("Could not retrieve document count")
        
        # Test empty search
        logger.info("Testing with empty/broad search...")
        broad_results = pipeline.vector_store.search("", k=5)
        logger.info(f"Broad search returned {len(broad_results)} results")
        
        # Show document types distribution
        if broad_results:
            doc_types = {}
            sources = {}
            
            for doc, _ in broad_results:
                content_type = doc.get('content_type', 'unknown')
                source = doc.get('source', 'unknown')
                
                doc_types[content_type] = doc_types.get(content_type, 0) + 1
                sources[source] = sources.get(source, 0) + 1
            
            logger.info(f"Document types found: {dict(doc_types)}")
            logger.info(f"Sources found: {dict(sources)}")
        
    except Exception as e:
        logger.error(f"Diagnostic checks failed: {e}")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())