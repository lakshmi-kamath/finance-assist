import asyncio
import logging
from knowledge_base.config import get_config
from knowledge_base.knowledge_base_manager import KnowledgeBaseManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)

async def main():
    config = get_config()
    kb_manager = KnowledgeBaseManager(config)
    
    # Check health and stats
    logging.info(f"Health check: {kb_manager.check_health()}")
    logging.info(f"Pipeline statistics: {kb_manager.get_statistics()}")
    
    # Run pipeline
    result = await kb_manager.run_pipeline()
    logging.info(f"Pipeline result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
    
    # Option 2: Schedule regular updates (uncomment to use)
