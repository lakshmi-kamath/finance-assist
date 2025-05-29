import asyncio
import logging
import schedule
import time
import threading
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Optional
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/scheduler.log')
    ]
)

logger = logging.getLogger(__name__)

class KnowledgeBaseScheduler:
    """Independent scheduler for running main.py knowledge base pipeline"""
    
    def __init__(self):
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.last_run_time: Optional[datetime] = None
        self.last_run_success: bool = False
        self.run_count = 0
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
    
    def run_knowledge_base_pipeline(self):
        """Execute the knowledge base pipeline (main.py)"""
        try:
            logger.info("Starting scheduled knowledge base pipeline execution...")
            start_time = datetime.now()
            
            # Run main.py as subprocess
            result = subprocess.run(
                [sys.executable, 'main.py'], 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            execution_time = datetime.now() - start_time
            
            if result.returncode == 0:
                logger.info(f"Knowledge base pipeline completed successfully in {execution_time}")
                logger.info(f"Pipeline output: {result.stdout}")
                self.last_run_success = True
            else:
                logger.error(f"Knowledge base pipeline failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                self.last_run_success = False
            
            self.last_run_time = datetime.now()
            self.run_count += 1
            
            # Log execution summary
            logger.info(f"Pipeline execution #{self.run_count} completed. Success: {self.last_run_success}")
            
        except subprocess.TimeoutExpired:
            logger.error("Knowledge base pipeline execution timed out after 30 minutes")
            self.last_run_success = False
            self.last_run_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error executing knowledge base pipeline: {e}")
            self.last_run_success = False
            self.last_run_time = datetime.now()
    
    def health_check(self):
        """Perform scheduler health check"""
        try:
            # Check if main.py exists
            main_py_exists = Path('main.py').exists()
            
            # Check if logs directory is writable
            logs_writable = os.access('logs', os.W_OK)
            
            # Check scheduler status
            scheduler_healthy = self.is_running
            
            health_status = {
                'scheduler_running': scheduler_healthy,
                'main_py_exists': main_py_exists,
                'logs_writable': logs_writable,
                'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
                'last_run_success': self.last_run_success,
                'total_runs': self.run_count,
                'next_run': self.get_next_run_time()
            }
            
            overall_healthy = all([scheduler_healthy, main_py_exists, logs_writable])
            
            logger.info(f"Scheduler health check: {'Healthy' if overall_healthy else 'Issues detected'}")
            return {
                'healthy': overall_healthy,
                'status': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_next_run_time(self) -> Optional[str]:
        """Get the next scheduled run time"""
        try:
            jobs = schedule.jobs
            if jobs:
                next_run = min(job.next_run for job in jobs)
                return next_run.isoformat()
            return None
        except Exception:
            return None
    
    def get_statistics(self):
        """Get scheduler statistics"""
        uptime = datetime.now() - self.start_time if hasattr(self, 'start_time') else timedelta(0)
        
        return {
            'scheduler_uptime_hours': uptime.total_seconds() / 3600,
            'total_pipeline_runs': self.run_count,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'last_run_success': self.last_run_success,
            'success_rate': f"{(sum(1 for _ in range(self.run_count) if self.last_run_success) / max(self.run_count, 1)) * 100:.1f}%" if self.run_count > 0 else "N/A",
            'next_scheduled_run': self.get_next_run_time(),
            'is_running': self.is_running
        }
    
    def start_scheduler(self):
        """Start the scheduler in a separate thread"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting knowledge base scheduler...")
        self.start_time = datetime.now()
        
        # Schedule daily execution at 7:45 AM
        schedule.every().day.at("07:45").do(self.run_knowledge_base_pipeline)
        
        # Optional: Add a weekly full pipeline run (can be uncommented)
        # schedule.every().sunday.at("06:00").do(self.run_knowledge_base_pipeline)
        
        self.is_running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Knowledge base scheduler started successfully")
        logger.info(f"Next scheduled run: {self.get_next_run_time()}")
    
    def _run_scheduler(self):
        """Internal scheduler loop"""
        logger.info("Scheduler thread started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Continue running even if there's an error
        
        logger.info("Scheduler thread stopped")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not currently running")
            return
        
        logger.info("Stopping knowledge base scheduler...")
        self.is_running = False
        
        # Clear scheduled jobs
        schedule.clear()
        
        # Wait for thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Knowledge base scheduler stopped")
    
    def run_pipeline_now(self):
        """Manually trigger pipeline execution"""
        logger.info("Manually triggering knowledge base pipeline...")
        threading.Thread(target=self.run_knowledge_base_pipeline, daemon=True).start()
    
    def get_status(self):
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'last_run_success': self.last_run_success,
            'run_count': self.run_count,
            'next_run_time': self.get_next_run_time(),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if hasattr(self, 'start_time') else 0
        }

# Global scheduler instance
_scheduler_instance: Optional[KnowledgeBaseScheduler] = None

def get_scheduler() -> KnowledgeBaseScheduler:
    """Get or create scheduler instance"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = KnowledgeBaseScheduler()
    return _scheduler_instance

def start_scheduler():
    """Start the global scheduler"""
    scheduler = get_scheduler()
    scheduler.start_scheduler()
    return scheduler

def stop_scheduler():
    """Stop the global scheduler"""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop_scheduler()

def main():
    """Main function for running scheduler standalone"""
    logger.info("Starting Knowledge Base Scheduler as standalone service...")
    
    try:
        # Create and start scheduler
        scheduler = start_scheduler()
        
        logger.info("Scheduler is running. Press Ctrl+C to stop.")
        logger.info(f"Pipeline scheduled to run daily at 7:45 AM")
        logger.info(f"Next run: {scheduler.get_next_run_time()}")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(10)
                
                # Periodic status logging (every 10 minutes)
                if int(time.time()) % 600 == 0:
                    status = scheduler.get_status()
                    logger.info(f"Scheduler status: Running={status['is_running']}, "
                              f"Total runs={status['run_count']}, "
                              f"Last success={status['last_run_success']}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping scheduler...")
            
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        raise
    
    finally:
        stop_scheduler()
        logger.info("Knowledge Base Scheduler stopped")

if __name__ == "__main__":
    main()