import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/startup.log')
    ]
)

logger = logging.getLogger(__name__)

class ServiceManager:
    """Manages FastAPI server, Streamlit app, and Knowledge Base scheduler"""
    
    def __init__(self):
        self.fastapi_process: Optional[subprocess.Popen] = None
        self.streamlit_process: Optional[subprocess.Popen] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        self.services_running = False
        self.shutdown_event = threading.Event()
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Service configurations
        self.fastapi_config = {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1
        }
        
        self.streamlit_config = {
            'port': 8501,
            'host': '0.0.0.0'
        }
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required files and dependencies exist"""
        checks = {
            'fastapi_server.py': Path('fastapi_server.py').exists(),
            'streamlit_app.py': Path('streamlit_app.py').exists(),
            'scheduler.py': Path('scheduler.py').exists(),
            'main.py': Path('main.py').exists(),
            'api_directory': Path('api').exists(),
            'utils_directory': Path('utils').exists(),
            'orchestrator_directory': Path('orchestrator').exists(),
            'logs_directory': Path('logs').exists(),
            'env_file': Path('.env').exists()
        }
        
        missing = [name for name, exists in checks.items() if not exists]
        
        if missing:
            logger.warning(f"Missing dependencies: {missing}")
        else:
            logger.info("All dependencies found")
        
        return checks
    
    def check_ports(self) -> Dict[str, bool]:
        """Check if required ports are available"""
        def is_port_available(port: int) -> bool:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return False
            return True
        
        ports = {
            'fastapi_8000': is_port_available(8000),
            'streamlit_8501': is_port_available(8501)
        }
        
        for service, available in ports.items():
            if not available:
                logger.warning(f"Port for {service} is already in use")
            
        return ports
    
    def start_fastapi_server(self) -> bool:
        """Start FastAPI server"""
        try:
            logger.info("Starting FastAPI server...")
            
            # Start FastAPI with uvicorn
            cmd = [
                sys.executable, '-m', 'uvicorn',
                'fastapi_server:app',
                '--host', self.fastapi_config['host'],
                '--port', str(self.fastapi_config['port']),
                '--workers', str(self.fastapi_config['workers']),
                '--log-level', 'info'
            ]
            
            self.fastapi_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait a moment and check if process started successfully
            time.sleep(3)
            
            if self.fastapi_process.poll() is None:
                logger.info(f"FastAPI server started successfully on port {self.fastapi_config['port']}")
                
                # Start log monitoring thread
                threading.Thread(
                    target=self._monitor_process_logs,
                    args=(self.fastapi_process, "FastAPI"),
                    daemon=True
                ).start()
                
                return True
            else:
                logger.error("FastAPI server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting FastAPI server: {e}")
            return False
    
    def start_streamlit_app(self) -> bool:
        """Start Streamlit application"""
        try:
            logger.info("Starting Streamlit application...")
            
            # Start Streamlit
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                'streamlit_app.py',
                '--server.port', str(self.streamlit_config['port']),
                '--server.address', self.streamlit_config['host'],
                '--server.headless', 'true',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'false'
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait a moment and check if process started successfully
            time.sleep(5)
            
            if self.streamlit_process.poll() is None:
                logger.info(f"Streamlit application started successfully on port {self.streamlit_config['port']}")
                
                # Start log monitoring thread
                threading.Thread(
                    target=self._monitor_process_logs,
                    args=(self.streamlit_process, "Streamlit"),
                    daemon=True
                ).start()
                
                return True
            else:
                logger.error("Streamlit application failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Streamlit application: {e}")
            return False
    
    def start_scheduler(self) -> bool:
        """Start Knowledge Base scheduler"""
        try:
            logger.info("Starting Knowledge Base scheduler...")
            
            from scheduler import start_scheduler
            
            # Start scheduler in separate thread
            def run_scheduler():
                try:
                    scheduler = start_scheduler()
                    logger.info("Knowledge Base scheduler started successfully")
                    
                    # Keep scheduler thread alive
                    while not self.shutdown_event.is_set():
                        time.sleep(10)
                        
                except Exception as e:
                    logger.error(f"Error in scheduler thread: {e}")
            
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            time.sleep(2)  # Give scheduler time to start
            return True
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            return False
    
    def _monitor_process_logs(self, process: subprocess.Popen, service_name: str):
        """Monitor process logs and forward to main logger"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"[{service_name}] {line.strip()}")
                    
                if process.poll() is not None:
                    break
                    
        except Exception as e:
            logger.error(f"Error monitoring {service_name} logs: {e}")
    
    def wait_for_services(self, timeout: int = 60) -> Dict[str, bool]:
        """Wait for services to be ready"""
        logger.info("Waiting for services to be ready...")
        
        start_time = time.time()
        services_status = {
            'fastapi': False,
            'streamlit': False,
            'scheduler': False
        }
        
        while time.time() - start_time < timeout:
            # Check FastAPI
            if not services_status['fastapi']:
                try:
                    response = requests.get(f"http://localhost:{self.fastapi_config['port']}/health", timeout=5)
                    if response.status_code == 200:
                        services_status['fastapi'] = True
                        logger.info("FastAPI service is ready")
                except requests.exceptions.RequestException:
                    pass
            
            # Check Streamlit (just check if process is running)
            if not services_status['streamlit'] and self.streamlit_process:
                if self.streamlit_process.poll() is None:
                    services_status['streamlit'] = True
                    logger.info("Streamlit service is ready")
            
            # Check Scheduler
            if not services_status['scheduler'] and self.scheduler_thread:
                if self.scheduler_thread.is_alive():
                    services_status['scheduler'] = True
                    logger.info("Scheduler service is ready")
            
            # All services ready?
            if all(services_status.values()):
                logger.info("All services are ready!")
                return services_status
            
            time.sleep(2)
        
        # Timeout reached
        not_ready = [name for name, ready in services_status.items() if not ready]
        logger.warning(f"Timeout waiting for services. Not ready: {not_ready}")
        
        return services_status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'services': {
                'fastapi': {
                    'running': self.fastapi_process is not None and self.fastapi_process.poll() is None,
                    'pid': self.fastapi_process.pid if self.fastapi_process else None,
                    'port': self.fastapi_config['port']
                },
                'streamlit': {
                    'running': self.streamlit_process is not None and self.streamlit_process.poll() is None,
                    'pid': self.streamlit_process.pid if self.streamlit_process else None,
                    'port': self.streamlit_config['port']
                },
                'scheduler': {
                    'running': self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
                    'thread_id': self.scheduler_thread.ident if self.scheduler_thread else None
                }
            },
            'system': {
                'services_running': self.services_running,
                'uptime_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0,
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            },
            'urls': {
                'fastapi_api': f"http://localhost:{self.fastapi_config['port']}",
                'fastapi_docs': f"http://localhost:{self.fastapi_config['port']}/docs",
                'streamlit_app': f"http://localhost:{self.streamlit_config['port']}"
            }
        }
    
    def start_all_services(self) -> bool:
        """Start all services"""
        logger.info("=== Starting Finance Assistant System ===")
        
        self.start_time = time.time()
        
        # Check dependencies
        deps = self.check_dependencies()
        if not all(deps.values()):
            logger.error("Missing dependencies. Cannot start services.")
            return False
        
        # Check ports
        ports = self.check_ports()
        if not all(ports.values()):
            logger.error("Required ports are not available. Cannot start services.")
            return False
        
        success_count = 0
        
        # Start FastAPI server
        if self.start_fastapi_server():
            success_count += 1
        
        # Start Streamlit app
        if self.start_streamlit_app():
            success_count += 1
        
        # Start scheduler
        if self.start_scheduler():
            success_count += 1
        
        if success_count == 3:
            self.services_running = True
            logger.info("All services started successfully!")
            
            # Wait for services to be ready
            ready_status = self.wait_for_services()
            
            # Display system information
            self.display_system_info()
            
            return True
        else:
            logger.error(f"Only {success_count}/3 services started successfully")
            return False
    
    def display_system_info(self):
        """Display system information and URLs"""
        status = self.get_system_status()
        
        logger.info("=== Finance Assistant System Ready ===")
        logger.info(f"🌐 Streamlit App: {status['urls']['streamlit_app']}")
        logger.info(f"🔧 FastAPI Docs: {status['urls']['fastapi_docs']}")
        logger.info(f"📊 API Endpoint: {status['urls']['fastapi_api']}")
        logger.info("📅 Knowledge Base Scheduler: Running (7:45 AM daily)")
        logger.info("="*50)
        
        print("\n" + "="*50)
        print("🚀 FINANCE ASSISTANT SYSTEM READY")
        print("="*50)
        print(f"📱 Web App: {status['urls']['streamlit_app']}")
        print(f"🔧 API Docs: {status['urls']['fastapi_docs']}")
        print(f"📊 API: {status['urls']['fastapi_api']}")
        print("📅 Scheduler: Running (Knowledge base updates at 7:45 AM)")
        print("="*50)
        print("Press Ctrl+C to stop all services")
        print("="*50 + "\n")
    
    def stop_all_services(self):
        """Stop all services gracefully"""
        logger.info("Stopping all services...")
        
        self.services_running = False
        self.shutdown_event.set()
        
        # Stop FastAPI
        if self.fastapi_process:
            try:
                logger.info("Stopping FastAPI server...")
                self.fastapi_process.terminate()
                self.fastapi_process.wait(timeout=10)
                logger.info("FastAPI server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("FastAPI server didn't stop gracefully, killing...")
                self.fastapi_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FastAPI: {e}")
        
        # Stop Streamlit
        if self.streamlit_process:
            try:
                logger.info("Stopping Streamlit application...")
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=10)
                logger.info("Streamlit application stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Streamlit didn't stop gracefully, killing...")
                self.streamlit_process.kill()
            except Exception as e:
                logger.error(f"Error stopping Streamlit: {e}")
        
        # Stop scheduler
        if self.scheduler_thread:
            try:
                logger.info("Stopping scheduler...")
                from scheduler import stop_scheduler
                stop_scheduler()
                self.scheduler_thread.join(timeout=5)
                logger.info("Scheduler stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
        
        logger.info("All services stopped")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if hasattr(signal_handler, 'service_manager'):
        signal_handler.service_manager.stop_all_services()
    sys.exit(0)

def main():
    """Main startup function"""
    logger.info("Finance Assistant System Startup")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create service manager
    service_manager = ServiceManager()
    signal_handler.service_manager = service_manager  # Store reference for signal handler
    
    try:
        # Start all services
        if service_manager.start_all_services():
            logger.info("System startup completed successfully")
            
            # Keep main thread alive and monitor services
            while service_manager.services_running:
                time.sleep(30)  # Check every 30 seconds
                
                # Health check
                status = service_manager.get_system_status()
                running_services = sum(1 for svc in status['services'].values() if svc['running'])
                
                if running_services < 3:
                    logger.warning(f"Only {running_services}/3 services are running")
                
                # Log periodic status
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"System status: {running_services}/3 services running, "
                              f"CPU: {status['system']['cpu_percent']:.1f}%, "
                              f"Memory: {status['system']['memory_percent']:.1f}%")
        
        else:
            logger.error("Failed to start system")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    
    except Exception as e:
        logger.error(f"Startup error: {e}")
        return 1
    
    finally:
        service_manager.stop_all_services()
        logger.info("Finance Assistant System shutdown completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())