#!/usr/bin/env python3
"""
Smart Second Brain - System Startup Script
Starts all components: Backend API, Frontend, and Monitoring
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('system_startup.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemManager:
    """Manages all system components"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.processes: Dict[str, subprocess.Popen] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = True
        
        # Component configurations
        self.components = {
            'backend': {
                'name': 'Backend API',
                'port': 8000,
                'health_url': 'http://localhost:8000/smart-second-brain/api/v1/graph/health',
                'command': [
                    sys.executable, '-m', 'uvicorn', 'api.main:app',
                    '--host', '0.0.0.0', '--port', '8000', '--reload'
                ],
                'cwd': str(self.root_dir),
                'env': {
                    'PYTHONPATH': str(self.root_dir),
                    **os.environ
                }
            },
            'frontend': {
                'name': 'Frontend (Streamlit)',
                'port': 5173,
                'health_url': 'http://localhost:5173',
                'command': [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '5173', '--server.address', '0.0.0.0'],
                'cwd': str(self.root_dir / 'frontend'),
                'env': os.environ.copy()
            }
        }
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop_all()
        sys.exit(0)
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def wait_for_service(self, name: str, url: str, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        import requests
        
        logger.info(f"Waiting for {name} to become available...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {name} is ready!")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(1)
            if time.time() - start_time < timeout:
                logger.info(f"⏳ Still waiting for {name}... ({int(timeout - (time.time() - start_time))}s left)")
        
        logger.error(f"❌ {name} failed to start within {timeout} seconds")
        return False
    
    def start_component(self, component_id: str) -> bool:
        """Start a single component"""
        component = self.components[component_id]
        name = component['name']
        port = component['port']
        
        # Check if port is available
        if not self.check_port_available(port):
            logger.error(f"❌ Port {port} is already in use for {name}")
            return False
        
        logger.info(f"🚀 Starting {name} on port {port}...")
        
        try:
            # Start the process
            process = subprocess.Popen(
                component['command'],
                cwd=component['cwd'],
                env=component['env'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[component_id] = process
            logger.info(f"✅ {name} process started (PID: {process.pid})")
            
            # Wait for service to be ready
            if self.wait_for_service(name, component['health_url']):
                return True
            else:
                self.stop_component(component_id)
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return False
    
    def stop_component(self, component_id: str):
        """Stop a single component"""
        if component_id in self.processes:
            process = self.processes[component_id]
            name = self.components[component_id]['name']
            
            logger.info(f"🛑 Stopping {name}...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {name} stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning(f"⚠️ {name} didn't stop gracefully, forcing...")
                process.kill()
                process.wait()
                logger.info(f"✅ {name} force stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
            
            del self.processes[component_id]
    
    def start_all(self) -> bool:
        """Start all components"""
        logger.info("🚀 Starting Smart Second Brain System...")
        
        # Start components in order
        startup_order = ['backend', 'frontend']
        started_components = []
        
        for component_id in startup_order:
            if not self.start_component(component_id):
                logger.error(f"❌ Failed to start {component_id}, stopping all components...")
                for started_id in started_components:
                    self.stop_component(started_id)
                return False
            
            started_components.append(component_id)
            time.sleep(2)  # Brief pause between starts
        
        logger.info("🎉 All components started successfully!")
        return True
    
    def stop_all(self):
        """Stop all components"""
        logger.info("🛑 Stopping all components...")
        
        # Stop in reverse order
        for component_id in reversed(list(self.processes.keys())):
            self.stop_component(component_id)
        
        logger.info("✅ All components stopped")
    
    def monitor_components(self):
        """Monitor running components"""
        logger.info("📊 Starting component monitoring...")
        
        while self.running:
            for component_id, process in self.processes.items():
                if process.poll() is not None:
                    name = self.components[component_id]['name']
                    logger.warning(f"⚠️ {name} has stopped unexpectedly (exit code: {process.returncode})")
                    
                    # Restart the component
                    logger.info(f"🔄 Restarting {name}...")
                    self.stop_component(component_id)
                    if not self.start_component(component_id):
                        logger.error(f"❌ Failed to restart {name}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def show_status(self):
        """Show current system status"""
        logger.info("📊 System Status:")
        logger.info("=" * 50)
        
        for component_id, component in self.components.items():
            name = component['name']
            port = component['port']
            
            if component_id in self.processes:
                process = self.processes[component_id]
                if process.poll() is None:
                    status = "🟢 Running"
                    pid = process.pid
                else:
                    status = "🔴 Stopped"
                    pid = "N/A"
            else:
                status = "⚪ Not Started"
                pid = "N/A"
            
            logger.info(f"{name:<20} | {status:<12} | Port: {port:<5} | PID: {pid}")
        
        logger.info("=" * 50)
    
    def run(self):
        """Main run loop"""
        try:
            # Start all components
            if not self.start_all():
                return False
            
            # Show initial status
            self.show_status()
            
            # Start monitoring in background
            monitor_thread = threading.Thread(target=self.monitor_components, daemon=True)
            monitor_thread.start()
            
            logger.info("🎯 System is running! Press Ctrl+C to stop all components.")
            logger.info("📱 Frontend (Streamlit): http://localhost:5173")
            logger.info("🔌 Backend API: http://localhost:8000")
            logger.info("📚 API Docs: http://localhost:8000/docs")
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.stop_all()
            logger.info("👋 Goodbye!")

def main():
    """Main entry point"""
    print("🧠 Smart Second Brain - System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("api").exists() or not Path("frontend").exists():
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected: api/ and frontend/ folders")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        sys.exit(1)
    
    # Check required packages
    required_packages = ['uvicorn', 'fastapi', 'streamlit']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Error: Missing required packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    print("✅ Environment check passed")
    print("🚀 Starting system components...")
    print()
    
    # Create and run system manager
    manager = SystemManager()
    success = manager.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
