#!/usr/bin/env python3
"""
Smart Second Brain - System Stop Script
Gracefully stops all running components
"""

import os
import signal
import subprocess
import psutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_processes_by_name(name_patterns):
    """Find processes by name patterns"""
    processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            for pattern in name_patterns:
                if pattern in cmdline:
                    processes.append(proc)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return processes

def stop_processes_on_ports(ports):
    """Stop processes running on specific ports"""
    for port in ports:
        try:
            # Find process using the port
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    try:
                        proc = psutil.Process(conn.pid)
                        logger.info(f"🛑 Stopping process on port {port} (PID: {proc.pid})")
                        
                        # Try graceful shutdown first
                        proc.terminate()
                        proc.wait(timeout=10)
                        logger.info(f"✅ Process on port {port} stopped gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(f"⚠️ Process on port {port} didn't stop gracefully, forcing...")
                        proc.kill()
                        proc.wait()
                        logger.info(f"✅ Process on port {port} force stopped")
                    except psutil.NoSuchProcess:
                        logger.info(f"ℹ️ Process on port {port} already stopped")
                    break
            else:
                logger.info(f"ℹ️ No process found on port {port}")
        except Exception as e:
            logger.error(f"❌ Error stopping process on port {port}: {e}")

def main():
    """Main stop function"""
    print("🛑 Smart Second Brain - System Stop")
    print("=" * 40)
    
    # Ports used by our system
    system_ports = [8000, 5173]
    
    # Process name patterns to look for
    process_patterns = [
        'uvicorn',
        'python app.py',
        'start_system.py'
    ]
    
    logger.info("🔍 Looking for Smart Second Brain processes...")
    
    # Find and stop processes by name
    found_processes = find_processes_by_name(process_patterns)
    
    if found_processes:
        logger.info(f"Found {len(found_processes)} Smart Second Brain processes:")
        for proc in found_processes:
            logger.info(f"  - PID {proc.pid}: {' '.join(proc.cmdline())}")
        
        # Stop processes gracefully
        for proc in found_processes:
            try:
                logger.info(f"🛑 Stopping process {proc.pid}...")
                proc.terminate()
                proc.wait(timeout=10)
                logger.info(f"✅ Process {proc.pid} stopped gracefully")
            except psutil.TimeoutExpired:
                logger.warning(f"⚠️ Process {proc.pid} didn't stop gracefully, forcing...")
                proc.kill()
                proc.wait()
                logger.info(f"✅ Process {proc.pid} force stopped")
            except psutil.NoSuchProcess:
                logger.info(f"ℹ️ Process {proc.pid} already stopped")
    else:
        logger.info("ℹ️ No Smart Second Brain processes found by name")
    
    # Also check and stop processes on our ports
    logger.info("🔍 Checking for processes on system ports...")
    stop_processes_on_ports(system_ports)
    
    logger.info("✅ System stop completed!")
    logger.info("💡 If you still see processes, you may need to stop them manually")

if __name__ == "__main__":
    main()
