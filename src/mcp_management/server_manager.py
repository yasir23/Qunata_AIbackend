"""
MCP Server Manager for centralized orchestration of all MCP servers.

This module provides comprehensive management of MCP servers including lifecycle
management, health monitoring, load balancing, and configuration management.
"""

import os
import asyncio
import logging
import subprocess
import signal
import time
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)

class ServerStatus(Enum):
    """MCP Server status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNHEALTHY = "unhealthy"

class ServerType(Enum):
    """MCP Server type enumeration."""
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    GITHUB = "github"

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    server_type: ServerType
    port: int
    script_path: str
    health_check_path: str
    required_env_vars: List[str]
    startup_timeout: int = 30
    health_check_interval: int = 30
    max_retries: int = 3
    restart_delay: int = 5
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50

@dataclass
class ServerInstance:
    """Runtime instance of an MCP server."""
    config: MCPServerConfig
    process: Optional[subprocess.Popen] = None
    status: ServerStatus = ServerStatus.STOPPED
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    restart_count: int = 0
    error_message: Optional[str] = None
    resource_usage: Dict[str, Any] = None

class MCPServerManager:
    """Centralized MCP server orchestration and management."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize MCP Server Manager.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.servers: Dict[str, ServerInstance] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Default server configurations
        self.default_configs = {
            "reddit": MCPServerConfig(
                name="reddit",
                server_type=ServerType.REDDIT,
                port=8001,
                script_path="scripts/start_reddit_server.py",
                health_check_path="/health",
                required_env_vars=["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
                startup_timeout=30,
                health_check_interval=30,
                max_retries=3
            ),
            "youtube": MCPServerConfig(
                name="youtube",
                server_type=ServerType.YOUTUBE,
                port=8002,
                script_path="scripts/start_youtube_server.py",
                health_check_path="/health",
                required_env_vars=["YOUTUBE_API_KEY"],
                startup_timeout=30,
                health_check_interval=30,
                max_retries=3
            ),
            "github": MCPServerConfig(
                name="github",
                server_type=ServerType.GITHUB,
                port=8003,
                script_path="scripts/start_github_server.py",
                health_check_path="/health",
                required_env_vars=["GITHUB_TOKEN"],
                startup_timeout=30,
                health_check_interval=30,
                max_retries=3
            )
        }
        
        # Initialize server instances
        for name, config in self.default_configs.items():
            self.servers[name] = ServerInstance(config=config)
        
        logger.info(f"MCP Server Manager initialized with {len(self.servers)} servers")
    
    async def start_all_servers(self) -> Dict[str, bool]:
        """
        Start all configured MCP servers.
        
        Returns:
            Dictionary mapping server names to start success status
        """
        logger.info("Starting all MCP servers...")
        
        results = {}
        start_tasks = []
        
        for server_name in self.servers:
            task = asyncio.create_task(self.start_server(server_name))
            start_tasks.append((server_name, task))
        
        for server_name, task in start_tasks:
            try:
                success = await task
                results[server_name] = success
                if success:
                    logger.info(f"✅ {server_name} server started successfully")
                else:
                    logger.error(f"❌ {server_name} server failed to start")
            except Exception as e:
                logger.error(f"❌ {server_name} server start failed with exception: {e}")
                results[server_name] = False
        
        # Start monitoring task
        if not self.monitoring_task or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        successful_starts = sum(1 for success in results.values() if success)
        logger.info(f"Started {successful_starts}/{len(results)} MCP servers successfully")
        
        return results
    
    async def start_server(self, server_name: str) -> bool:
        """
        Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        config = server.config
        
        # Check if server is already running
        if server.status == ServerStatus.RUNNING:
            logger.info(f"Server {server_name} is already running")
            return True
        
        # Check if server is currently starting
        if server.status == ServerStatus.STARTING:
            logger.info(f"Server {server_name} is already starting")
            return await self._wait_for_server_ready(server_name)
        
        logger.info(f"Starting {server_name} server on port {config.port}...")
        
        try:
            # Validate environment variables
            missing_vars = self._check_required_env_vars(config.required_env_vars)
            if missing_vars:
                error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
                logger.error(f"Cannot start {server_name}: {error_msg}")
                server.status = ServerStatus.ERROR
                server.error_message = error_msg
                return False
            
            # Check if port is available
            if not await self._is_port_available(config.port):
                error_msg = f"Port {config.port} is already in use"
                logger.error(f"Cannot start {server_name}: {error_msg}")
                server.status = ServerStatus.ERROR
                server.error_message = error_msg
                return False
            
            # Update server status
            server.status = ServerStatus.STARTING
            server.error_message = None
            
            # Build command to start server
            script_path = self.project_root / config.script_path
            if not script_path.exists():
                error_msg = f"Server script not found: {script_path}"
                logger.error(f"Cannot start {server_name}: {error_msg}")
                server.status = ServerStatus.ERROR
                server.error_message = error_msg
                return False
            
            # Start the server process
            cmd = [
                "python", str(script_path),
                "--port", str(config.port)
            ]
            
            server.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_root),
                preexec_fn=os.setsid  # Create new process group
            )
            
            server.pid = server.process.pid
            server.start_time = datetime.now()
            server.restart_count += 1
            
            logger.info(f"Started {server_name} process with PID {server.pid}")
            
            # Wait for server to be ready
            success = await self._wait_for_server_ready(server_name)
            
            if success:
                server.status = ServerStatus.RUNNING
                logger.info(f"✅ {server_name} server is running and healthy")
            else:
                server.status = ServerStatus.ERROR
                await self._cleanup_server_process(server)
                logger.error(f"❌ {server_name} server failed to start properly")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting {server_name} server: {e}")
            server.status = ServerStatus.ERROR
            server.error_message = str(e)
            await self._cleanup_server_process(server)
            return False
    
    async def stop_server(self, server_name: str, force: bool = False) -> bool:
        """
        Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop
            force: Whether to force kill the server
            
        Returns:
            True if server stopped successfully, False otherwise
        """
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server.status == ServerStatus.STOPPED:
            logger.info(f"Server {server_name} is already stopped")
            return True
        
        logger.info(f"Stopping {server_name} server...")
        
        try:
            server.status = ServerStatus.STOPPING
            
            success = await self._cleanup_server_process(server, force=force)
            
            if success:
                server.status = ServerStatus.STOPPED
                server.pid = None
                server.start_time = None
                server.last_health_check = None
                server.health_check_failures = 0
                logger.info(f"✅ {server_name} server stopped successfully")
            else:
                server.status = ServerStatus.ERROR
                logger.error(f"❌ Failed to stop {server_name} server")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping {server_name} server: {e}")
            server.status = ServerStatus.ERROR
            server.error_message = str(e)
            return False
    
    async def stop_all_servers(self, force: bool = False) -> Dict[str, bool]:
        """
        Stop all MCP servers.
        
        Args:
            force: Whether to force kill all servers
            
        Returns:
            Dictionary mapping server names to stop success status
        """
        logger.info("Stopping all MCP servers...")
        
        # Stop monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.shutdown_event.set()
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.monitoring_task.cancel()
        
        results = {}
        stop_tasks = []
        
        for server_name in self.servers:
            if self.servers[server_name].status != ServerStatus.STOPPED:
                task = asyncio.create_task(self.stop_server(server_name, force=force))
                stop_tasks.append((server_name, task))
        
        for server_name, task in stop_tasks:
            try:
                success = await task
                results[server_name] = success
            except Exception as e:
                logger.error(f"Error stopping {server_name}: {e}")
                results[server_name] = False
        
        successful_stops = sum(1 for success in results.values() if success)
        logger.info(f"Stopped {successful_stops}/{len(results)} MCP servers successfully")
        
        return results
    
    async def restart_server(self, server_name: str) -> bool:
        """
        Restart a specific MCP server.
        
        Args:
            server_name: Name of the server to restart
            
        Returns:
            True if server restarted successfully, False otherwise
        """
        logger.info(f"Restarting {server_name} server...")
        
        # Stop the server first
        stop_success = await self.stop_server(server_name)
        if not stop_success:
            logger.error(f"Failed to stop {server_name} for restart")
            return False
        
        # Wait a bit before restarting
        await asyncio.sleep(self.servers[server_name].config.restart_delay)
        
        # Start the server
        start_success = await self.start_server(server_name)
        
        if start_success:
            logger.info(f"✅ {server_name} server restarted successfully")
        else:
            logger.error(f"❌ Failed to restart {server_name} server")
        
        return start_success
    
    async def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server status information or None if server not found
        """
        if server_name not in self.servers:
            return None
        
        server = self.servers[server_name]
        config = server.config
        
        # Get resource usage if process is running
        resource_usage = None
        if server.pid:
            resource_usage = await self._get_process_resource_usage(server.pid)
        
        return {
            "name": server_name,
            "type": config.server_type.value,
            "status": server.status.value,
            "port": config.port,
            "pid": server.pid,
            "start_time": server.start_time.isoformat() if server.start_time else None,
            "uptime_seconds": (datetime.now() - server.start_time).total_seconds() if server.start_time else None,
            "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
            "health_check_failures": server.health_check_failures,
            "restart_count": server.restart_count,
            "error_message": server.error_message,
            "resource_usage": resource_usage,
            "config": {
                "startup_timeout": config.startup_timeout,
                "health_check_interval": config.health_check_interval,
                "max_retries": config.max_retries,
                "memory_limit_mb": config.memory_limit_mb,
                "cpu_limit_percent": config.cpu_limit_percent
            }
        }
    
    async def get_all_servers_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all servers.
        
        Returns:
            Dictionary mapping server names to their status information
        """
        status_dict = {}
        
        for server_name in self.servers:
            status_dict[server_name] = await self.get_server_status(server_name)
        
        return status_dict
    
    async def health_check_server(self, server_name: str) -> bool:
        """
        Perform health check on a specific server.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            True if server is healthy, False otherwise
        """
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        config = server.config
        
        if server.status != ServerStatus.RUNNING:
            return False
        
        try:
            # Check if process is still running
            if not server.process or server.process.poll() is not None:
                logger.warning(f"{server_name} process is no longer running")
                server.status = ServerStatus.ERROR
                return False
            
            # Perform HTTP health check
            health_url = f"http://localhost:{config.port}{config.health_check_path}"
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            server.last_health_check = datetime.now()
                            server.health_check_failures = 0
                            return True
                        else:
                            logger.warning(f"{server_name} health check failed with status {response.status}")
                            server.health_check_failures += 1
                            return False
                except asyncio.TimeoutError:
                    logger.warning(f"{server_name} health check timed out")
                    server.health_check_failures += 1
                    return False
                except Exception as e:
                    logger.warning(f"{server_name} health check failed: {e}")
                    server.health_check_failures += 1
                    return False
        
        except Exception as e:
            logger.error(f"Error during health check for {server_name}: {e}")
            server.health_check_failures += 1
            return False
    
    async def _wait_for_server_ready(self, server_name: str) -> bool:
        """Wait for server to be ready and healthy."""
        server = self.servers[server_name]
        config = server.config
        
        start_time = time.time()
        
        while time.time() - start_time < config.startup_timeout:
            # Check if process is still running
            if server.process and server.process.poll() is not None:
                logger.error(f"{server_name} process exited during startup")
                return False
            
            # Try health check
            if await self.health_check_server(server_name):
                return True
            
            await asyncio.sleep(1)
        
        logger.error(f"{server_name} failed to become ready within {config.startup_timeout} seconds")
        return False
    
    async def _cleanup_server_process(self, server: ServerInstance, force: bool = False) -> bool:
        """Clean up server process."""
        if not server.process:
            return True
        
        try:
            if force or server.process.poll() is None:
                # Try graceful shutdown first
                if not force:
                    try:
                        os.killpg(os.getpgid(server.process.pid), signal.SIGTERM)
                        # Wait for graceful shutdown
                        try:
                            server.process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            # Force kill if graceful shutdown failed
                            os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
                            server.process.wait(timeout=5)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        # Process already dead or force kill failed
                        pass
                else:
                    # Force kill immediately
                    os.killpg(os.getpgid(server.process.pid), signal.SIGKILL)
                    server.process.wait(timeout=5)
            
            server.process = None
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up server process: {e}")
            return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop for all servers."""
        logger.info("Starting MCP server monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                for server_name, server in self.servers.items():
                    if server.status == ServerStatus.RUNNING:
                        # Perform health check
                        is_healthy = await self.health_check_server(server_name)
                        
                        if not is_healthy:
                            config = server.config
                            
                            if server.health_check_failures >= config.max_retries:
                                logger.error(f"{server_name} exceeded max health check failures, marking as unhealthy")
                                server.status = ServerStatus.UNHEALTHY
                                
                                # Attempt restart
                                logger.info(f"Attempting to restart unhealthy server: {server_name}")
                                restart_success = await self.restart_server(server_name)
                                
                                if not restart_success:
                                    logger.error(f"Failed to restart {server_name}, marking as error")
                                    server.status = ServerStatus.ERROR
                        
                        # Check resource usage
                        if server.pid:
                            resource_usage = await self._get_process_resource_usage(server.pid)
                            server.resource_usage = resource_usage
                            
                            # Check resource limits
                            if resource_usage:
                                memory_mb = resource_usage.get("memory_mb", 0)
                                cpu_percent = resource_usage.get("cpu_percent", 0)
                                
                                if memory_mb > config.memory_limit_mb:
                                    logger.warning(f"{server_name} exceeding memory limit: {memory_mb}MB > {config.memory_limit_mb}MB")
                                
                                if cpu_percent > config.cpu_limit_percent:
                                    logger.warning(f"{server_name} exceeding CPU limit: {cpu_percent}% > {config.cpu_limit_percent}%")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(min(server.config.health_check_interval for server in self.servers.values()))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        logger.info("MCP server monitoring loop stopped")
    
    def _check_required_env_vars(self, required_vars: List[str]) -> List[str]:
        """Check for missing required environment variables."""
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        return missing
    
    async def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            # Try to connect to the port
            reader, writer = await asyncio.open_connection('localhost', port)
            writer.close()
            await writer.wait_closed()
            return False  # Port is in use
        except (ConnectionRefusedError, OSError):
            return True  # Port is available
    
    async def _get_process_resource_usage(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get resource usage for a process."""
        try:
            process = psutil.Process(pid)
            
            # Get memory info
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Get CPU usage
            cpu_percent = process.cpu_percent()
            
            # Get process status
            status = process.status()
            
            # Get number of threads
            num_threads = process.num_threads()
            
            return {
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": round(cpu_percent, 2),
                "status": status,
                "num_threads": num_threads,
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get MCP server manager statistics."""
        running_servers = sum(1 for server in self.servers.values() if server.status == ServerStatus.RUNNING)
        total_servers = len(self.servers)
        
        return {
            "total_servers": total_servers,
            "running_servers": running_servers,
            "stopped_servers": sum(1 for server in self.servers.values() if server.status == ServerStatus.STOPPED),
            "error_servers": sum(1 for server in self.servers.values() if server.status == ServerStatus.ERROR),
            "unhealthy_servers": sum(1 for server in self.servers.values() if server.status == ServerStatus.UNHEALTHY),
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
            "server_types": list(set(server.config.server_type.value for server in self.servers.values())),
            "total_restarts": sum(server.restart_count for server in self.servers.values()),
            "uptime_hours": {
                name: round((datetime.now() - server.start_time).total_seconds() / 3600, 2) 
                if server.start_time else 0
                for name, server in self.servers.items()
            }
        }
