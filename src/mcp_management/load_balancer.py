"""
Load Balancer for MCP servers.

This module provides load balancing capabilities for MCP servers including
request distribution, failover, and performance optimization.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp

from .server_manager import ServerStatus, MCPServerManager
from .health_monitor import HealthStatus, MCPHealthMonitor

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"

class RequestType(Enum):
    """Request type enumeration for routing."""
    REDDIT_SEARCH = "reddit_search"
    REDDIT_POSTS = "reddit_posts"
    REDDIT_COMMENTS = "reddit_comments"
    YOUTUBE_SEARCH = "youtube_search"
    YOUTUBE_COMMENTS = "youtube_comments"
    YOUTUBE_INFO = "youtube_info"
    GITHUB_ISSUES = "github_issues"
    GITHUB_SEARCH = "github_search"
    GITHUB_REPOS = "github_repos"
    GITHUB_INSIGHTS = "github_insights"

@dataclass
class ServerMetrics:
    """Server performance metrics for load balancing."""
    server_name: str
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    weight: float = 1.0
    health_score: float = 1.0

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: int = 30
    failover_enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 10.0
    request_timeout: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

class MCPLoadBalancer:
    """Load balancer for MCP servers with failover and performance optimization."""
    
    def __init__(
        self, 
        server_manager: MCPServerManager,
        health_monitor: Optional[MCPHealthMonitor] = None,
        config: Optional[LoadBalancerConfig] = None
    ):
        """
        Initialize load balancer.
        
        Args:
            server_manager: MCP server manager instance
            health_monitor: Health monitor instance (optional)
            config: Load balancer configuration
        """
        self.server_manager = server_manager
        self.health_monitor = health_monitor
        self.config = config or LoadBalancerConfig()
        
        # Server metrics and state
        self.server_metrics: Dict[str, ServerMetrics] = {}
        self.round_robin_index: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Request routing configuration
        self.server_type_mapping = {
            RequestType.REDDIT_SEARCH: "reddit",
            RequestType.REDDIT_POSTS: "reddit", 
            RequestType.REDDIT_COMMENTS: "reddit",
            RequestType.YOUTUBE_SEARCH: "youtube",
            RequestType.YOUTUBE_COMMENTS: "youtube",
            RequestType.YOUTUBE_INFO: "youtube",
            RequestType.GITHUB_ISSUES: "github",
            RequestType.GITHUB_SEARCH: "github",
            RequestType.GITHUB_REPOS: "github",
            RequestType.GITHUB_INSIGHTS: "github"
        }
        
        # Initialize server metrics
        self._initialize_server_metrics()
        
        logger.info(f"MCP Load Balancer initialized with strategy: {self.config.strategy.value}")
    
    def _initialize_server_metrics(self):
        """Initialize metrics for all servers."""
        for server_name in self.server_manager.servers:
            self.server_metrics[server_name] = ServerMetrics(server_name=server_name)
            self.round_robin_index[server_name] = 0
            self.circuit_breakers[server_name] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half_open
            }
    
    async def route_request(
        self, 
        request_type: RequestType, 
        request_data: Dict[str, Any],
        preferred_server: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Route a request to the best available server.
        
        Args:
            request_type: Type of request to route
            request_data: Request payload
            preferred_server: Preferred server name (optional)
            
        Returns:
            Tuple of (server_name, response) or (None, None) if all servers failed
        """
        # Determine target server type
        target_server_type = self.server_type_mapping.get(request_type)
        if not target_server_type:
            logger.error(f"Unknown request type: {request_type}")
            return None, None
        
        # Get available servers of the target type
        available_servers = await self._get_available_servers(target_server_type)
        
        if not available_servers:
            logger.error(f"No available servers for type: {target_server_type}")
            return None, None
        
        # Use preferred server if specified and available
        if preferred_server and preferred_server in available_servers:
            available_servers = [preferred_server]
        
        # Try servers with retries
        for attempt in range(self.config.max_retries):
            # Select server based on load balancing strategy
            selected_server = self._select_server(available_servers, target_server_type)
            
            if not selected_server:
                logger.error(f"No server selected for request type: {request_type}")
                break
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(selected_server):
                logger.warning(f"Circuit breaker open for server: {selected_server}")
                available_servers.remove(selected_server)
                if not available_servers:
                    break
                continue
            
            # Execute request
            start_time = time.time()
            try:
                response = await self._execute_request(selected_server, request_type, request_data)
                
                # Update metrics on success
                response_time = (time.time() - start_time) * 1000
                await self._update_server_metrics(selected_server, success=True, response_time=response_time)
                self._reset_circuit_breaker(selected_server)
                
                return selected_server, response
                
            except Exception as e:
                # Update metrics on failure
                response_time = (time.time() - start_time) * 1000
                await self._update_server_metrics(selected_server, success=False, response_time=response_time)
                self._update_circuit_breaker(selected_server)
                
                logger.warning(f"Request failed on {selected_server} (attempt {attempt + 1}): {e}")
                
                # Remove failed server from available list for next attempt
                if selected_server in available_servers:
                    available_servers.remove(selected_server)
                
                # If no more servers available, break
                if not available_servers:
                    break
                
                # Wait before retry
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
        
        logger.error(f"All servers failed for request type: {request_type}")
        return None, None
    
    async def _get_available_servers(self, server_type: str) -> List[str]:
        """Get list of available servers for a given type."""
        available_servers = []
        
        # Get all servers of the specified type
        all_servers_status = await self.server_manager.get_all_servers_status()
        
        for server_name, status in all_servers_status.items():
            if status and status.get("type") == server_type:
                # Check if server is running
                if status.get("status") == ServerStatus.RUNNING.value:
                    # Check health if health monitor is available
                    if self.health_monitor:
                        try:
                            health_result = await self.health_monitor.check_server_health(server_name)
                            if health_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                                available_servers.append(server_name)
                        except Exception as e:
                            logger.warning(f"Health check failed for {server_name}: {e}")
                    else:
                        # No health monitor, assume server is available if running
                        available_servers.append(server_name)
        
        return available_servers
    
    def _select_server(self, available_servers: List[str], server_type: str) -> Optional[str]:
        """Select server based on load balancing strategy."""
        if not available_servers:
            return None
        
        if len(available_servers) == 1:
            return available_servers[0]
        
        strategy = self.config.strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_servers, server_type)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available_servers)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(available_servers, server_type)
        elif strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._select_by_response_time(available_servers)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(available_servers)
        else:
            # Default to round robin
            return self._select_round_robin(available_servers, server_type)
    
    def _select_round_robin(self, available_servers: List[str], server_type: str) -> str:
        """Select server using round robin strategy."""
        if server_type not in self.round_robin_index:
            self.round_robin_index[server_type] = 0
        
        index = self.round_robin_index[server_type] % len(available_servers)
        selected_server = available_servers[index]
        
        self.round_robin_index[server_type] = (index + 1) % len(available_servers)
        
        return selected_server
    
    def _select_least_connections(self, available_servers: List[str]) -> str:
        """Select server with least active connections."""
        min_connections = float('inf')
        selected_server = available_servers[0]
        
        for server_name in available_servers:
            metrics = self.server_metrics.get(server_name)
            if metrics and metrics.active_connections < min_connections:
                min_connections = metrics.active_connections
                selected_server = server_name
        
        return selected_server
    
    def _select_weighted_round_robin(self, available_servers: List[str], server_type: str) -> str:
        """Select server using weighted round robin strategy."""
        # Calculate total weight
        total_weight = sum(
            self.server_metrics.get(server, ServerMetrics(server)).weight 
            for server in available_servers
        )
        
        if total_weight == 0:
            return self._select_round_robin(available_servers, server_type)
        
        # Generate random number and select based on weight
        random_weight = random.uniform(0, total_weight)
        current_weight = 0
        
        for server_name in available_servers:
            metrics = self.server_metrics.get(server_name, ServerMetrics(server_name))
            current_weight += metrics.weight
            if random_weight <= current_weight:
                return server_name
        
        # Fallback to last server
        return available_servers[-1]
    
    def _select_by_response_time(self, available_servers: List[str]) -> str:
        """Select server with best average response time."""
        best_response_time = float('inf')
        selected_server = available_servers[0]
        
        for server_name in available_servers:
            metrics = self.server_metrics.get(server_name)
            if metrics and metrics.avg_response_time_ms < best_response_time:
                best_response_time = metrics.avg_response_time_ms
                selected_server = server_name
        
        return selected_server
    
    async def _execute_request(
        self, 
        server_name: str, 
        request_type: RequestType, 
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute request on selected server."""
        # Get server status to get port
        server_status = await self.server_manager.get_server_status(server_name)
        if not server_status:
            raise Exception(f"Server {server_name} not found")
        
        port = server_status["port"]
        
        # Increment active connections
        metrics = self.server_metrics.get(server_name)
        if metrics:
            metrics.active_connections += 1
        
        try:
            # Build request URL based on request type
            endpoint = self._get_endpoint_for_request_type(request_type)
            url = f"http://localhost:{port}{endpoint}"
            
            # Execute HTTP request
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(
                    total=self.config.request_timeout,
                    connect=self.config.connection_timeout
                )
                
                async with session.post(url, json=request_data, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
        
        finally:
            # Decrement active connections
            if metrics:
                metrics.active_connections = max(0, metrics.active_connections - 1)
    
    def _get_endpoint_for_request_type(self, request_type: RequestType) -> str:
        """Get API endpoint for request type."""
        endpoint_mapping = {
            RequestType.REDDIT_SEARCH: "/search",
            RequestType.REDDIT_POSTS: "/posts",
            RequestType.REDDIT_COMMENTS: "/comments",
            RequestType.YOUTUBE_SEARCH: "/search",
            RequestType.YOUTUBE_COMMENTS: "/comments",
            RequestType.YOUTUBE_INFO: "/info",
            RequestType.GITHUB_ISSUES: "/issues",
            RequestType.GITHUB_SEARCH: "/search",
            RequestType.GITHUB_REPOS: "/repos",
            RequestType.GITHUB_INSIGHTS: "/insights"
        }
        
        return endpoint_mapping.get(request_type, "/")
    
    async def _update_server_metrics(
        self, 
        server_name: str, 
        success: bool, 
        response_time: float
    ):
        """Update server performance metrics."""
        metrics = self.server_metrics.get(server_name)
        if not metrics:
            metrics = ServerMetrics(server_name=server_name)
            self.server_metrics[server_name] = metrics
        
        # Update request counts
        metrics.total_requests += 1
        if not success:
            metrics.failed_requests += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if metrics.avg_response_time_ms == 0:
            metrics.avg_response_time_ms = response_time
        else:
            metrics.avg_response_time_ms = (
                alpha * response_time + (1 - alpha) * metrics.avg_response_time_ms
            )
        
        metrics.last_request_time = datetime.now()
        
        # Update health score based on success rate
        if metrics.total_requests > 0:
            success_rate = (metrics.total_requests - metrics.failed_requests) / metrics.total_requests
            metrics.health_score = success_rate
        
        # Update weight based on performance (inverse of response time and failure rate)
        if metrics.avg_response_time_ms > 0 and metrics.health_score > 0:
            metrics.weight = metrics.health_score / (metrics.avg_response_time_ms / 1000)
        else:
            metrics.weight = 1.0
    
    def _is_circuit_breaker_open(self, server_name: str) -> bool:
        """Check if circuit breaker is open for server."""
        breaker = self.circuit_breakers.get(server_name)
        if not breaker:
            return False
        
        if breaker["state"] == "open":
            # Check if timeout has passed
            if breaker["last_failure"]:
                time_since_failure = (datetime.now() - breaker["last_failure"]).total_seconds()
                if time_since_failure >= self.config.circuit_breaker_timeout:
                    breaker["state"] = "half_open"
                    return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, server_name: str):
        """Update circuit breaker state on failure."""
        breaker = self.circuit_breakers.get(server_name)
        if not breaker:
            return
        
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.now()
        
        if breaker["failures"] >= self.config.circuit_breaker_threshold:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for server: {server_name}")
    
    def _reset_circuit_breaker(self, server_name: str):
        """Reset circuit breaker on successful request."""
        breaker = self.circuit_breakers.get(server_name)
        if not breaker:
            return
        
        breaker["failures"] = 0
        breaker["state"] = "closed"
    
    def get_server_metrics(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific server."""
        metrics = self.server_metrics.get(server_name)
        if not metrics:
            return None
        
        return {
            "server_name": metrics.server_name,
            "active_connections": metrics.active_connections,
            "total_requests": metrics.total_requests,
            "failed_requests": metrics.failed_requests,
            "success_rate": (
                (metrics.total_requests - metrics.failed_requests) / metrics.total_requests * 100
                if metrics.total_requests > 0 else 0
            ),
            "avg_response_time_ms": round(metrics.avg_response_time_ms, 2),
            "last_request_time": metrics.last_request_time.isoformat() if metrics.last_request_time else None,
            "weight": round(metrics.weight, 3),
            "health_score": round(metrics.health_score, 3),
            "circuit_breaker": self.circuit_breakers.get(server_name, {})
        }
    
    def get_all_server_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all servers."""
        return {
            server_name: self.get_server_metrics(server_name)
            for server_name in self.server_metrics
        }
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(metrics.total_requests for metrics in self.server_metrics.values())
        total_failures = sum(metrics.failed_requests for metrics in self.server_metrics.values())
        
        active_servers = sum(
            1 for metrics in self.server_metrics.values() 
            if metrics.active_connections > 0 or (
                metrics.last_request_time and 
                (datetime.now() - metrics.last_request_time).total_seconds() < 300
            )
        )
        
        circuit_breakers_open = sum(
            1 for breaker in self.circuit_breakers.values()
            if breaker["state"] == "open"
        )
        
        return {
            "strategy": self.config.strategy.value,
            "total_servers": len(self.server_metrics),
            "active_servers": active_servers,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_success_rate": (
                (total_requests - total_failures) / total_requests * 100
                if total_requests > 0 else 0
            ),
            "circuit_breakers_open": circuit_breakers_open,
            "failover_enabled": self.config.failover_enabled,
            "max_retries": self.config.max_retries,
            "avg_response_time_ms": (
                sum(metrics.avg_response_time_ms for metrics in self.server_metrics.values()) / 
                len(self.server_metrics) if self.server_metrics else 0
            )
        }
    
    def update_config(self, config: LoadBalancerConfig):
        """Update load balancer configuration."""
        self.config = config
        logger.info(f"Load balancer configuration updated: strategy={config.strategy.value}")
    
    def reset_metrics(self):
        """Reset all server metrics."""
        for metrics in self.server_metrics.values():
            metrics.active_connections = 0
            metrics.total_requests = 0
            metrics.failed_requests = 0
            metrics.avg_response_time_ms = 0.0
            metrics.last_request_time = None
            metrics.weight = 1.0
            metrics.health_score = 1.0
        
        # Reset circuit breakers
        for breaker in self.circuit_breakers.values():
            breaker["failures"] = 0
            breaker["last_failure"] = None
            breaker["state"] = "closed"
        
        logger.info("Load balancer metrics reset")
