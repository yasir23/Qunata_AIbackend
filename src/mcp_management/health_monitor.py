"""
Health Monitor for MCP servers.

This module provides comprehensive health monitoring capabilities for MCP servers
including health checks, alerting, and performance monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp

from .server_manager import ServerStatus, MCPServerManager

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class HealthCheckResult:
    """Health check result data structure."""
    server_name: str
    status: HealthStatus
    response_time_ms: float
    metrics: List[HealthMetric]
    errors: List[str]
    timestamp: datetime
    details: Dict[str, Any] = None

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    server_name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MCPHealthMonitor:
    """Comprehensive health monitoring for MCP servers."""
    
    def __init__(self, server_manager: MCPServerManager):
        """
        Initialize health monitor.
        
        Args:
            server_manager: MCP server manager instance
        """
        self.server_manager = server_manager
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.history_retention_hours = 24
        self.max_history_entries = 1000
        
        # Health thresholds
        self.response_time_warning_ms = 1000
        self.response_time_critical_ms = 5000
        self.failure_rate_warning = 0.1  # 10%
        self.failure_rate_critical = 0.3  # 30%
        
        logger.info("MCP Health Monitor initialized")
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Health monitoring is already running")
            return
        
        logger.info("Starting MCP health monitoring")
        self.shutdown_event.clear()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        if not self.monitoring_task or self.monitoring_task.done():
            logger.info("Health monitoring is not running")
            return
        
        logger.info("Stopping MCP health monitoring")
        self.shutdown_event.set()
        
        try:
            await asyncio.wait_for(self.monitoring_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Health monitoring task did not stop gracefully, cancelling")
            self.monitoring_task.cancel()
    
    async def check_server_health(self, server_name: str) -> HealthCheckResult:
        """
        Perform comprehensive health check on a server.
        
        Args:
            server_name: Name of the server to check
            
        Returns:
            Health check result
        """
        start_time = time.time()
        errors = []
        metrics = []
        status = HealthStatus.UNKNOWN
        
        try:
            # Get server status from manager
            server_status = await self.server_manager.get_server_status(server_name)
            
            if not server_status:
                errors.append(f"Server {server_name} not found")
                status = HealthStatus.UNHEALTHY
            else:
                # Check if server is running
                if server_status["status"] != ServerStatus.RUNNING.value:
                    errors.append(f"Server is not running (status: {server_status['status']})")
                    status = HealthStatus.UNHEALTHY
                else:
                    # Perform HTTP health check
                    port = server_status["port"]
                    health_url = f"http://localhost:{port}/health"
                    
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                                response_time_ms = (time.time() - start_time) * 1000
                                
                                if response.status == 200:
                                    # Parse health response
                                    try:
                                        health_data = await response.json()
                                        
                                        # Extract metrics from health response
                                        if isinstance(health_data, dict):
                                            for key, value in health_data.items():
                                                if isinstance(value, (int, float)):
                                                    metrics.append(HealthMetric(
                                                        name=key,
                                                        value=float(value),
                                                        unit="count" if key.endswith("_count") else "ms" if key.endswith("_time") else "value"
                                                    ))
                                    except Exception as e:
                                        logger.debug(f"Could not parse health response JSON: {e}")
                                    
                                    # Determine status based on response time
                                    if response_time_ms > self.response_time_critical_ms:
                                        status = HealthStatus.UNHEALTHY
                                        errors.append(f"Response time too high: {response_time_ms:.1f}ms")
                                    elif response_time_ms > self.response_time_warning_ms:
                                        status = HealthStatus.DEGRADED
                                    else:
                                        status = HealthStatus.HEALTHY
                                    
                                    # Add response time metric
                                    metrics.append(HealthMetric(
                                        name="response_time",
                                        value=response_time_ms,
                                        unit="ms",
                                        threshold_warning=self.response_time_warning_ms,
                                        threshold_critical=self.response_time_critical_ms
                                    ))
                                    
                                else:
                                    errors.append(f"Health check returned status {response.status}")
                                    status = HealthStatus.UNHEALTHY
                                    
                    except asyncio.TimeoutError:
                        errors.append("Health check timed out")
                        status = HealthStatus.UNHEALTHY
                    except Exception as e:
                        errors.append(f"Health check failed: {str(e)}")
                        status = HealthStatus.UNHEALTHY
                
                # Add resource usage metrics if available
                resource_usage = server_status.get("resource_usage")
                if resource_usage:
                    metrics.append(HealthMetric(
                        name="memory_usage",
                        value=resource_usage.get("memory_mb", 0),
                        unit="MB"
                    ))
                    metrics.append(HealthMetric(
                        name="cpu_usage",
                        value=resource_usage.get("cpu_percent", 0),
                        unit="%"
                    ))
                    metrics.append(HealthMetric(
                        name="thread_count",
                        value=resource_usage.get("num_threads", 0),
                        unit="count"
                    ))
        
        except Exception as e:
            logger.error(f"Error during health check for {server_name}: {e}")
            errors.append(f"Health check error: {str(e)}")
            status = HealthStatus.UNHEALTHY
        
        # Calculate final response time
        response_time_ms = (time.time() - start_time) * 1000
        
        result = HealthCheckResult(
            server_name=server_name,
            status=status,
            response_time_ms=response_time_ms,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now(),
            details=server_status
        )
        
        # Store in history
        self._store_health_result(result)
        
        # Check for alerts
        await self._check_alerts(result)
        
        return result
    
    async def check_all_servers_health(self) -> Dict[str, HealthCheckResult]:
        """
        Check health of all servers.
        
        Returns:
            Dictionary mapping server names to health check results
        """
        results = {}
        
        # Get all server names from manager
        all_status = await self.server_manager.get_all_servers_status()
        
        # Perform health checks concurrently
        tasks = []
        for server_name in all_status.keys():
            task = asyncio.create_task(self.check_server_health(server_name))
            tasks.append((server_name, task))
        
        for server_name, task in tasks:
            try:
                result = await task
                results[server_name] = result
            except Exception as e:
                logger.error(f"Health check failed for {server_name}: {e}")
                results[server_name] = HealthCheckResult(
                    server_name=server_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    metrics=[],
                    errors=[f"Health check failed: {str(e)}"],
                    timestamp=datetime.now()
                )
        
        return results
    
    def get_server_health_history(self, server_name: str, hours: int = 1) -> List[HealthCheckResult]:
        """
        Get health history for a server.
        
        Args:
            server_name: Name of the server
            hours: Number of hours of history to retrieve
            
        Returns:
            List of health check results
        """
        if server_name not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            result for result in self.health_history[server_name]
            if result.timestamp >= cutoff_time
        ]
    
    def get_server_health_summary(self, server_name: str, hours: int = 1) -> Dict[str, Any]:
        """
        Get health summary for a server.
        
        Args:
            server_name: Name of the server
            hours: Number of hours to summarize
            
        Returns:
            Health summary statistics
        """
        history = self.get_server_health_history(server_name, hours)
        
        if not history:
            return {
                "server_name": server_name,
                "period_hours": hours,
                "total_checks": 0,
                "availability": 0.0,
                "avg_response_time_ms": 0.0,
                "error_rate": 0.0,
                "status_distribution": {},
                "recent_errors": []
            }
        
        # Calculate statistics
        total_checks = len(history)
        healthy_checks = sum(1 for result in history if result.status == HealthStatus.HEALTHY)
        availability = (healthy_checks / total_checks) * 100
        
        response_times = [result.response_time_ms for result in history if result.response_time_ms > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        error_checks = sum(1 for result in history if result.errors)
        error_rate = (error_checks / total_checks) * 100
        
        # Status distribution
        status_counts = {}
        for result in history:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Recent errors (last 10)
        recent_errors = []
        for result in reversed(history[-10:]):
            if result.errors:
                recent_errors.extend([
                    {
                        "timestamp": result.timestamp.isoformat(),
                        "error": error
                    }
                    for error in result.errors
                ])
        
        return {
            "server_name": server_name,
            "period_hours": hours,
            "total_checks": total_checks,
            "availability": round(availability, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "error_rate": round(error_rate, 2),
            "status_distribution": status_counts,
            "recent_errors": recent_errors[:10]  # Limit to 10 most recent
        }
    
    def get_all_servers_health_summary(self, hours: int = 1) -> Dict[str, Dict[str, Any]]:
        """
        Get health summary for all servers.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Dictionary mapping server names to health summaries
        """
        summaries = {}
        
        for server_name in self.health_history.keys():
            summaries[server_name] = self.get_server_health_summary(server_name, hours)
        
        return summaries
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.active_alerts.values()
            if alert.timestamp >= cutoff_time
        ]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Add callback function for alert notifications.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Remove alert callback function.
        
        Args:
            callback: Function to remove
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("Starting health monitoring loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Perform health checks on all servers
                await self.check_all_servers_health()
                
                # Clean up old history
                self._cleanup_old_history()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
        
        logger.info("Health monitoring loop stopped")
    
    def _store_health_result(self, result: HealthCheckResult):
        """Store health check result in history."""
        server_name = result.server_name
        
        if server_name not in self.health_history:
            self.health_history[server_name] = []
        
        self.health_history[server_name].append(result)
        
        # Limit history size
        if len(self.health_history[server_name]) > self.max_history_entries:
            self.health_history[server_name] = self.health_history[server_name][-self.max_history_entries:]
    
    def _cleanup_old_history(self):
        """Clean up old health history entries."""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
        
        for server_name in self.health_history:
            self.health_history[server_name] = [
                result for result in self.health_history[server_name]
                if result.timestamp >= cutoff_time
            ]
    
    async def _check_alerts(self, result: HealthCheckResult):
        """Check for alert conditions and trigger alerts."""
        server_name = result.server_name
        
        # Check for server down alert
        if result.status == HealthStatus.UNHEALTHY:
            alert_id = f"{server_name}_unhealthy"
            if alert_id not in self.active_alerts or self.active_alerts[alert_id].resolved:
                alert = Alert(
                    id=alert_id,
                    server_name=server_name,
                    level=AlertLevel.CRITICAL,
                    message=f"Server {server_name} is unhealthy: {', '.join(result.errors)}",
                    timestamp=datetime.now()
                )
                self.active_alerts[alert_id] = alert
                await self._trigger_alert(alert)
        else:
            # Resolve unhealthy alert if it exists
            alert_id = f"{server_name}_unhealthy"
            if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolved_at = datetime.now()
        
        # Check for high response time alert
        if result.response_time_ms > self.response_time_critical_ms:
            alert_id = f"{server_name}_high_response_time"
            if alert_id not in self.active_alerts or self.active_alerts[alert_id].resolved:
                alert = Alert(
                    id=alert_id,
                    server_name=server_name,
                    level=AlertLevel.ERROR,
                    message=f"Server {server_name} has high response time: {result.response_time_ms:.1f}ms",
                    timestamp=datetime.now()
                )
                self.active_alerts[alert_id] = alert
                await self._trigger_alert(alert)
        elif result.response_time_ms <= self.response_time_warning_ms:
            # Resolve high response time alert if it exists
            alert_id = f"{server_name}_high_response_time"
            if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolved_at = datetime.now()
        
        # Check failure rate over time
        history = self.get_server_health_history(server_name, hours=1)
        if len(history) >= 5:  # Need at least 5 checks
            failed_checks = sum(1 for h in history if h.status == HealthStatus.UNHEALTHY)
            failure_rate = failed_checks / len(history)
            
            if failure_rate >= self.failure_rate_critical:
                alert_id = f"{server_name}_high_failure_rate"
                if alert_id not in self.active_alerts or self.active_alerts[alert_id].resolved:
                    alert = Alert(
                        id=alert_id,
                        server_name=server_name,
                        level=AlertLevel.ERROR,
                        message=f"Server {server_name} has high failure rate: {failure_rate*100:.1f}%",
                        timestamp=datetime.now()
                    )
                    self.active_alerts[alert_id] = alert
                    await self._trigger_alert(alert)
            elif failure_rate <= self.failure_rate_warning:
                # Resolve high failure rate alert if it exists
                alert_id = f"{server_name}_high_failure_rate"
                if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                    self.active_alerts[alert_id].resolved = True
                    self.active_alerts[alert_id].resolved_at = datetime.now()
    
    async def _trigger_alert(self, alert: Alert):
        """Trigger alert notifications."""
        logger.warning(f"ALERT [{alert.level.value.upper()}] {alert.server_name}: {alert.message}")
        
        # Call all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get health monitor statistics."""
        total_servers = len(self.health_history)
        active_alerts = len(self.get_active_alerts())
        
        # Calculate overall health
        recent_results = []
        for server_name in self.health_history:
            history = self.get_server_health_history(server_name, hours=1)
            if history:
                recent_results.append(history[-1])
        
        healthy_servers = sum(1 for result in recent_results if result.status == HealthStatus.HEALTHY)
        overall_health = (healthy_servers / len(recent_results) * 100) if recent_results else 0
        
        return {
            "total_servers_monitored": total_servers,
            "active_alerts": active_alerts,
            "overall_health_percentage": round(overall_health, 2),
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done(),
            "health_check_interval_seconds": self.health_check_interval,
            "history_retention_hours": self.history_retention_hours,
            "alert_callbacks_registered": len(self.alert_callbacks),
            "total_history_entries": sum(len(history) for history in self.health_history.values())
        }
