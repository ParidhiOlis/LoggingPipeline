"""
Server-Side Metrics Logging System
Logs operational metrics without exposing sensitive query/response data
Focuses on: API metrics, latency, resource usage, errors
"""

import structlog
import json
import uuid
import time
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sys


class RequestStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"


class ErrorType(Enum):
    TIMEOUT = "TIMEOUT"
    OOM = "OUT_OF_MEMORY"
    RATE_LIMIT = "RATE_LIMIT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class APIRequestMetrics:
    """API request metrics (no sensitive data)"""
    timestamp: str
    request_id: str
    endpoint: str
    platform: str
    method: str
    status_code: int
    status: str
    latency_ms: float
    input_token_count: int
    output_token_count: int
    error_type: str = ""
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceMetrics:
    """System resource usage metrics"""
    timestamp: str
    metric_id: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Aggregated latency metrics over time window"""
    timestamp: str
    metric_id: str
    window_duration_seconds: int
    request_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ServerMetricsLogger:
    """
    Server-side metrics logging
    NO sensitive data (queries/responses) - only operational metrics
    """
    
    def __init__(self, base_log_dir: str = "server_metrics", log_level: str = "INFO"):
        self.base_log_dir = Path(base_log_dir)
        self.log_level = log_level
        
        # Latency tracking for percentile calculations
        self.latency_buffer = []
        self.buffer_max_size = 1000
        
        # Setup directories
        self._setup_directories()
        
        # Configure structlog
        self._configure_structlog()
        
        # Create metric-specific loggers
        self.api_logger = structlog.get_logger("api_metrics")
        self.resource_logger = structlog.get_logger("resource_metrics")
        self.latency_logger = structlog.get_logger("latency_metrics")
        self.error_logger = structlog.get_logger("error_metrics")
    
    def _setup_directories(self):
        """Create necessary log directories"""
        for directory in ["api_requests", "resources", "latency", "errors"]:
            (self.base_log_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _configure_structlog(self):
        """Configure structlog for JSON output"""
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.log_level.upper())
        )
        
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_logger_name,
                structlog.processors.JSONRenderer(sort_keys=True)
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now(timezone.utc).isoformat()
    
    def _generate_unique_id(self) -> str:
        """Generate unique ID for metrics"""
        return str(uuid.uuid4())
    
    def log_api_request(
        self,
        endpoint: str,
        platform: str,
        method: str,
        status_code: int,
        status: RequestStatus,
        latency_ms: float,
        input_token_count: int,
        output_token_count: int,
        error_type: Optional[ErrorType] = None
    ) -> str:
        """
        Log API request metrics (NO sensitive data)
        
        Args:
            endpoint: API endpoint path (e.g., /api/query, /api/extract)
            platform: Platform name (slack, outlook, etc.)
            method: HTTP method (GET, POST, etc.)
            status_code: HTTP status code
            status: Request status
            latency_ms: Request latency in milliseconds
            input_token_count: Number of input tokens (count only, not content)
            output_token_count: Number of output tokens (count only, not content)
            error_type: Type of error if request failed
            
        Returns:
            str: Request ID for correlation
        """
        request_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        # Add to latency buffer for percentile calculation
        self.latency_buffer.append(latency_ms)
        if len(self.latency_buffer) > self.buffer_max_size:
            self.latency_buffer.pop(0)
        
        api_metrics = APIRequestMetrics(
            timestamp=current_time,
            request_id=request_id,
            endpoint=endpoint,
            platform=platform,
            method=method,
            status_code=status_code,
            status=status.value,
            latency_ms=latency_ms,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            error_type=error_type.value if error_type else "",
            error_count=1 if error_type else 0
        )
        
        self.api_logger.info(
            "api_request_metric",
            **api_metrics.to_dict()
        )
        
        return request_id
    
    def log_resource_usage(
        self,
        include_gpu: bool = False
    ) -> str:
        """
        Log current system resource usage
        
        Args:
            include_gpu: Whether to include GPU metrics (requires pynvml)
            
        Returns:
            str: Metric ID
        """
        metric_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        # Get CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        gpu_memory_mb = None
        gpu_utilization = None
        
        if include_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_utilization_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_memory_mb = gpu_mem_info.used / (1024 ** 2)
                gpu_utilization = gpu_utilization_info.gpu
                
                pynvml.nvmlShutdown()
            except Exception:
                pass  # GPU metrics unavailable
        
        resource_metrics = ResourceMetrics(
            timestamp=current_time,
            metric_id=metric_id,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 ** 2),
            memory_available_mb=memory.available / (1024 ** 2),
            gpu_memory_used_mb=gpu_memory_mb,
            gpu_utilization_percent=gpu_utilization
        )
        
        self.resource_logger.info(
            "resource_usage_metric",
            **resource_metrics.to_dict()
        )
        
        return metric_id
    
    def log_latency_percentiles(
        self,
        window_duration_seconds: int = 60
    ) -> Optional[str]:
        """
        Calculate and log latency percentiles from buffer
        
        Args:
            window_duration_seconds: Time window for metrics
            
        Returns:
            str: Metric ID or None if insufficient data
        """
        if len(self.latency_buffer) < 10:
            return None  # Not enough data for meaningful percentiles
        
        metric_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        # Calculate percentiles
        sorted_latencies = sorted(self.latency_buffer)
        n = len(sorted_latencies)
        
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]
        avg = sum(sorted_latencies) / n
        
        latency_metrics = LatencyMetrics(
            timestamp=current_time,
            metric_id=metric_id,
            window_duration_seconds=window_duration_seconds,
            request_count=n,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_latency_ms=avg,
            min_latency_ms=sorted_latencies[0],
            max_latency_ms=sorted_latencies[-1]
        )
        
        self.latency_logger.info(
            "latency_percentile_metric",
            **latency_metrics.to_dict()
        )
        
        return metric_id
    
    def log_error(
        self,
        error_type: ErrorType,
        endpoint: str,
        platform: str,
        error_message: str = "",
        stack_trace: str = ""
    ) -> str:
        """
        Log error occurrence (sanitized error info only)
        
        Args:
            error_type: Type of error
            endpoint: Endpoint where error occurred
            platform: Platform where error occurred
            error_message: Sanitized error message (no sensitive data)
            stack_trace: Stack trace (if safe to log)
            
        Returns:
            str: Error ID
        """
        error_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        error_data = {
            "error_id": error_id,
            "timestamp": current_time,
            "error_type": error_type.value,
            "endpoint": endpoint,
            "platform": platform,
            "error_message": error_message,
            "has_stack_trace": bool(stack_trace)
        }
        
        # Only log stack trace in development/debug mode
        if self.log_level == "DEBUG":
            error_data["stack_trace"] = stack_trace
        
        self.error_logger.error(
            "error_occurred",
            **error_data
        )
        
        return error_id


class ServerMetricsMiddleware:
    """
    Middleware for automatic request metrics logging
    Use this in your API framework (Flask, FastAPI, etc.)
    """
    
    def __init__(self, metrics_logger: ServerMetricsLogger, platform: str):
        self.metrics_logger = metrics_logger
        self.platform = platform
    
    def log_request(self, endpoint: str, method: str, token_counts: Dict[str, int]):
        """
        Context manager for logging request metrics
        
        Usage:
            with middleware.log_request("/api/query", "POST", {"input": 10, "output": 50}):
                # Your request handling code
                result = process_request()
        """
        return RequestMetricsContext(self.metrics_logger, endpoint, method, self.platform, token_counts)


class RequestMetricsContext:
    """Context manager for automatic request metrics logging"""
    
    def __init__(self, logger: ServerMetricsLogger, endpoint: str, method: str, platform: str, token_counts: Dict[str, int]):
        self.logger = logger
        self.endpoint = endpoint
        self.method = method
        self.platform = platform
        self.token_counts = token_counts
        self.start_time = None
        self.status_code = 500
        self.status = RequestStatus.FAILURE
        self.error_type = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000
        
        # Determine status based on exception
        if exc_type is None:
            self.status_code = 200
            self.status = RequestStatus.SUCCESS
        elif exc_type == TimeoutError:
            self.status_code = 504
            self.status = RequestStatus.TIMEOUT
            self.error_type = ErrorType.TIMEOUT
        else:
            self.status_code = 500
            self.status = RequestStatus.FAILURE
            self.error_type = ErrorType.INTERNAL_ERROR
        
        # Log the request
        self.logger.log_api_request(
            endpoint=self.endpoint,
            platform=self.platform,
            method=self.method,
            status_code=self.status_code,
            status=self.status,
            latency_ms=latency_ms,
            input_token_count=self.token_counts.get("input", 0),
            output_token_count=self.token_counts.get("output", 0),
            error_type=self.error_type
        )
        
        # Don't suppress the exception
        return False


# # Example usage
# def example_server_logging():
#     """Example of server-side metrics logging"""
    
#     print("SERVER-SIDE METRICS LOGGING DEMO")
#     print("=" * 50)
    
#     # Initialize logger
#     metrics_logger = ServerMetricsLogger(base_log_dir="server_metrics")
    
#     # Simulate API requests
#     print("\n1. API Request Metrics (NO sensitive data)")
#     print("-" * 40)
    
#     # Successful request
#     request_id = metrics_logger.log_api_request(
#         endpoint="/api/query",
#         platform="slack",
#         method="POST",
#         status_code=200,
#         status=RequestStatus.SUCCESS,
#         latency_ms=125.5,
#         input_token_count=50,  # Only count, not actual content
#         output_token_count=150
#     )
#     print(f"Logged successful request: {request_id[:8]}...")
    
#     # Failed request
#     error_id = metrics_logger.log_api_request(
#         endpoint="/api/extract",
#         platform="sharepoint",
#         method="POST",
#         status_code=500,
#         status=RequestStatus.FAILURE,
#         latency_ms=5000.0,
#         input_token_count=0,
#         output_token_count=0,
#         error_type=ErrorType.TIMEOUT
#     )
#     print(f"Logged failed request: {error_id[:8]}...")
    
#     # Resource usage
#     print("\n2. Resource Usage Metrics")
#     print("-" * 40)
    
#     resource_id = metrics_logger.log_resource_usage(include_gpu=False)
#     print(f"Logged resource usage: {resource_id[:8]}...")
    
#     # Latency percentiles
#     print("\n3. Latency Percentiles")
#     print("-" * 40)
    
#     # Add some sample latencies
#     import random
#     for _ in range(100):
#         metrics_logger.latency_buffer.append(random.uniform(50, 500))
    
#     latency_id = metrics_logger.log_latency_percentiles(window_duration_seconds=60)
#     print(f"Logged latency percentiles: {latency_id[:8]}...")
    
#     # Error logging
#     print("\n4. Error Logging")
#     print("-" * 40)
    
#     error_log_id = metrics_logger.log_error(
#         error_type=ErrorType.OOM,
#         endpoint="/api/index",
#         platform="notion",
#         error_message="Memory limit exceeded during embedding generation"
#     )
#     print(f"Logged error: {error_log_id[:8]}...")
    
#     print("\n")
#     print("All metrics logged to: server_metrics/")
#     print("  - api_requests/")
#     print("  - resources/")
#     print("  - latency/")
#     print("  - errors/")
#     print("\nNO sensitive data (queries/responses) logged!")


# if __name__ == "__main__":
#     example_server_logging()