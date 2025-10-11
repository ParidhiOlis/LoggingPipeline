import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from fastapi import FastAPI, Request, Response, status
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

# Import your existing logging components
from LoggingPipeline.Logging.server_metrics_logger import ServerMetricsLogger, RequestStatus, ErrorType
from LoggingPipeline.Logging.logSystem import AuditLogger, LogStatus, InputType
from LoggingPipeline.Logging.Clickhouse_integration import ClickHouseAuditIngester, ClickHouseConfig

# Import service factory
from app.services.service_factory import ServiceFactory

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

# Global instances
_audit_logger = None
_metrics_logger = None
_clickhouse_ingester = None

class LoggingConfig:
    """Centralized logging configuration manager."""
    
    def __init__(self):
        self._initialized = False
        self._log_dir = "logs"
        self._log_level = "INFO"
        self._clickhouse_config = None
        self._audit_logger = None
        self._metrics_logger = None
        self._clickhouse_ingester = None
    
    async def initialize(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        clickhouse_config: Optional[Dict[str, Any]] = None
    ) -> tuple[ServerMetricsLogger, AuditLogger]:
        if self._initialized:
            return self._metrics_logger, self._audit_logger
            
        self._log_dir = log_dir
        self._log_level = log_level
        self._clickhouse_config = clickhouse_config or {}
        
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path / 'application.log')
            ],
            force=True  # Override any existing handlers
        )
        
        # Initialize the audit logger
        self._audit_logger = AuditLogger()
        
        # Initialize the metrics logger
        self._metrics_logger = ServerMetricsLogger()
        
        # Set up ClickHouse if enabled
        if self._clickhouse_config.get("enabled", False):
            await self._setup_clickhouse()
        
        # Initialize the service factory with the audit logger
        ServiceFactory.initialize(logger=self._audit_logger)
        
        self._initialized = True
        
        # Set global instances
        global _audit_logger, _metrics_logger, _clickhouse_ingester
        _audit_logger = self._audit_logger
        _metrics_logger = self._metrics_logger
        _clickhouse_ingester = self._clickhouse_ingester
        
        return self._metrics_logger, self._audit_logger
    
    async def _setup_clickhouse(self):
        """Set up ClickHouse integration for logging."""
        try:
            config = ClickHouseConfig(
                host=self._clickhouse_config.get("host", "localhost"),
                port=self._clickhouse_config.get("port", 9000),
                user=self._clickhouse_config.get("user", "default"),
                password=self._clickhouse_config.get("password", ""),
                database=self._clickhouse_config.get("database", "default"),
                table_name=self._clickhouse_config.get("table_name", "audit_logs"),
                batch_size=self._clickhouse_config.get("batch_size", 100),
                flush_interval=self._clickhouse_config.get("flush_interval", 10)
            )
            
            self._clickhouse_ingester = ClickHouseAuditIngester(config)
            
            # Start the ingester in the background
            asyncio.create_task(self._clickhouse_ingester.start_ingestion())
            
            # Register the ingester with the audit logger
            self._audit_logger.add_ingestion_handler(self._clickhouse_ingester.ingest_log)
            
            logging.info("ClickHouse logging integration initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize ClickHouse logging: {str(e)}")
            if self._clickhouse_config.get("required", False):
                raise
    
    async def shutdown(self):
        """Shut down the logging system and clean up resources."""
        if not self._initialized:
            return
            
        logging.info("Shutting down logging system...")
        
        # Shut down ClickHouse ingester if it exists
        if self._clickhouse_ingester:
            await self._clickhouse_ingester.stop_ingestion()
        
        # Shut down the audit logger
        if self._audit_logger:
            await self._audit_logger.shutdown()
        
        # Clear the service factory
        ServiceFactory._logger = None
        ServiceFactory._services = {}
        
        self._initialized = False
        logging.info("Logging system shut down successfully")

# Global instance
logging_config = LoggingConfig()

# Backward compatibility
def setup_logging(
    log_dir: str = "logs", 
    log_level: str = "INFO",
    clickhouse_config: Optional[Dict[str, Any]] = None
) -> tuple[ServerMetricsLogger, AuditLogger]:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're in an async context, we need to run this in a new task
        future = asyncio.ensure_future(
            logging_config.initialize(log_dir, log_level, clickhouse_config)
        )
        # This will block until the future is complete
        return loop.run_until_complete(future)
    else:
        # If we're not in an async context, we can just run it directly
        return loop.run_until_complete(
            logging_config.initialize(log_dir, log_level, clickhouse_config)
        )

def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        raise RuntimeError("Logging system has not been initialized. Call setup_logging() first.")
    return _audit_logger

def get_metrics_logger() -> ServerMetricsLogger:
    """Get the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        raise RuntimeError("Logging system has not been initialized. Call setup_logging() first.")
    return _metrics_logger

async def shutdown_logging():
    """Shut down the logging system."""
    await logging_config.shutdown()

class RequestContext:
    """Context manager for request logging."""
    
    def __init__(self, request: Request):
        self.request = request
        self.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        self.start_time = time.time()
        self.audit_logger = get_audit_logger()
        self.metrics_logger = get_metrics_logger()
    
    async def __aenter__(self):
        """Log the start of a request."""
        # Log the request
        await self.audit_logger.log_request(
            request_id=self.request_id,
            method=self.request.method,
            url=str(self.request.url),
            headers=dict(self.request.headers),
            metadata={
                "client_ip": self.request.client.host if self.request.client else None,
                "user_agent": self.request.headers.get("user-agent"),
            }
        )
        
        # Log request body if present and not too large
        content_type = self.request.headers.get("content-type", "")
        if self.request.method in ["POST", "PUT", "PATCH"] and "application/json" in content_type:
            try:
                request_body = await self.request.json()
                await self.audit_logger.log_debug(
                    message="Request body",
                    request_id=self.request_id,
                    data={"body": request_body}
                )
            except json.JSONDecodeError as e:
                await self.audit_logger.log_warning(
                    message="Failed to parse request body as JSON",
                    request_id=self.request_id,
                    error=str(e)
                )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Log the completion of a request and any exceptions."""
        response_time = (time.time() - self.start_time) * 1000  # in milliseconds
        
        if exc_type is not None:
            # Log the error
            error_message = str(exc_val) if exc_val else "Unknown error"
            error_type = exc_type.__name__ if exc_type else "UnknownError"
            
            status_code = 500
            if hasattr(exc_val, 'status_code'):
                status_code = exc_val.status_code
            
            await self.audit_logger.log_error(
                message="Request processing failed",
                request_id=self.request_id,
                error=error_message,
                error_type=error_type,
                status_code=status_code,
                response_time_ms=response_time,
                metadata={
                    "method": self.request.method,
                    "url": str(self.request.url),
                    "client_ip": self.request.client.host if self.request.client else None,
                },
                exc_info=exc_val is not None
            )
            
            # Log metrics for the error
            await self.metrics_logger.log_metric(
                metric_name="http_request_duration_ms",
                value=response_time,
                tags={
                    "method": self.request.method,
                    "path": self.request.url.path,
                    "status_code": status_code,
                    "error": "true",
                    "error_type": error_type
                }
            )
            
            # Don't suppress the exception
            return False
        
        return True
    
    async def log_response(self, response: Response):
        """Log the response for a successful request."""
        response_time = (time.time() - self.start_time) * 1000  # in milliseconds
        
        # Log the response
        await self.audit_logger.log_response(
            request_id=self.request_id,
            status_code=response.status_code,
            headers=dict(response.headers),
            response_time_ms=response_time,
            metadata={
                "content_type": response.headers.get("content-type"),
                "content_length": response.headers.get("content-length")
            }
        )
        
        # Log performance metrics
        await self.metrics_logger.log_metric(
            metric_name="http_request_duration_ms",
            value=response_time,
            tags={
                "method": self.request.method,
                "path": self.request.url.path,
                "status_code": response.status_code,
                "endpoint": self.request.url.path.split("/")[1] if len(self.request.url.path.split("/")) > 1 else "root"
            }
        )

def log_endpoint():
    """Decorator to add logging to FastAPI route handlers."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the request object from the function arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                for arg in kwargs.values():
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if request is None:
                # If we can't find the request, just call the function
                return await func(*args, **kwargs)
            
            # Create a request context
            async with RequestContext(request) as context:
                try:
                    # Call the route handler
                    response = await func(*args, **kwargs)
                    
                    # If the response is a Response object, log it
                    if isinstance(response, Response):
                        await context.log_response(response)
                    
                    return response
                except Exception as e:
                    # The exception will be handled by the RequestContext.__aexit__ method
                    raise
        
        return wrapper
    return decorator

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip logging for certain paths (e.g., health checks)
        if request.url.path in ["/health", "/favicon.ico"]:
            return await call_next(request)
        
        # Create a request context
        async with RequestContext(request) as context:
            try:
                # Process the request
                response = await call_next(request)
                
                # Log the response
                await context.log_response(response)
                
                return response
                
            except Exception as e:
                # The exception will be handled by the RequestContext.__aexit__ method
                raise

def init_logging(app: FastAPI, log_level: str = "INFO"):
    """Initialize logging for a FastAPI application."""
    # Configure logging
    clickhouse_config = {
        "enabled": os.getenv("CLICKHOUSE_ENABLED", "false").lower() == "true",
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "9000")),
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        "database": os.getenv("CLICKHOUSE_DB", "default"),
        "table_name": os.getenv("CLICKHOUSE_TABLE", "audit_logs"),
        "batch_size": int(os.getenv("CLICKHOUSE_BATCH_SIZE", "100")),
        "flush_interval": int(os.getenv("CLICKHOUSE_FLUSH_INTERVAL", "10")),
        "required": os.getenv("CLICKHOUSE_REQUIRED", "false").lower() == "true"
    }
    
    # Set up logging
    metrics_logger, audit_logger = setup_logging(
        log_dir="logs",
        log_level=log_level,
        clickhouse_config=clickhouse_config
    )
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id_header(request: Request, call_next):
        """Add X-Request-ID header to all responses."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        """Log application startup."""
        await audit_logger.log_system_event(
            event_type="application_startup",
            status=LogStatus.SUCCESS,
            message="Application started successfully",
            metadata={
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        )
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Log application shutdown and clean up resources."""
        await audit_logger.log_system_event(
            event_type="application_shutdown",
            status=LogStatus.SUCCESS,
            message="Application is shutting down"
        )
        await shutdown_logging()
    
    return app
