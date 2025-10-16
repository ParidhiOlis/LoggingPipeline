"""
Logging utilities for the OLIS RAG Pipeline.

This module provides utilities for request/response logging, middleware, and
integration with the core logging configuration.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from fastapi import FastAPI, Request, Response, status
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

# Import core logging components
from LoggingPipeline.Logging.logSystem import AuditLogger, LogStatus, InputType
from LoggingPipeline.Logging.server_metrics_logger import ServerMetricsLogger, RequestStatus

# Import the core logging configuration
from app.logging_config import logging_config

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

# Global instances (for backward compatibility)
_audit_logger = None
_metrics_logger = None

# Alias for backward compatibility
def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = logging_config.get_audit_logger()
    return _audit_logger

def get_metrics_logger() -> ServerMetricsLogger:
    """Get the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = logging_config.get_metrics_logger()
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
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Log the completion of a request and any exceptions."""
        response_time = (time.time() - self.start_time) * 1000  # in milliseconds
        
        # Log the response or error
        if exc_type is not None:
            await self.audit_logger.log_error(
                message=f"Request failed: {str(exc_val)}",
                request_id=self.request_id,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                stack_trace=str(exc_tb) if exc_tb else None,
                response_time_ms=response_time
            )
        
        # Log metrics
        self.metrics_logger.log_request(
            endpoint=self.request.url.path,
            method=self.request.method,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR if exc_type else status.HTTP_200_OK,
            response_time_ms=response_time,
            error_type=exc_type.__name__ if exc_type else None
        )
        
        return False  # Don't suppress exceptions

    async def log_response(self, response: Response):
        """Log the response for a successful request."""
        response_time = (time.time() - self.start_time) * 1000  # in milliseconds
        
        await self.audit_logger.log_response(
            request_id=self.request_id,
            status_code=response.status_code,
            headers=dict(response.headers),
            response_time_ms=response_time
        )
        
        # Log metrics
        self.metrics_logger.log_request(
            endpoint=self.request.url.path,
            method=self.request.method,
            status_code=response.status_code,
            response_time_ms=response_time
        )

def log_endpoint():
    """Decorator to add logging to FastAPI route handlers."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the request object from the arguments
            request = next(
                (arg for arg in args if isinstance(arg, Request)),
                None
            )
            
            if not request:
                return await func(*args, **kwargs)
            
            async with RequestContext(request) as context:
                try:
                    response = await func(*args, **kwargs)
                    if isinstance(response, Response):
                        await context.log_response(response)
                    return response
                except Exception as e:
                    # The error will be logged by the context manager
                    raise
        
        return wrapper
    return decorator

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Process the request and log the response."""
        async with RequestContext(request) as context:
            try:
                response = await call_next(request)
                await context.log_response(response)
                return response
            except Exception as e:
                # The error will be logged by the context manager
                raise

def init_logging(app: FastAPI, log_level: str = "INFO"):
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Set up request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Log application startup
    @app.on_event("startup")
    async def startup_event():
        logging.info("Application startup complete")
    
    # Log application shutdown
    @app.on_event("shutdown")
    async def shutdown_event():
        await shutdown_logging()
        logging.info("Application shutdown complete")
