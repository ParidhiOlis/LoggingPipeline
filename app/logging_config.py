"""Centralized logging configuration for the OLIS RAG Pipeline."""

import logging
import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from LoggingPipeline.Logging.server_metrics_logger import ServerMetricsLogger, RequestStatus, ErrorType
from LoggingPipeline.Logging.logSystem import AuditLogger, LogStatus, InputType
from LoggingPipeline.Logging.Clickhouse_integration import ClickHouseAuditIngester, ClickHouseConfig
from app.services.service_factory import ServiceFactory

logger = logging.getLogger(__name__)

# Global instance of the audit logger
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
    ) -> Tuple[ServerMetricsLogger, AuditLogger]:
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
                logging.StreamHandler(sys.stdout),
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
            
            logger.info("ClickHouse logging integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse logging: {str(e)}")
            if self._clickhouse_config.get("required", False):
                raise
    
    async def shutdown(self):
        """Shut down the logging system and clean up resources."""
        if not self._initialized:
            return
            
        logger.info("Shutting down logging system...")
        
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
        logger.info("Logging system shut down successfully")

# Global instance
logging_config = LoggingConfig()

# Backward compatibility
def setup_logging(
    log_dir: str = "logs", 
    log_level: str = "INFO",
    clickhouse_config: Optional[Dict[str, Any]] = None
) -> Tuple[ServerMetricsLogger, AuditLogger]:
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
    clickhouse_logger = None
    
    if clickhouse_enabled:
        try:
            clickhouse_logger = get_clickhouse_logger({
                'host': os.getenv('CLICKHOUSE_HOST', 'localhost'),
                'port': os.getenv('CLICKHOUSE_PORT', '8123'),
                'username': os.getenv('CLICKHOUSE_USER', 'default'),
                'password': os.getenv('CLICKHOUSE_PASSWORD', ''),
                'database': os.getenv('CLICKHOUSE_DB', 'audit_logs'),
                'secure': os.getenv('CLICKHOUSE_SECURE', 'false').lower() == 'true'
            })
            logger.info("ClickHouse logging enabled and configured")
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse logger: {str(e)}")
    
    # Initialize loggers
    metrics_logger = ServerMetricsLogger()
    audit_logger = AuditLogger()
    
    # Add ClickHouse logging to metrics logger if enabled
    if clickhouse_logger:
        def log_to_clickhouse(endpoint: str, platform: str, method: str, status_code: int, 
                             status: RequestStatus, latency_ms: float, input_token_count: int, 
                             output_token_count: int, error_type: Optional[ErrorType] = None, 
                             request_id: Optional[str] = None):
            """Helper function to log metrics to ClickHouse"""
            clickhouse_logger.log_api_request({
                'endpoint': endpoint,
                'platform': platform,
                'method': method,
                'status_code': status_code,
                'status': status.name,
                'latency_ms': latency_ms,
                'input_token_count': input_token_count,
                'output_token_count': output_token_count,
                'error_type': error_type.name if error_type else None,
                'request_id': request_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Replace the default log_api_request method with our ClickHouse-enabled one
        metrics_logger.log_api_request = log_to_clickhouse
    
    # Configure audit logger to use ClickHouse if enabled
    if clickhouse_logger:
        def audit_log_to_clickhouse(message: str, level: str = "INFO", **kwargs):
            """Helper function to log audit events to ClickHouse"""
            clickhouse_logger.log_application_log({
                'level': level.upper(),
                'message': message,
                'logger': 'audit',
                'request_id': kwargs.get('request_id'),
                'extra': kwargs
            })
        
        # Replace the default log method with our ClickHouse-enabled one
        audit_logger.log = audit_log_to_clickhouse
    
    logger.info("Logging system initialized")
    logger.info(f"Logging to directory: {log_path.absolute()}")
    
    return metrics_logger, audit_logger
