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

class LoggingContext:
    
    def __init__(self, audit_logger, metrics_logger):
        self.audit_logger = audit_logger
        self.metrics_logger = metrics_logger
        self._loop = None
    
    async def log_query_async(self, log_data: Dict[str, Any]) -> str:
        """Log a query asynchronously."""
        return await self.audit_logger.log_query(log_data)
    
    def log_query_sync(self, log_data: Dict[str, Any]) -> str:
        """Log a query synchronously."""
        if not self._loop:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        return self._loop.run_until_complete(
            self.audit_logger.log_query(log_data)
        )
    
    async def log_extraction_async(self, log_data: Dict[str, Any]) -> str:
        """Log an extraction event asynchronously."""
        return await self.audit_logger.log_extraction(log_data)
    
    def log_extraction_sync(self, log_data: Dict[str, Any]) -> str:
        """Log an extraction event synchronously."""
        if not self._loop:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        
        return self._loop.run_until_complete(
            self.audit_logger.log_extraction(log_data)
        )


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
        self._logging_context = None
    
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
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If we're in an async context, create a new loop to avoid nesting
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(
                logging_config.initialize(log_dir, log_level, clickhouse_config)
            )
        finally:
            new_loop.close()
    else:
        # If we're not in an async context, use the current loop
        try:
            return loop.run_until_complete(
                logging_config.initialize(log_dir, log_level, clickhouse_config)
            )
        finally:
            # Clean up the loop if we created it
            try:
                if loop.is_running():
                    loop.close()
            except:
                pass

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
    try:
        await logging_config.shutdown()
        logger.info("Logging system shut down successfully")
    except Exception as e:
        logger.error(f"Error during logging shutdown: {e}", exc_info=True)
        raise
