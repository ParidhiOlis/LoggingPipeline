"""
Adapter for the existing ClickHouse integration to work with our logging system.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import the existing ClickHouse integration
from LoggingPipeline.Logging.Clickhouse_integration import (
    ClickHouseConfig,
    ClickHouseAuditIngester
)

class ClickHouseLogger:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'host': 'localhost',
            'port': 8123,
            'username': 'default',
            'password': '',
            'database': 'audit_logs',
            'secure': False
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Override with environment variables if they exist
        self.config.update({
            'host': self.config.get('host') or 'localhost',
            'port': int(self.config.get('port', 8123)),
            'username': self.config.get('username') or 'default',
            'password': self.config.get('password') or '',
            'database': self.config.get('database') or 'audit_logs',
            'secure': self.config.get('secure', False)
        })
        
        # Initialize ClickHouse client
        self.ingester = None
        self._connect()
    
    def _connect(self):
        """Establish connection to ClickHouse."""
        try:
            ch_config = ClickHouseConfig(
                host=self.config['host'],
                port=self.config['port'],
                username=self.config['username'],
                password=self.config['password'],
                database=self.config['database'],
                secure=self.config['secure']
            )
            self.ingester = ClickHouseAuditIngester(ch_config)
            self.logger.info("Successfully connected to ClickHouse")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            self.ingester = None
            return False
    
    def log_api_request(self, request_data: Dict[str, Any]) -> bool:
        if not self.ingester:
            if not self._connect():
                return False
        
        try:
            # Format the log entry to match what ClickHouseAuditIngester expects
            log_entry = {
                'timestamp': request_data.get('timestamp', datetime.utcnow().isoformat()),
                'level': 'INFO',
                'logger': 'api.request',
                'message': f"{request_data.get('method', '')} {request_data.get('endpoint', '')} - {request_data.get('status_code', 0)}",
                'request_id': request_data.get('request_id', ''),
                'method': request_data.get('method', ''),
                'endpoint': request_data.get('endpoint', ''),
                'status_code': request_data.get('status_code', 0),
                'duration_ms': request_data.get('duration_ms', 0),
                'client_ip': request_data.get('client_ip', ''),
                'user_agent': request_data.get('user_agent', ''),
                'error': request_data.get('error', ''),
                'error_type': request_data.get('error_type', '')
            }
            
            # Insert the log entry
            self.ingester._insert_batch('api_requests', [log_entry], {})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log API request to ClickHouse: {str(e)}", exc_info=True)
            return False
    
    def log_application_log(self, log_data: Dict[str, Any]) -> bool:
        if not self.ingester:
            if not self._connect():
                return False
        
        try:
            # Format the log entry to match what ClickHouseAuditIngester expects
            log_entry = {
                'timestamp': log_data.get('timestamp', datetime.utcnow().isoformat()),
                'level': log_data.get('level', 'INFO'),
                'logger': log_data.get('logger', 'application'),
                'message': log_data.get('message', ''),
                'request_id': log_data.get('request_id', ''),
                'exception': log_data.get('exception', ''),
                'stack_trace': log_data.get('stack_trace', '')
            }
            
            # Add any extra fields
            if 'extra' in log_data and isinstance(log_data['extra'], dict):
                log_entry.update(log_data['extra'])
            
            # Insert the log entry
            self.ingester._insert_batch('application_logs', [log_entry], {})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log application log to ClickHouse: {str(e)}", exc_info=True)
            return False
    
    def close(self):
        if self.ingester:
            try:
                self.ingester.close()
                self.ingester = None
            except Exception as e:
                self.logger.error(f"Error closing ClickHouse connection: {str(e)}")


def get_clickhouse_logger(config: Optional[Dict[str, Any]] = None) -> ClickHouseLogger:
    return ClickHouseLogger(config)
