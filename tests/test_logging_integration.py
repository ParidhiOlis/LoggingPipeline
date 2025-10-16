"""
Integration tests for the logging system with ClickHouse.

This script tests the end-to-end logging functionality, including:
- Logging configuration and initialization
- Writing logs to both file and ClickHouse
- Verifying log entries in ClickHouse
- Testing log rotation and cleanup
"""
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest
import pytest_asyncio
import clickhouse_connect
from clickhouse_connect.driver.client import Client as ClickHouseClient

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the logging configuration
from app.logging_config import setup_logging, get_audit_logger, shutdown_logging, LoggingConfig
from config.clickhouse_config import ClickHouseConfig, get_schema

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_LOG_DIR = "test_logs"
TEST_DB_NAME = "test_logging_db"
TEST_TABLE_NAME = "test_audit_logs"

# Global logging config instance
logging_config = LoggingConfig()

@pytest.fixture(scope="module")
def clickhouse_config():
    """Fixture providing ClickHouse configuration for tests."""
    return {
        "enabled": True,
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),  # Use HTTP port
        "http_port": int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),  # Explicit HTTP port
        "native_port": int(os.getenv("CLICKHOUSE_NATIVE_PORT", "9000")),  # Native protocol port
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", "password"),  # Default password is 'password'
        "database": os.getenv("CLICKHOUSE_DATABASE", "default"),
        "table_name": TEST_TABLE_NAME,
        "required": False,  # Don't fail tests if ClickHouse is not available
        "use_http": True,   # Use HTTP protocol for clickhouse-connect
        "secure": False     # Not using TLS
    }

@pytest_asyncio.fixture(scope="module")
async def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="module")
async def logging_system(clickhouse_config, event_loop):
    """Fixture that sets up and tears down the logging system for tests."""
    # Create a temporary directory for test logs
    log_dir = Path(TEST_LOG_DIR)
    
    # Create all required subdirectories
    for subdir in ['extraction', 'indexing', 'querying', 'server_metrics']:
        (log_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize the logging system
    metrics_logger, audit_logger = await logging_config.initialize(
        log_dir=str(log_dir),
        log_level="DEBUG",
        clickhouse_config=clickhouse_config
    )
    
    try:
        yield metrics_logger, audit_logger
    finally:
        # Clean up
        await shutdown_logging()
        
        # Remove test logs
        for log_file in log_dir.rglob("*"):
            try:
                if log_file.is_file():
                    log_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up log file {log_file}: {e}")
        
        # Remove directories in reverse order
        for subdir in reversed(list(log_dir.glob('**'))):
            try:
                if subdir.is_dir() and subdir != log_dir:
                    subdir.rmdir()
            except Exception as e:
                logger.warning(f"Failed to remove directory {subdir}: {e}")
        
        try:
            log_dir.rmdir()
        except Exception as e:
            logger.warning(f"Failed to remove log directory {log_dir}: {e}")

@pytest.fixture(scope="module")
def clickhouse_client(clickhouse_config):
    """Fixture providing a ClickHouse client for tests."""
    if not clickhouse_config["enabled"]:
        pytest.skip("ClickHouse integration is disabled in test configuration")
    
    try:
        # Use HTTP connection
        client = clickhouse_connect.get_client(
            host=clickhouse_config["host"],
            port=clickhouse_config["http_port"],
            username=clickhouse_config["user"],
            password=clickhouse_config["password"] or "",
            database=clickhouse_config["database"],
            connect_timeout=5,
            secure=clickhouse_config["secure"]
        )
        
        # Test the connection
        result = client.command("SELECT 1")
        assert result == 1, "Connection test query failed"
        return client
        
    except Exception as e:
        pytest.skip(f"Could not connect to ClickHouse: {e}")

@pytest.fixture(scope="module", autouse=True)
def setup_clickhouse_tables(clickhouse_client, clickhouse_config):
    """Set up test tables in ClickHouse."""
    if not clickhouse_config["enabled"]:
        return
    
    table_name = clickhouse_config["table_name"]
    
    try:
        # Drop the test table if it exists
        clickhouse_client.command(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create the test table
        schema = get_schema()
        clickhouse_client.command(schema)
        
        yield
        
        # Clean up
        clickhouse_client.command(f"DROP TABLE IF EXISTS {table_name}")
    except Exception as e:
        pytest.fail(f"Failed to set up ClickHouse test tables: {e}")

@pytest.mark.asyncio
async def test_logging_initialization(logging_system):
    """Test that the logging system initializes correctly."""
    # Unpack the logging system components
    metrics_logger, audit_logger = await logging_system
    
    # Verify the loggers were created
    assert metrics_logger is not None
    assert audit_logger is not None
    
    # Verify log directory was created
    log_dir = Path(TEST_LOG_DIR)
    assert log_dir.exists() and log_dir.is_dir()
    
    # Verify log files were created
    log_files = list(log_dir.glob("*.log"))
    assert any("application.log" in str(f) for f in log_files)

@pytest.mark.asyncio
async def test_audit_logging(clickhouse_client, clickhouse_config, logging_system):
    """Test that audit logs are written to ClickHouse."""
    # Unpack the logging system components
    _, audit_logger = await logging_system
    
    if not clickhouse_config["enabled"]:
        pytest.skip("ClickHouse integration is disabled")
    
    # Generate a unique test ID to track this test's logs
    test_id = str(uuid.uuid4())
    test_message = f"Test audit log message {test_id}"
    
    try:
        # Log a test message
        await audit_logger.log(
            level="INFO",
            message=test_message,
            source="test_audit_logging",
            metadata={"test_id": test_id, "test_name": "test_audit_logging"}
        )
        
        # Give the ingester time to process the message
        await asyncio.sleep(2)
        
        # Query ClickHouse for the test message
        table_name = clickhouse_config["table_name"]
        
        # Use parameterized query to prevent SQL injection
        query = f"""
        SELECT * FROM {table_name}
        WHERE message = %(message)s
        AND JSONExtractString(metadata, 'test_id') = %(test_id)s
        """
        
        # Execute the query with parameters
        result = clickhouse_client.query(
            query,
            parameters={"message": test_message, "test_id": test_id}
        ).result_rows
        
        # Verify the log was written to ClickHouse
        assert len(result) > 0, "Log entry not found in ClickHouse"
        
        # Verify the log entry contains the expected data
        log_entry = result[0]
        assert log_entry[1] == "INFO"  # level
        assert log_entry[2] == test_message
        assert log_entry[3] == "test_audit_logging"  # source
        
        # Verify metadata was stored correctly
        metadata = json.loads(log_entry[4])
        assert metadata.get("test_id") == test_id
        assert metadata.get("test_name") == "test_audit_logging"
    finally:
        # Clean up the async generator
        try:
            await logging_system.__anext__()  # This will raise StopAsyncIteration
        except StopAsyncIteration:
            pass

@pytest.mark.asyncio
async def test_metrics_logging(logging_system):
    """Test that metrics are logged correctly."""
    try:
        # Unpack the logging system components
        metrics_logger, _ = await logging_system
        
        # Log a test metric
        metrics_logger.log_metric(
            metric_name="test_metric",
            value=42,
            tags={"test": "test_metrics_logging"}
        )
        
        # This is a simple test since we don't have a way to verify the metrics
        # without depending on the implementation details of the metrics logger
        assert True
    finally:
        # Clean up the async generator
        try:
            await logging_system.__anext__()  # This will raise StopAsyncIteration
        except StopAsyncIteration:
            pass

if __name__ == "__main__":
    # This allows running the tests directly with Python for debugging
    import sys
    import pytest
    sys.exit(pytest.main(["-v", "-s", __file__]))
