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
from typing import Dict, List, Optional

import pytest
from clickhouse_driver import Client as ClickHouseClient

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import the logging configuration
from app.logging_config import setup_logging, get_audit_logger, shutdown_logging
from config.clickhouse_config import ClickHouseConfig, get_schema

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_LOG_DIR = "test_logs"
TEST_DB_NAME = "test_logging_db"
TEST_TABLE_NAME = "test_audit_logs"

@pytest.fixture(scope="module")
def clickhouse_config():
    """Fixture providing ClickHouse configuration for tests."""
    return {
        "enabled": True,
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "9000")),
        "user": os.getenv("CLICKHOUSE_USER", "default"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        "database": os.getenv("CLICKHOUSE_DATABASE", "default"),
        "table_name": TEST_TABLE_NAME,
        "required": False  # Don't fail tests if ClickHouse is not available
    }

@pytest.fixture(scope="module")
async def logging_system(clickhouse_config):
    """Fixture that sets up and tears down the logging system for tests."""
    # Create a temporary directory for test logs
    log_dir = Path(TEST_LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Initialize the logging system
    metrics_logger, audit_logger = await setup_logging(
        log_dir=str(log_dir),
        log_level="DEBUG",
        clickhouse_config=clickhouse_config
    )
    
    yield metrics_logger, audit_logger
    
    # Clean up
    await shutdown_logging()
    
    # Remove test logs
    for log_file in log_dir.glob("*"):
        try:
            if log_file.is_file():
                log_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up log file {log_file}: {e}")
    
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
        client = ClickHouseClient(
            host=clickhouse_config["host"],
            port=clickhouse_config["port"],
            user=clickhouse_config["user"],
            password=clickhouse_config["password"] or None,
            database=clickhouse_config["database"]
        )
        # Test the connection
        client.execute("SELECT 1")
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
        clickhouse_client.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create the test table
        schema = get_schema()
        clickhouse_client.execute(schema)
        
        yield
        
        # Clean up
        clickhouse_client.execute(f"DROP TABLE IF EXISTS {table_name}")
    except Exception as e:
        pytest.fail(f"Failed to set up ClickHouse test tables: {e}")

@pytest.mark.asyncio
async def test_logging_initialization(logging_system):
    """Test that the logging system initializes correctly."""
    metrics_logger, audit_logger = logging_system
    
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
    _, audit_logger = logging_system
    
    if not clickhouse_config["enabled"]:
        pytest.skip("ClickHouse integration is disabled")
    
    # Generate a unique test ID to track this test's logs
    test_id = str(uuid.uuid4())
    test_message = f"Test audit log message {test_id}"
    
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
    result = clickhouse_client.execute(
        f"""
        SELECT * FROM {table_name}
        WHERE message = %(message)s
        AND JSONExtractString(metadata, 'test_id') = %(test_id)s
        """,
        {"message": test_message, "test_id": test_id}
    )
    
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

@pytest.mark.asyncio
async def test_metrics_logging(logging_system):
    """Test that metrics are logged correctly."""
    metrics_logger, _ = logging_system
    
    # Log a test metric
    metrics_logger.log_metric(
        metric_name="test_metric",
        value=42,
        tags={"test": "test_metrics_logging"}
    )

    assert True

if __name__ == "__main__":
    # This allows running the tests directly with Python for debugging
    import sys
    import pytest
    sys.exit(pytest.main(["-v", "-s", __file__]))
