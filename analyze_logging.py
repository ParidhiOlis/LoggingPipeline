"""
Script to analyze and verify logging functionality.
This script helps verify that the logging system is working correctly.
"""
import sys
import os
import platform
import time
import asyncio
import datetime
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.logging_config import setup_logging, get_audit_logger, get_metrics_logger
from LoggingPipeline.Logging.server_metrics_logger import ServerMetricsLogger, RequestStatus
from LoggingPipeline.Logging.logSystem import AuditLogger, LogStatus, InputType
from LoggingPipeline.Logging.Clickhouse_integration import ClickHouseAuditIngester, ClickHouseConfig
from app.services.service_factory import ServiceFactory
from app.logging_config import LoggingConfig

# Global logging config instance
logging_config = LoggingConfig()

async def setup_logging_async(
    log_dir: str = "logs",
    log_level: str = "INFO",
    clickhouse_config: dict = None
) -> tuple[ServerMetricsLogger, AuditLogger]:
    """
    Set up logging asynchronously.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
        clickhouse_config: ClickHouse configuration
        
    Returns:
        Tuple of (metrics_logger, audit_logger)
    """
    return await logging_config.initialize(
        log_dir=log_dir,
        log_level=log_level,
        clickhouse_config=clickhouse_config or {}
    )

def print_banner():
    """Print a banner for the logging analysis."""
    print("=" * 80)
    print("OLIS Logging System Analysis")
    print("=" * 80)
    print("This script will verify the logging system configuration and functionality.\n")

def check_clickhouse_connection():
    """Check if ClickHouse is running and initialize the database."""
    print("\nChecking ClickHouse connection and initializing database...")
    try:
        from clickhouse_driver import Client
        from config.clickhouse_config import ClickHouseConfig, initialize_database, default_config
        
        # Get configuration from environment variables
        config = ClickHouseConfig.from_env()
        
        # Create a ClickHouse client using clickhouse_driver
        client = Client(
            host=config.host,
            port=config.port,  # Default native protocol port (9000)
            user=config.user,
            password=config.password,
            database='default'  # Connect to default database first
        )
        
        # Initialize the database
        success = initialize_database(client)
        
        if success:
            print("✅ ClickHouse database initialized successfully")
            return True
        else:
            print("❌ Failed to initialize ClickHouse database")
            return False
    except Exception as e:
        print(f"❌ Error connecting to ClickHouse: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_logging_async():
    """Test basic logging functionality in an async context."""
    print("\nTesting logging functionality...")
    
    try:
        # Set up logging
        metrics_logger, audit_logger = setup_logging(
            log_dir="test_logs",
            log_level="DEBUG"
        )
        
        # Test audit logging
        print("Testing audit logging...")
        ingestion_time = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "test_file"
        doc_id = "test_doc_123"
        status = LogStatus.SUCCESS
        error = ""
        
        log_data = {
            'ingestionTime': ingestion_time,
            'sourceType': source_type,
            'docId': doc_id,
            'status': status.value if hasattr(status, 'value') else status,
            'error': error,
            'logtime': datetime.datetime.utcnow().isoformat()  # Add logtime field
        }
        
        log_id = await audit_logger.log_extraction(log_data)
        print(f"✅ Logged test message with ID: {log_id}")
        
        # Test metrics logging
        print("\nTesting metrics logging...")
        metrics_logger.log_api_request(
            endpoint="/test",
            method="GET",
            status_code=200,
            latency_ms=42.5,
            input_token_count=10,
            output_token_count=20,
            model_name="test_model",
            status=RequestStatus.SUCCESS
        )
        print("✅ Logged test metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during logging test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_logging_async_internal():
    """Internal async test function that handles logging setup and testing."""
    try:
        # First, set up logging
        metrics_logger, audit_logger = await setup_logging_async(
            log_dir="test_logs",
            log_level="DEBUG"
        )
        
        # Test audit logging
        print("Testing audit logging...")
        ingestion_time = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "test_file"
        doc_id = "test_doc_123"
        status = LogStatus.SUCCESS
        error = ""
        
        log_data = {
            'ingestionTime': ingestion_time,
            'sourceType': source_type,
            'docId': doc_id,
            'status': status.value if hasattr(status, 'value') else status,
            'error': error,
            'logtime': datetime.datetime.utcnow().isoformat()
        }
        
        log_id = await audit_logger.log_extraction(log_data)
        print(f"✅ Logged test message with ID: {log_id}")
        
        # Test metrics logging
        print("\nTesting metrics logging...")
        metrics_logger = get_metrics_logger()
        if metrics_logger is None:
            print("❌ Metrics logger not initialized")
            return False
            
        # Log an API request (using the available method)
        metrics_logger.log_api_request(
            endpoint="/test",
            method="GET",
            status_code=200,
            latency_ms=42.5,
            input_token_count=10,
            output_token_count=20,
            status=RequestStatus.SUCCESS,
            platform="test_platform"  # Add the required platform parameter
        )
        print("✅ Logged test API request")
        
        # Log a simple counter if the method exists
        if hasattr(metrics_logger, 'increment_counter'):
            metrics_logger.increment_counter("test_counter", tags={"test": "test_metrics_logging"})
            print("✅ Logged test counter")
        else:
            print("ℹ️  increment_counter method not available in ServerMetricsLogger")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during logging test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up - no need to shut down the audit logger as it's managed by the logging system
        pass

def test_logging():
    """Test logging in a synchronous context."""
    try:
        # Set up logging first
        setup_logging(
            log_dir="test_logs",
            log_level="DEBUG"
        )
        
        # Now run the test
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, run the async version directly
            return loop.create_task(test_logging_async_internal())
        else:
            # Otherwise, run it in the event loop
            return loop.run_until_complete(test_logging_async_internal())
    except Exception as e:
        print(f"❌ Error setting up logging: {e}")
        return False

async def main():
    """Main function to run logging analysis."""
    print_banner()
    
    # Check ClickHouse connection
    clickhouse_ok = check_clickhouse_connection()
    if not clickhouse_ok:
        print("❌ Cannot proceed without ClickHouse connection")
        return
    
    # Run tests
    test_result = await test_logging_async_internal()
    
    if test_result:
        print("\n✅ Logging system analysis completed successfully")
    else:
        print("\n❌ Logging system analysis completed with errors")
        sys.exit(1)

def run_async(coro):
    """Run an async function in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

if __name__ == "__main__":
    try:
        run_async(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Ensure we clean up any running loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.close()
        except:
            pass
