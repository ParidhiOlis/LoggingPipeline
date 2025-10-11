"""
Script to analyze and verify logging functionality.
This script helps verify that the logging system is working correctly.
"""
import sys
import platform
import time
from pathlib import Path

# Add the project root to the Python path
root_dir = str(Path(__file__).parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from app.logging_config import setup_logging, get_audit_logger, get_metrics_logger
from config.clickhouse_config import ClickHouseConfig, initialize_database
from LoggingPipeline.Logging.logSystem import LogStatus
from LoggingPipeline.Logging.server_metrics_logger import RequestStatus

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
        if initialize_database():
            print("✅ ClickHouse database initialized successfully")
            return True
        else:
            print("❌ Failed to initialize ClickHouse database")
            return False
    except Exception as e:
        print(f"❌ Error connecting to ClickHouse: {e}")
        return False

def test_logging():
    """Test basic logging functionality."""
    print("\nTesting logging functionality...")
    
    # Set up logging
    metrics_logger, audit_logger = setup_logging(
        log_dir="test_logs",
        log_level="DEBUG"
    )
    
    try:
        # Test audit logging
        print("Testing audit logging...")
        ingestion_time = time.strftime("%Y-%m-%d %H:%M:%S")
        source_type = "test_file"
        doc_id = "test_doc_123"
        status = LogStatus.SUCCESS
        error = ""
        
        log_id = audit_logger.log_extraction(
            ingestion_time=ingestion_time,
            source_type=source_type,
            doc_id=doc_id,
            status=status,
            error=error
        )
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
            status=RequestStatus.SUCCESS,
            platform="test_platform"
        )
        print("✅ Logged test metrics")
        return True
    except Exception as e:
        print(f"❌ Error during logging test: {e}")
        return False

def main():
    """Main function to run logging analysis."""
    print_banner()
    
    # Check ClickHouse connection
    if not check_clickhouse_connection():
        print("\n⚠️  Some tests may fail without a working ClickHouse connection")
    
    # Test logging functionality
    if test_logging():
        print("\n✅ Logging system analysis complete")
    else:
        print("\n❌ Logging system analysis completed with errors")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
