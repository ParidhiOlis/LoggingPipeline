# test_logging_basic.py
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from Logging.logSystem import AuditLogger, LogStatus, InputType
from Logging.logManager import LogFileManager
from structlog import configure
from structlog.stdlib import add_log_level, filter_by_level, LoggerFactory, BoundLogger
from structlog.processors import JSONRenderer, TimeStamper
from structlog.types import Processor

def configure_logging(log_dir: Path, log_level: str = "INFO") -> None:
    """Configure structlog to write to files with proper formatting"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define processors
    processors: List[Processor] = [
        filter_by_level,
        add_log_level,
        TimeStamper(fmt="iso", utc=True),
        JSONRenderer()
    ]
    
    # Configure structlog
    configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_dir / "test_audit.log")
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level.upper())
    logger.addHandler(file_handler)

def test_basic_logging():
    """Test basic logging functionality"""
    # Set up test directory
    test_dir = Path("test_logs_basic")
    test_dir.mkdir(exist_ok=True)
    print(f"Using test directory: {test_dir.absolute()}")
    
    # Configure logging
    configure_logging(test_dir, "DEBUG")
    
    # Initialize audit logger
    audit_logger = AuditLogger(
        base_log_dir=test_dir,
        log_level="DEBUG",
        platform="test_platform"
    )
    
    # Test extraction log
    print("\nTesting extraction log...")
    audit_logger.log_extraction(
        ingestion_time=datetime.now(timezone.utc).isoformat(),
        source_type="test_source",
        doc_id="test_doc_123",
        status=LogStatus.SUCCESS
    )
    
    # Test query log
    print("Testing query log...")
    audit_logger.log_query(
        input_type=InputType.QUESTION,
        input_value="What is the test query?",
        input_tokens=10,
        dependency_id="dep_123",
        status=200,
        error="",
        output_val="This is a test response",
        zero_response=False,
        output_files=[],
        output_tokens=20,
        response_time=0.5
    )
    
    # Flush logs to ensure they're written to disk
    logging.shutdown()
    
    # List generated log files
    print("\nGenerated log files:")
    log_files = list(test_dir.glob('*.log'))
    
    if not log_files:
        print("No log files were generated!")
        return
    
    for log_file in log_files:
        print(f"\nContents of {log_file.relative_to(test_dir)}:")
        print("=" * 50)
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Pretty print JSON if possible
                    log_entry = json.loads(line)
                    print(json.dumps(log_entry, indent=2))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print("Raw content:", line)
        print("=" * 50)

if __name__ == "__main__":
    test_basic_logging()
    print("\nTest completed. Check the output above for results.")