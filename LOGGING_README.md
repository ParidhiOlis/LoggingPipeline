# OLIS Logging System

A comprehensive logging solution for the OLIS RAG Pipeline, providing both client-side and server-side logging capabilities with ClickHouse integration.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

The OLIS Logging System provides:
- **Audit Logging**: Track user actions and system events
- **Metrics Collection**: Monitor system performance and resource usage
- **Error Tracking**: Capture and analyze errors across the system
- **Real-time Monitoring**: Web-based dashboard for monitoring logs and metrics

## Architecture

### Core Components

#### 1. Logging Pipeline
- `LoggingPipeline/Logging/logSystem.py`: Core audit logging functionality
- `LoggingPipeline/Logging/logManager.py`: Log file management and rotation
- `LoggingPipeline/Logging/server_metrics_logger.py`: Server performance metrics
- `LoggingPipeline/Logging/Clickhouse_integration.py`: Database operations

#### 2. Application Integration
- `app/logging_config.py`: Centralized logging configuration
- `app/log_rotation.py`: Log rotation settings
- `app/services/base_service.py`: Base service with logging capabilities

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file):
```env
LOG_LEVEL=INFO
LOG_DIR=logs
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
```

## Configuration

### Logging Levels
- `DEBUG`: Detailed debug information
- `INFO`: General operational logs
- `WARNING`: Indicates potential issues
- `ERROR`: Errors that don't prevent the application from running
- `CRITICAL`: Critical errors causing application failure

### ClickHouse Configuration
Edit `config/clickhouse_config.py` to configure database settings.

## Usage

### Basic Logging

```python
from app.logging_config import setup_logging, get_audit_logger

# Initialize logging
metrics_logger, audit_logger = setup_logging()

# Log an event
audit_logger.log_extraction(
    source_type="test",
    doc_id="doc123",
    status="SUCCESS",
    message="Test log entry"
)
```

### Server Metrics

```python
from LoggingPipeline.Logging.server_metrics_logger import ServerMetricsLogger

# Initialize metrics logger
metrics_logger = ServerMetricsLogger()

# Log API request
metrics_logger.log_api_request(
    endpoint="/api/test",
    platform="test",
    method="GET",
    status_code=200,
    status="SUCCESS",
    latency_ms=150,
    input_token_count=100,
    output_token_count=500
)
```

## Testing

The test suite verifies all logging functionality:

### Test Files

1. **Integration Tests** (`tests/test_logging_integration.py`)
   - `test_logging_initialization`: Verifies logging system startup
   - `test_audit_logging`: Tests audit log creation and retrieval
   - `test_metrics_logging`: Validates metrics collection

2. **Unit Tests** (in development)
   - `test_log_rotation.py`: Tests log rotation functionality
   - `test_clickhouse_integration.py`: Verifies database operations

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_logging_integration.py -v

# Run with coverage report
pytest --cov=LoggingPipeline tests/
```

### Key Metrics Tracked
- API request rates and latencies
- Error rates and types
- System resource usage (CPU, memory, GPU)
- Token usage and limits

### Log Files
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`
- Access logs: `logs/access.log`
