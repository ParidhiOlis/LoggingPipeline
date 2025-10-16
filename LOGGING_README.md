# OLIS Logging System

A comprehensive logging solution for the OLIS RAG Pipeline, providing both client-side and server-side logging capabilities with ClickHouse integration.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Privacy and Security](#privacy-and-security)
- [Installation](#installation)
- [Configuration](#configuration)
- [ClickHouse Setup](#clickhouse-setup)
- [Usage](#usage)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

The OLIS Logging System provides:
- **Privacy-Preserving Logging**: Secure logging with sensitive data protection
- **Audit Logging**: Track user actions and system events
- **Metrics Collection**: Monitor system performance and resource usage
- **Error Tracking**: Capture and analyze errors across the system
- **Real-time Monitoring**: Integration with monitoring tools

## Architecture

### Core Components

#### 1. Logging Pipeline
- `LoggingPipeline/Logging/logSystem.py`: Core audit logging with privacy protection
- `LoggingPipeline/Logging/logManager.py`: Log file management and rotation
- `LoggingPipeline/Logging/server_metrics_logger.py`: Server performance metrics
- `LoggingPipeline/Logging/Clickhouse_integration.py`: Secure database operations

#### 2. Application Integration
- `app/logging_config.py`: Centralized logging configuration
- `app/logging_utils.py`: Logging utilities and middleware
- `app/services/base_service.py`: Base service with built-in logging

## Privacy and Security

### Data Protection Measures
- **Input/Output Hashing**: All sensitive data is hashed before logging
- **No Raw Data Storage**: Only hashes and lengths of sensitive data are stored
- **Minimal Error Details**: Error messages are sanitized to remove PII
- **Request Body Protection**: Request bodies are hashed, not stored in plaintext

### Privacy by Design
- **No Query/Response Storage**: Actual queries and responses are never logged
- **Hashed Identifiers**: User-identifiable information is hashed
- **Controlled Logging**: Sensitive operations have minimal logging

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create a `.env` file):
```env
# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# ClickHouse Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=default

# Privacy Settings
HASH_SENSITIVE_DATA=true
LOG_REQUEST_BODY=false
LOG_RESPONSE_BODY=false
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
```

## ClickHouse Setup

### 1. Running ClickHouse with Docker

```bash
# Start ClickHouse container
docker run -d --name clickhouse-server \
    --ulimit nofile=262144:262144 \
    -p 8123:8123 -p 9000:9000 -p 9009:9009 \
    -e CLICKHOUSE_DB=default \
    -e CLICKHOUSE_USER=default \
    -e CLICKHOUSE_PASSWORD= \
    -e CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1 \
    clickhouse/clickhouse-server:23.3

# Verify it's running
docker ps | findstr clickhouse
```

### 2. Create Required Tables

Use the `scripts/init_clickhouse.py` script to set up the required tables:

```bash
python scripts/init_clickhouse.py
```

### 3. Verify Connection

```bash
# Connect to ClickHouse
clickhouse-client --host localhost --port 9000 --user default

# Check if tables were created
SHOW TABLES;
```

## Configuration

### Logging Levels
- `DEBUG`: Detailed debug information
- `INFO`: General operational logs
- `WARNING`: Indicates potential issues
- `ERROR`: Errors that don't prevent the application from running
- `CRITICAL`: Critical errors causing application failure

### Privacy Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `HASH_SENSITIVE_DATA` | `true` | Whether to hash sensitive data before logging |
| `LOG_REQUEST_BODY` | `false` | Whether to log request bodies (not recommended in production) |
| `LOG_RESPONSE_BODY` | `false` | Whether to log response bodies (not recommended in production) |

### ClickHouse Configuration
Edit `config/clickhouse_config.py` to configure database settings.

## Usage

### Basic Logging

```python
from app.logging_config import setup_logging, get_audit_logger

# Initialize logging
metrics_logger, audit_logger = setup_logging()

# Log an audit event
audit_logger.log_extraction(
    source_type="api",
    doc_id="doc123",
    status="SUCCESS",
    metadata={"user": "user123"}
)

# Log a metric
metrics_logger.log_request(
    endpoint="/api/query",
    method="POST",
    status_code=200,
    response_time_ms=150.5
)
```

### Request Logging Middleware

The logging middleware automatically logs all HTTP requests and responses:

```python
from fastapi import FastAPI
from app.logging_utils import init_logging

app = FastAPI()
init_logging(app)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_logging_integration.py

# Run with verbose output
pytest -v tests/
```

### Test Coverage

```bash
# Install coverage if not already installed
pip install pytest-cov

# Run tests with coverage
pytest --cov=app --cov=LoggingPipeline tests/

# Generate HTML report
pytest --cov=app --cov=LoggingPipeline --cov-report=html tests/
```

## Monitoring

### Log File Location
Logs are stored in the directory specified by `LOG_DIR` (default: `logs/`):

```
logs/
├── application.log    # Main application logs
├── queries/          # Query logs
├── extraction/       # Document extraction logs
└── indexing/         # Indexing operation logs
```

### ClickHouse Queries

Example queries to analyze logs in ClickHouse:

```sql
-- Get recent errors
SELECT * FROM audit_logs 
WHERE status >= 400 
ORDER BY logtime DESC 
LIMIT 10;

-- Get average response time by endpoint
SELECT 
    splitByString('?', endpoint)[1] as endpoint,
    avg(response_time_ms) as avg_response_time,
    count() as request_count
FROM request_logs 
GROUP BY endpoint
ORDER BY avg_response_time DESC;
```

## Troubleshooting

### Common Issues

1. **ClickHouse Connection Issues**
   - Verify ClickHouse is running: `docker ps | findstr clickhouse`
   - Check connection: `clickhouse-client --host localhost --port 9000`
   - Verify credentials in `.env` match ClickHouse configuration

2. **Permission Denied Errors**
   - Ensure the log directory is writable: `chmod -R 755 logs/`
   - Check ClickHouse user permissions

3. **Missing Logs**
   - Verify log level is set appropriately (DEBUG shows more details)
   - Check application logs for any initialization errors

### Debugging

To enable debug logging, set the log level to DEBUG:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

For additional support:
1. Check the application logs in `logs/application.log`
2. Review the ClickHouse server logs: `docker logs clickhouse-server`
3. Open an issue in the repository with relevant logs and steps to reproduce

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
