"""ClickHouse configuration for the OLIS RAG Pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ClickHouseConfig:
    """Configuration for ClickHouse connection and tables."""
    
    host: str = "localhost"
    port: int = 9000
    user: str = "default"
    password: str = "password"
    database: str = "default"
    use_http: bool = False
    table_name: str = "audit_logs"
    batch_size: int = 100
    flush_interval: int = 10
    min_pool_size: int = 1
    max_pool_size: int = 10
    connect_timeout: int = 10
    send_receive_timeout: int = 30
    sync_request_timeout: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    secure: bool = False
    verify: bool = False
    settings: Optional[Dict[str, str]] = None
    
    def get_connection_params(self) -> Dict[str, any]:
        """Get connection parameters as a dictionary."""
        if self.use_http:
            return {
                'url': f'http://{self.host}:{self.port}/',
                'database': self.database,
                'user': self.user,
                'password': self.password,
                'settings': self.settings or {}
            }
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "secure": self.secure,
            "verify": self.verify,
            "settings": self.settings or {},
            "connect_timeout": self.connect_timeout,
            "send_receive_timeout": self.send_receive_timeout,
            "sync_request_timeout": self.sync_request_timeout,
        }
    
    @classmethod
    def from_env(cls) -> 'ClickHouseConfig':
        """Create a ClickHouseConfig from environment variables."""
        import os
        
        return cls(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
            user=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", "password"),
            database=os.getenv("CLICKHOUSE_DB", "default"),
            table_name=os.getenv("CLICKHOUSE_TABLE", "audit_logs"),
            batch_size=int(os.getenv("CLICKHOUSE_BATCH_SIZE", "100")),
            flush_interval=int(os.getenv("CLICKHOUSE_FLUSH_INTERVAL", "10")),
            min_pool_size=int(os.getenv("CLICKHOUSE_MIN_POOL_SIZE", "1")),
            max_pool_size=int(os.getenv("CLICKHOUSE_MAX_POOL_SIZE", "10")),
            connect_timeout=int(os.getenv("CLICKHOUSE_CONNECT_TIMEOUT", "10")),
            send_receive_timeout=int(os.getenv("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "30")),
            sync_request_timeout=int(os.getenv("CLICKHOUSE_SYNC_REQUEST_TIMEOUT", "5")),
            retry_attempts=int(os.getenv("CLICKHOUSE_RETRY_ATTEMPTS", "3")),
            retry_delay=float(os.getenv("CLICKHOUSE_RETRY_DELAY", "1.0")),
            secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
            verify=os.getenv("CLICKHOUSE_VERIFY_SSL", "false").lower() == "true",
        )

# Default configuration
default_config = ClickHouseConfig()

# Create a configuration from environment variables
env_config = ClickHouseConfig.from_env()

def get_schema() -> str:
    """Get the SQL schema for creating the audit logs table."""
    return """
    CREATE TABLE IF NOT EXISTS audit_logs (
        -- Log metadata
        log_id String,
        timestamp DateTime64(6, 'UTC') CODEC(Delta, ZSTD),
        level String,
        logger String,
        message String,
        
        -- Request/response info
        request_id String,
        method String,
        url String,
        status_code UInt16,
        response_time_ms Float64,
        
        -- Error info
        error String DEFAULT '',
        error_type String DEFAULT '',
        stack_trace String DEFAULT '',
        
        -- User/authentication
        user_id String DEFAULT '',
        client_ip String,
        user_agent String,
        
        -- Additional metadata
        metadata String,
        
        -- Indexes
        INDEX idx_request_id request_id TYPE bloom_filter GRANULARITY 3,
        INDEX idx_timestamp timestamp TYPE minmax GRANULARITY 3,
        INDEX idx_level level TYPE bloom_filter GRANULARITY 3,
        INDEX idx_status_code status_code TYPE minmax GRANULARITY 3,
        INDEX idx_method method TYPE bloom_filter GRANULARITY 3,
        INDEX idx_url url TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1
    )
    ENGINE = MergeTree()
    ORDER BY (toDate(timestamp), log_id)
    TTL toDateTime(timestamp) + INTERVAL 90 DAY
    SETTINGS index_granularity = 8192, ttl_only_drop_parts = 1;
    """

def get_retention_policy_sql() -> str:
    """Get the SQL for setting up retention policies."""
    return """
    -- Example retention policy: move old data to a different volume after 30 days
    ALTER TABLE audit_logs 
    MODIFY TTL 
        toDateTime(timestamp) + INTERVAL 30 DAY TO VOLUME 'slow',
        toDateTime(timestamp) + INTERVAL 90 DAY DELETE;
    """

def get_materialized_views() -> Dict[str, str]:
    """Get SQL for creating materialized views for common queries."""
    return {
        "daily_stats": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_stats
        ENGINE = SummingMergeTree()
        ORDER BY (date, status_code, method, endpoint)
        POPULATE
        AS SELECT
            toDate(timestamp) AS date,
            status_code,
            method,
            splitByChar('?', splitByChar(' ', url)[2])[1] AS endpoint,
            count() AS requests,
            avg(response_time_ms) AS avg_response_time_ms,
            quantile(0.95)(response_time_ms) AS p95_response_time_ms,
            sum(if(startsWith(level, '4') OR startsWith(level, '5'), 1, 0)) AS error_count
        FROM audit_logs
        GROUP BY date, status_code, method, endpoint;
        """,
        
        "error_rates": """
        CREATE MATERIALIZED VIEW IF NOT EXISTS error_rates
        ENGINE = AggregatingMergeTree()
        ORDER BY (date, hour, endpoint, status_code)
        POPULATE
        AS SELECT
            toDate(timestamp) AS date,
            toHour(timestamp) AS hour,
            splitByChar('?', splitByChar(' ', url)[2])[1] AS endpoint,
            status_code,
            count() AS count
        FROM audit_logs
        WHERE status_code >= 400
        GROUP BY date, hour, endpoint, status_code;
        """
    }

def get_indexes() -> List[Dict[str, str]]:
    """Get additional indexes to improve query performance."""
    return [
        {
            "name": "idx_user_agent",
            "sql": "INDEX idx_user_agent user_agent TYPE tokenbf_v1(32768, 3, 0) GRANULARITY 1"
        },
        {
            "name": "idx_error_type",
            "sql": "INDEX idx_error_type error_type TYPE bloom_filter GRANULARITY 3"
        }
    ]

def get_ttl_settings():
    """Get TTL settings for the audit logs table."""
    return """
    TTL toDate(logtime) + INTERVAL 30 DAY
    """.strip()

def initialize_database(client=None):
    """Initialize the ClickHouse database and create tables if they don't exist."""
    from clickhouse_driver import Client
    
    # Create a client if one wasn't provided
    close_client = False
    if client is None:
        client = Client(
            host=default_config.host,
            port=default_config.port,
            user=default_config.user,
            password=default_config.password,
            database='default'  # Connect to default database first
        )
        close_client = True
    
    try:
        # Create the database if it doesn't exist
        client.execute(f"CREATE DATABASE IF NOT EXISTS {default_config.database}")
        
        # Switch to the database
        client.database = default_config.database
        
        # Create the audit_logs table if it doesn't exist
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {default_config.table_name} (
            log_id String,
            logtime DateTime,
            event String,
            level String,
            logger String,
            message String,
            
            -- Common fields
            status String,
            error String,
            timestamp DateTime,
            
            -- Extraction fields
            docId String,
            sourceType String,
            ingestionTime String,
            
            -- Query fields
            inputType String,
            inputValue String,
            inputTokens UInt32,
            dependencyId String,
            outputVal String,
            outputTokens UInt32,
            responseTime Float64,
            zeroResponse UInt8,
            outputFiles Array(String),
            
            -- Additional metadata
            metadata String
        ) ENGINE = MergeTree()
        ORDER BY (logtime, event, level)
        {get_ttl_settings()}
        """.format(
            table_name=default_config.table_name,
            ttl_settings=get_ttl_settings()
        )
        
        client.execute(create_table_sql)
        print(f"✅ Created table '{default_config.table_name}' in database '{default_config.database}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
        
    finally:
        if close_client and client:
            client.disconnect()
