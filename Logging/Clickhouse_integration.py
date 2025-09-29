"""
ClickHouse Integration for Audit Logs
Handles batch ingestion of audit logs into ClickHouse for analysis and monitoring
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import clickhouse_connect


@dataclass
class ClickHouseConfig:
    host: str = "localhost"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "audit_logs"
    secure: bool = False


class ClickHouseAuditIngester:
    # Handles ingestion of audit logs into ClickHouse for analysis
    # Creates tables, validates data, and performs batch inserts
    def __init__(self, config: ClickHouseConfig, log_file_manager=None):
        self.config = config
        self.log_file_manager = log_file_manager
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        # Connect to ClickHouse
        self._connect()
        
        # Ensure database and tables exist
        self._setup_database()
    
    def _connect(self):
        try:
            self.client = clickhouse_connect.get_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure
            )
            
            # Test connection
            result = self.client.query("SELECT 1").result_rows
            self.logger.info(f"Connected to ClickHouse at {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to ClickHouse: {e}")
            raise
    
    def _setup_database(self):
        try:
            # Create database
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.config.database}")
            
            # Create extraction logs table
            self.client.command("""
                CREATE TABLE IF NOT EXISTS extraction_logs (
                    log_id String,
                    timestamp DateTime64(3),
                    level String,
                    event String,
                    logtime DateTime64(3),
                    ingestionTime DateTime64(3),
                    sourceType String,
                    docId String,
                    status Enum('SUCCESS' = 1, 'FAIL' = 2),
                    error String,
                    platform String DEFAULT ''
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(logtime)
                ORDER BY (logtime, sourceType, docId)
                SETTINGS index_granularity = 8192
            """)
            
            # Create indexing logs table
            self.client.command("""
                CREATE TABLE IF NOT EXISTS indexing_logs (
                    log_id String,
                    timestamp DateTime64(3),
                    level String,
                    event String,
                    logtime DateTime64(3),
                    ingestionTime DateTime64(3),
                    sourceType String,
                    docId String,
                    chunkId String,
                    chunkSize UInt32,
                    status Enum('SUCCESS' = 1, 'FAIL' = 2),
                    error String,
                    platform String DEFAULT ''
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(logtime)
                ORDER BY (logtime, sourceType, docId, chunkId)
                SETTINGS index_granularity = 8192
            """)
            
            # Create query logs table
            self.client.command("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    log_id String,
                    timestamp DateTime64(3),
                    level String,
                    event String,
                    logtime DateTime64(3),
                    inputId String,
                    inputType Enum('sentence' = 1, 'question' = 2),
                    inputValue String,
                    inputTokens UInt32,
                    dependencyId String,
                    status UInt16,
                    error String,
                    outputVal String,
                    zeroResponse Bool,
                    outputFiles Array(String),
                    outputTokens UInt32,
                    responseTime Float64,
                    platform String DEFAULT ''
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(logtime)
                ORDER BY (logtime, inputId)
                SETTINGS index_granularity = 8192
            """)
            
            self.logger.info("Database and tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
            raise
    
    def _parse_log_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            parsed = {
                "log_id": entry.get("log_id", ""),
                "timestamp": entry.get("timestamp", ""),
                "level": entry.get("level", ""),
                "event": entry.get("event", ""),
                "platform": entry.get("platform", "")
            }
            
            if isinstance(parsed["timestamp"], str):
                parsed["timestamp"] = datetime.fromisoformat(parsed["timestamp"].replace('Z', '+00:00'))
            
            event = parsed["event"]
            
            if "extraction" in event:
                parsed.update({
                    "logtime": self._parse_datetime(entry.get("logtime")),
                    "ingestionTime": self._parse_datetime(entry.get("ingestionTime")),
                    "sourceType": entry.get("sourceType", ""),
                    "docId": entry.get("docId", ""),
                    "status": entry.get("status", "FAIL"),
                    "error": entry.get("error", "")
                })
                return {"table": "extraction_logs", "data": parsed}
                
            elif "indexing" in event:
                parsed.update({
                    "logtime": self._parse_datetime(entry.get("logtime")),
                    "ingestionTime": self._parse_datetime(entry.get("ingestionTime")),
                    "sourceType": entry.get("sourceType", ""),
                    "docId": entry.get("docId", ""),
                    "chunkId": entry.get("chunkId", ""),
                    "chunkSize": entry.get("chunkSize", 0),
                    "status": entry.get("status", "FAIL"),
                    "error": entry.get("error", "")
                })
                return {"table": "indexing_logs", "data": parsed}
                
            elif "query" in event:
                parsed.update({
                    "logtime": self._parse_datetime(entry.get("logtime")),
                    "inputId": entry.get("inputId", ""),
                    "inputType": entry.get("inputType", "question"),
                    "inputValue": entry.get("inputValue", ""),
                    "inputTokens": entry.get("inputTokens", 0),
                    "dependencyId": entry.get("dependencyId", "-1"),
                    "status": entry.get("status", 500),
                    "error": entry.get("error", ""),
                    "outputVal": entry.get("outputVal", ""),
                    "zeroResponse": entry.get("zeroResponse", True),
                    "outputFiles": entry.get("outputFiles", []),
                    "outputTokens": entry.get("outputTokens", 0),
                    "responseTime": entry.get("responseTime", 0.0)
                })
                return {"table": "query_logs", "data": parsed}
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to parse log entry: {e}")
            return None
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        if isinstance(dt_str, datetime):
            return dt_str
        try:
            if dt_str.endswith('Z'):
                dt_str = dt_str[:-1] + '+00:00'
            return datetime.fromisoformat(dt_str)
        except:
            return datetime.now()
    
    def ingest_log_files(self, days_back: int = 1, batch_size: int = 1000) -> Dict[str, Any]:
        # Ingest log files into ClickHouse
        if not self.log_file_manager:
            raise ValueError("LogFileManager not provided during initialization")
        
        stats = {
            "files_processed": 0,
            "total_entries": 0,
            "successful_inserts": 0,
            "failed_inserts": 0,
            "tables_updated": set(),
            "start_time": datetime.now(),
            "errors": []
        }
        
        # Get log files to process
        log_files = self.log_file_manager.list_log_files(days_back=days_back)
        
        # Group entries by table for batch insertion
        table_batches = {
            "extraction_logs": [],
            "indexing_logs": [],
            "query_logs": []
        }
        
        # Process each log file
        for log_file_info in log_files:
            stats["files_processed"] += 1
            self.logger.info(f"Processing log file: {log_file_info.path}")
            
            try:
                for entry in self.log_file_manager.read_log_entries(log_file_info.path):
                    stats["total_entries"] += 1
                    
                    # Parse the log entry
                    parsed_entry = self._parse_log_entry(entry)
                    if parsed_entry:
                        table_name = parsed_entry["table"]
                        table_batches[table_name].append(parsed_entry["data"])
                        
                        # Insert batch when it reaches batch_size
                        if len(table_batches[table_name]) >= batch_size:
                            self._insert_batch(table_name, table_batches[table_name], stats)
                            table_batches[table_name] = []
                    
            except Exception as e:
                error_msg = f"Error processing file {log_file_info.path}: {e}"
                self.logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        # Insert remaining batches
        for table_name, batch_data in table_batches.items():
            if batch_data:
                self._insert_batch(table_name, batch_data, stats)
        
        # Calculate final statistics
        stats["end_time"] = datetime.now()
        stats["duration"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        stats["tables_updated"] = list(stats["tables_updated"])
        
        self.logger.info(f"Ingestion completed: {stats['successful_inserts']}/{stats['total_entries']} records inserted")
        return stats
    
    def _insert_batch(self, table_name: str, batch_data: List[Dict[str, Any]], stats: Dict[str, Any]):
        # Insert a batch of data into the specified table
        if not batch_data:
            return
        
        try:
            # Insert the batch
            self.client.insert(table_name, batch_data)
            
            stats["successful_inserts"] += len(batch_data)
            stats["tables_updated"].add(table_name)
            
            self.logger.debug(f"Inserted {len(batch_data)} records into {table_name}")
            
        except Exception as e:
            stats["failed_inserts"] += len(batch_data)
            error_msg = f"Failed to insert batch into {table_name}: {e}"
            self.logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        tables = ["extraction_logs", "indexing_logs", "query_logs"]
        stats = {}
        
        for table in tables:
            try:
                result = self.client.query(f"SELECT count() FROM {table}")
                row_count = result.result_rows[0][0] if result.result_rows else 0
                
                date_result = self.client.query(f"""
                    SELECT 
                        min(logtime) as min_date,
                        max(logtime) as max_date
                    FROM {table}
                    WHERE logtime IS NOT NULL
                """)
                
                min_date = None
                max_date = None
                if date_result.result_rows:
                    min_date, max_date = date_result.result_rows[0]
                
                # Get error rate for tables with status column
                error_rate = 0.0
                if table in ["extraction_logs", "indexing_logs"]:
                    error_result = self.client.query(f"""
                        SELECT countIf(status = 'FAIL') * 100.0 / count()
                        FROM {table}
                        WHERE logtime >= now() - INTERVAL 1 DAY
                    """)
                    if error_result.result_rows:
                        error_rate = error_result.result_rows[0][0] or 0.0
                
                stats[table] = {
                    "row_count": row_count,
                    "min_date": str(min_date) if min_date else None,
                    "max_date": str(max_date) if max_date else None,
                    "error_rate_24h": error_rate
                }
                
            except Exception as e:
                self.logger.error(f"Error getting stats for {table}: {e}")
                stats[table] = {"error": str(e)}
        
        return stats
    
    def close(self):
        if self.client:
            self.client.close()
            self.logger.info("ClickHouse connection closed")
