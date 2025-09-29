"""
Audit Logging System
Three stages: Extraction, Indexing, and User Queries
"""

import structlog
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sys


class LogStatus(Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class InputType(Enum):
    SENTENCE = "sentence"
    QUESTION = "question"


@dataclass
class ExtractionLog:
    logtime: str
    ingestionTime: str
    sourceType: str
    docId: str
    status: str
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexingLog:
    logtime: str
    ingestionTime: str
    sourceType: str
    docId: str
    chunkId: str
    chunkSize: int
    status: str
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryLog:
    logtime: str
    inputId: str
    inputType: str
    inputValue: str
    inputTokens: int
    dependencyId: str
    status: int
    error: str
    outputVal: str
    zeroResponse: bool
    outputFiles: List[str]
    outputTokens: int
    responseTime: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AuditLogger:
    def __init__(self, base_log_dir: str = "pipeline_logs", log_level: str = "INFO", platform: str = None):
        self.base_log_dir = Path(base_log_dir)
        self.log_level = log_level
        self.platform = platform
        
        # log directories
        self._setup_directories()
        
        # Configure structlog for pipeline operations
        self._configure_structlog()
        
        # Create stage-specific loggers with platform context
        logger_name_suffix = f"_{platform}" if platform else ""
        self.extraction_logger = structlog.get_logger(f"extraction{logger_name_suffix}")
        self.indexing_logger = structlog.get_logger(f"indexing{logger_name_suffix}")
        self.query_logger = structlog.get_logger(f"queries{logger_name_suffix}")
    
    def _setup_directories(self):
        if self.platform:
            platform_dir = self.base_log_dir / self.platform
            for stage in ["extraction", "indexing", "queries"]:
                (platform_dir / stage).mkdir(parents=True, exist_ok=True)
        else:
            for directory in ["extraction", "indexing", "queries"]:
                (self.base_log_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _configure_structlog(self):
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, self.log_level.upper())
        )
        
        # Structlog configuration
        structlog.configure(
            processors=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_logger_name,
                structlog.processors.JSONRenderer(sort_keys=True)
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _get_current_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def _generate_unique_id(self) -> str:
        return str(uuid.uuid4())
    
    def log_extraction(
        self,
        ingestion_time: str,
        source_type: str,
        doc_id: str,
        status: LogStatus,
        error: str = ""
    ) -> str:
        log_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        extraction_log = ExtractionLog(
            logtime=current_time,
            ingestionTime=ingestion_time,
            sourceType=source_type,
            docId=doc_id,
            status=status.value,
            error=error
        )
        
        log_data = {
            "log_id": log_id,
            **extraction_log.to_dict()
        }
        
        if self.platform:
            log_data["platform"] = self.platform
        
        self.extraction_logger.info(
            "extraction_event",
            **log_data
        )
        
        return log_id
    
    def log_indexing(
        self,
        ingestion_time: str,
        source_type: str,
        doc_id: str,
        chunk_id: str,
        chunk_size: int,
        status: LogStatus,
        error: str = ""
    ) -> str:
        log_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        indexing_log = IndexingLog(
            logtime=current_time,
            ingestionTime=ingestion_time,
            sourceType=source_type,
            docId=doc_id,
            chunkId=chunk_id,
            chunkSize=chunk_size,
            status=status.value,
            error=error
        )
        
        log_data = {
            "log_id": log_id,
            **indexing_log.to_dict()
        }
        
        if self.platform:
            log_data["platform"] = self.platform
        
        self.indexing_logger.info(
            "indexing_event",
            **log_data
        )
        
        return log_id
    
    def log_query(
        self,
        input_type: InputType,
        input_value: str,
        input_tokens: int,
        dependency_id: str,
        status: int,
        error: str,
        output_val: str,
        zero_response: bool,
        output_files: List[str],
        output_tokens: int,
        response_time: float
    ) -> str:
        log_id = self._generate_unique_id()
        current_time = self._get_current_timestamp()
        
        query_log = QueryLog(
            logtime=current_time,
            inputId=log_id,
            inputType=input_type.value,
            inputValue=input_value,
            inputTokens=input_tokens,
            dependencyId=dependency_id,
            status=status,
            error=error,
            outputVal=output_val,
            zeroResponse=zero_response,
            outputFiles=output_files,
            outputTokens=output_tokens,
            responseTime=response_time
        )
        
        log_data = {
            "log_id": log_id,
            **query_log.to_dict()
        }
        
        if self.platform:
            log_data["platform"] = self.platform
        
        self.query_logger.info(
            "query_event",
            **log_data
        )
        
        return log_id