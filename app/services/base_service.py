from dataclasses import asdict
from typing import Optional, Dict, Any
from LoggingPipeline.Logging.logSystem import AuditLogger, LogStatus, InputType

class BaseService:
    def __init__(self, logger: AuditLogger, platform: str):
        # Base service class for platform-specific services.
        self.logger = logger
        self.platform = platform

    async def log_extraction(
        self,
        source_type: str,
        doc_id: str,
        status: LogStatus,
        error: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        log_data = {
            "platform": self.platform,
            "sourceType": source_type,
            "docId": doc_id,
            "status": status.value,
            "error": error,
            **({} if metadata is None else metadata)
        }
        return await self.logger.log_extraction(log_data)

    async def log_query(
        self,
        query_id: str,
        query_text: str,
        status: LogStatus,
        error: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        log_data = {
            "platform": self.platform,
            "queryId": query_id,
            "queryText": query_text,
            "status": status.value,
            "error": error,
            **({} if metadata is None else metadata)
        }
        return await self.logger.log_query(log_data)

    async def log_indexing(
        self,
        doc_id: str,
        status: LogStatus,
        error: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        log_data = {
            "platform": self.platform,
            "docId": doc_id,
            "status": status.value,
            "error": error,
            **({} if metadata is None else metadata)
        }
        return await self.logger.log_indexing(log_data)
