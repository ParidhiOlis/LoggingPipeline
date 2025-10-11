from typing import Optional, Dict, Any, List, Union
from .base_service import BaseService
from LoggingPipeline.Logging.logSystem import LogStatus

class OutlookService(BaseService):
    """Service class for handling Outlook-related operations with integrated logging."""
    
    def __init__(self, logger):
        super().__init__(logger=logger, platform="outlook")
        self.batch_size = 50  # Number of emails to process in a batch
        
    async def process_emails(
        self,
        folder_id: str,
        email_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        results = {"success": 0, "failed": 0}
        
        for email_id in email_ids:
            try:
                # email processing logic here
                # email = await self._fetch_email(folder_id, email_id)
                # processed_data = await self._process_email_content(email)
                
                # Log successful processing
                await self.log_extraction(
                    source_type="email",
                    doc_id=email_id,
                    status=LogStatus.SUCCESS,
                    metadata={
                        "folder_id": folder_id,
                        "email_id": email_id,
                        **({} if metadata is None else metadata)
                    }
                )
                results["success"] += 1
                
            except Exception as e:
                await self.log_extraction(
                    source_type="email",
                    doc_id=email_id,
                    status=LogStatus.FAILED,
                    error=str(e),
                    metadata={
                        "folder_id": folder_id,
                        "error_type": e.__class__.__name__,
                        **({} if metadata is None else metadata)
                    }
                )
                results["failed"] += 1
                
        return results

    async def search_emails(
        self,
        query: str,
        folder_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        query_id = f"outlook_search_{hash(query)}"
        
        try:
            # search logic here
            # results = await self._execute_outlook_search(query, folder_id, start_date, end_date)
            results = {}
            
            # Log successful query
            await self.log_query(
                query_id=query_id,
                query_text=query,
                status=LogStatus.SUCCESS,
                metadata={
                    "folder_id": folder_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "result_count": len(results.get("value", [])),
                    **({} if metadata is None else metadata)
                }
            )
            
            return results
            
        except Exception as e:
            await self.log_query(
                query_id=query_id,
                query_text=query,
                status=LogStatus.FAILED,
                error=str(e),
                metadata={
                    "folder_id": folder_id,
                    "error_type": e.__class__.__name__,
                    **({} if metadata is None else metadata)
                }
            )
            raise

    async def index_emails(
        self,
        email_ids: List[str],
        folder_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        results = {"success": 0, "failed": 0}
        
        for email_id in email_ids:
            try:
                # indexing logic here
                # await self._index_email(email_id, folder_id)
                
                # Log successful indexing
                await self.log_indexing(
                    doc_id=email_id,
                    status=LogStatus.SUCCESS,
                    metadata={
                        "folder_id": folder_id,
                        **({} if metadata is None else metadata)
                    }
                )
                results["success"] += 1
                
            except Exception as e:
                await self.log_indexing(
                    doc_id=email_id,
                    status=LogStatus.FAILED,
                    error=str(e),
                    metadata={
                        "folder_id": folder_id,
                        "error_type": e.__class__.__name__,
                        **({} if metadata is None else metadata)
                    }
                )
                results["failed"] += 1
                
        return results
