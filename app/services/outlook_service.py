"""
Outlook Service with integrated authentication, batch processing, and comprehensive logging.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, AsyncGenerator
import aiohttp
from urllib.parse import urljoin

from .base_service import BaseService
from LoggingPipeline.Logging.logSystem import LogStatus, InputType, AuditLogger
from app.auth.auth_client import auth_client


class OutlookService(BaseService):
    """
    Service class for handling Outlook email operations with integrated authentication,
    batch processing, and comprehensive logging.
    """
    
    BASE_URL = "https://graph.microsoft.com/v1.0"
    BATCH_SIZE = 50  # Number of emails to process in a single batch
    
    def __init__(self, logger=None):
        super().__init__(logger=logger or AuditLogger(platform="outlook"), platform="outlook")
        self._session = None
        self._token = None
        self._token_expiry = None
    
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize the service and create a session."""
        self._session = aiohttp.ClientSession()
        self._token = await auth_client.get_token("outlook")
        self._token_expiry = datetime.utcnow() + timedelta(minutes=55)  # Token expires in 1h
    
    async def close(self):
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _ensure_token_valid(self):
        """Ensure the current token is valid, refreshing if necessary."""
        if not self._token or datetime.utcnow() >= self._token_expiry:
            self._token = await auth_client.get_token("outlook")
            self._token_expiry = datetime.utcnow() + timedelta(minutes=55)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to the Microsoft Graph API."""
        await self._ensure_token_valid()
        
        url = urljoin(self.BASE_URL, endpoint)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            **kwargs.pop('headers', {})
        }
        
        async with self._session.request(method, url, headers=headers, **kwargs) as response:
            response.raise_for_status()
            if response.status == 204:  # No content
                return {}
            return await response.json()
    
    async def process_emails(
        self,
        folder_id: str,
        email_ids: List[str],
        batch_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of emails with comprehensive logging and error handling.
        
        Args:
            folder_id: ID of the folder containing the emails
            email_ids: List of email message IDs to process
            batch_size: Number of emails to process in parallel (default: BATCH_SIZE)
            metadata: Additional metadata to include in logs
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not self._session:
            await self.initialize()
            
        batch_size = batch_size or self.BATCH_SIZE
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
            "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Log the start of batch processing
        await self.log_extraction(
            source_type="email_batch",
            doc_id=results["batch_id"],
            status=LogStatus.PROCESSING,
            metadata={
                "folder_id": folder_id,
                "total_emails": len(email_ids),
                "batch_size": batch_size,
                **({} if metadata is None else metadata)
            }
        )
        
        # Process emails in batches
        for i in range(0, len(email_ids), batch_size):
            batch = email_ids[i:i + batch_size]
            tasks = [self._process_single_email(email_id, folder_id, metadata) for email_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                results["processed"] += 1
                if isinstance(result, Exception):
                    results["failed"] += 1
                    error_msg = str(result)
                    results["errors"].append({
                        "error": error_msg,
                        "type": result.__class__.__name__
                    })
                else:
                    results["succeeded"] += 1
        
        # Log the completion of batch processing
        await self.log_extraction(
            source_type="email_batch",
            doc_id=results["batch_id"],
            status=LogStatus.SUCCESS if results["failed"] == 0 else LogStatus.PARTIAL,
            metadata={
                "folder_id": folder_id,
                "total_processed": results["processed"],
                "succeeded": results["succeeded"],
                "failed": results["failed"],
                **({} if metadata is None else metadata)
            }
        )
        
        return results
    
    async def _process_single_email(
        self,
        email_id: str,
        folder_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a single email with error handling and logging."""
        try:
            # Fetch the email
            email = await self._fetch_email(email_id, folder_id)
            
            # Process the email content
            processed_data = await self._process_email_content(email)
            
            # Log successful processing
            await self.log_extraction(
                source_type="email",
                doc_id=email_id,
                status=LogStatus.SUCCESS,
                metadata={
                    "folder_id": folder_id,
                    "email_id": email_id,
                    "subject": email.get("subject"),
                    "received_date": email.get("receivedDateTime"),
                    **({} if metadata is None else metadata)
                }
            )
            
            return processed_data
            
        except Exception as e:
            error_msg = str(e)
            await self.log_extraction(
                source_type="email",
                doc_id=email_id,
                status=LogStatus.FAILED,
                error=error_msg,
                metadata={
                    "folder_id": folder_id,
                    "email_id": email_id,
                    "error_type": e.__class__.__name__,
                    **({} if metadata is None else metadata)
                }
            )
            raise
    
    async def search_emails(
        self,
        query: str,
        folder_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Search emails using Microsoft Graph API with comprehensive query capabilities.
        
        Args:
            query: Search query string
            folder_id: Optional folder ID to search within
            start_date: Optional start date for filtering (ISO format)
            end_date: Optional end date for filtering (ISO format)
            metadata: Additional metadata for logging
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        query_id = f"outlook_search_{hash(query)}_{datetime.utcnow().timestamp()}"
        
        try:
            # Build search parameters
            search_params = {
                "$search": f"\"{query}\"",
                "$orderby": "receivedDateTime desc",
                "$select": "id,subject,from,receivedDateTime,hasAttachments,importance,isRead",
                "$top": min(limit, 1000)  # Microsoft Graph has a max page size of 1000
            }
            
            # Add date filters if provided
            if start_date or end_date:
                date_filter = []
                if start_date:
                    date_filter.append(f"receivedDateTime ge {start_date}")
                if end_date:
                    date_filter.append(f"receivedDateTime le {end_date}")
                search_params["$filter"] = " and ".join(date_filter)
            
            # Execute search
            if folder_id:
                endpoint = f"/me/mailFolders/{folder_id}/messages"
            else:
                endpoint = "/me/messages"
            
            response = await self._make_request("GET", endpoint, params=search_params)
            results = response.get("value", [])
            
            # Log successful search
            await self.log_query(
                query_id=query_id,
                query_text=query,
                status=LogStatus.SUCCESS,
                metadata={
                    "folder_id": folder_id,
                    "result_count": len(results),
                    "start_date": start_date,
                    "end_date": end_date,
                    **({} if metadata is None else metadata)
                }
            )
            
            return {
                "query_id": query_id,
                "results": results,
                "count": len(results),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = str(e)
            await self.log_query(
                query_id=query_id,
                query_text=query,
                status=LogStatus.FAILED,
                error=error_msg,
                metadata={
                    "folder_id": folder_id,
                    "error_type": e.__class__.__name__,
                    **({} if metadata is None else metadata)
                }
            )
            
            return {
                "query_id": query_id,
                "results": [],
                "count": 0,
                "status": "error",
                "error": error_msg
            }
    
    async def index_emails(
        self,
        email_ids: List[str],
        folder_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Index a batch of emails with comprehensive logging and error handling.
        
        Args:
            email_ids: List of email IDs to index
            folder_id: Optional folder ID for context
            metadata: Additional metadata for logging
            batch_size: Number of emails to process in parallel
            
        Returns:
            Dictionary with indexing results and statistics
        """
        if not self._session:
            await self.initialize()
            
        batch_size = batch_size or self.BATCH_SIZE
        results = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "errors": [],
            "batch_id": f"index_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Log the start of indexing
        await self.log_indexing(
            doc_id=results["batch_id"],
            status=LogStatus.PROCESSING,
            metadata={
                "folder_id": folder_id,
                "total_emails": len(email_ids),
                "batch_size": batch_size,
                **({} if metadata is None else metadata)
            }
        )
        
        # Process emails in batches
        for i in range(0, len(email_ids), batch_size):
            batch = email_ids[i:i + batch_size]
            tasks = [self._index_single_email(email_id, folder_id, metadata) for email_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                results["processed"] += 1
                if isinstance(result, Exception):
                    results["failed"] += 1
                    error_msg = str(result)
                    results["errors"].append({
                        "error": error_msg,
                        "type": result.__class__.__name__
                    })
                else:
                    results["succeeded"] += 1
        
        # Log the completion of indexing
        await self.log_indexing(
            doc_id=results["batch_id"],
            status=LogStatus.SUCCESS if results["failed"] == 0 else LogStatus.PARTIAL,
            metadata={
                "folder_id": folder_id,
                "total_processed": results["processed"],
                "succeeded": results["succeeded"],
                "failed": results["failed"],
                **({} if metadata is None else metadata)
            }
        )
        
        return results
    
    async def _index_single_email(
        self,
        email_id: str,
        folder_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Index a single email with error handling and logging."""
        try:
            # Fetch the email
            email = await self._fetch_email(email_id, folder_id)
            
            # Process the email content
            processed_data = await self._process_email_content(email)
            
            # Here you would typically send the data to your search index
            # For example: await search_client.index_document("emails", processed_data)
            
            # Log successful indexing
            await self.log_indexing(
                doc_id=email_id,
                status=LogStatus.SUCCESS,
                metadata={
                    "folder_id": folder_id,
                    "email_id": email_id,
                    "subject": email.get("subject"),
                    **({} if metadata is None else metadata)
                }
            )
            
            return processed_data
            
        except Exception as e:
            error_msg = str(e)
            await self.log_indexing(
                doc_id=email_id,
                status=LogStatus.FAILED,
                error=error_msg,
                metadata={
                    "folder_id": folder_id,
                    "email_id": email_id,
                    "error_type": e.__class__.__name__,
                    **({} if metadata is None else metadata)
                }
            )
            raise
    
    async def _fetch_email(self, email_id: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch a single email from Microsoft Graph API."""
        if folder_id:
            endpoint = f"/me/mailFolders/{folder_id}/messages/{email_id}"
        else:
            endpoint = f"/me/messages/{email_id}"
            
        # Include common fields and expand attachments if needed
        endpoint += "?$select=id,subject,body,from,toRecipients,ccRecipients,bccRecipients,"
        endpoint += "sentDateTime,receivedDateTime,hasAttachments,importance,isRead,categories"
        
        return await self._make_request("GET", endpoint)
    
    async def _process_email_content(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw email data into a structured format."""
        return {
            "id": email_data.get("id"),
            "subject": email_data.get("subject", "(No subject)"),
            "from": email_data.get("from", {}).get("emailAddress", {}).get("address"),
            "to": [r.get("emailAddress", {}).get("address") for r in email_data.get("toRecipients", [])],
            "cc": [r.get("emailAddress", {}).get("address") for r in email_data.get("ccRecipients", [])],
            "bcc": [r.get("emailAddress", {}).get("address") for r in email_data.get("bccRecipients", [])],
            "sent_date": email_data.get("sentDateTime"),
            "received_date": email_data.get("receivedDateTime"),
            "has_attachments": email_data.get("hasAttachments", False),
            "importance": email_data.get("importance", "normal"),
            "is_read": email_data.get("isRead", False),
            "categories": email_data.get("categories", []),
            "body_preview": email_data.get("body", {}).get("content", "")[:500],
            "processed_at": datetime.utcnow().isoformat()
        }
