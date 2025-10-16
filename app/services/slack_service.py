"""
Enhanced SlackService with comprehensive ingestion pipeline support.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, AsyncGenerator, Union
import aiohttp
from urllib.parse import urljoin
import json
import logging

from .base_service import BaseService
from LoggingPipeline.Logging.logSystem import LogStatus, AuditLogger
from app.auth.auth_client import auth_client

logger = logging.getLogger(__name__)

class SlackService(BaseService):
    """
    Enhanced Slack service with comprehensive support for ingestion pipeline.
    Handles authentication, rate limiting, and provides high-level methods
    for common operations.
    """
    
    BASE_URL = "https://slack.com/api"
    BATCH_SIZE = 50  # Default batch size for bulk operations
    
    def __init__(self, logger=None):
        super().__init__(logger=logger or AuditLogger(platform="slack"), platform="slack")
        self._session = None
        self._token = None
        self._token_expiry = None
        self._rate_limit_remaining = 1
        self._rate_limit_reset = 0
    
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        """Initialize the service and create a session."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._token = await auth_client.get_token("slack")
            self._token_expiry = datetime.utcnow() + timedelta(minutes=55)
    
    async def close(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _ensure_token_valid(self):
        """Ensure the current token is valid, refreshing if necessary."""
        if not self._token or datetime.utcnow() >= self._token_expiry:
            self._token = await auth_client.get_token("slack")
            self._token_expiry = datetime.utcnow() + timedelta(minutes=55)
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to the Slack API with rate limiting."""
        await self._ensure_token_valid()
        
        # Handle rate limiting
        now = datetime.utcnow().timestamp()
        if self._rate_limit_remaining <= 0 and now < self._rate_limit_reset:
            sleep_time = self._rate_limit_reset - now + 1
            logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            await asyncio.sleep(sleep_time)
        
        url = urljoin(self.BASE_URL, endpoint)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json; charset=utf-8",
            **kwargs.pop('headers', {})
        }
        
        async with self._session.request(method, url, headers=headers, **kwargs) as response:
            # Update rate limit info from headers
            self._rate_limit_remaining = int(response.headers.get('Retry-After', 1))
            self._rate_limit_reset = int(response.headers.get('Retry-After', 0)) + int(time.time())
            
            if response.status == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 5))
                logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                return await self._make_request(method, endpoint, **kwargs)
                
            response.raise_for_status()
            data = await response.json()
            
            if not data.get('ok', False):
                error = data.get('error', 'unknown_error')
                logger.error(f"Slack API error: {error}")
                raise Exception(f"Slack API error: {error}")
                
            return data

    # Core API Methods
    
    async def list_channels(self, types: str = "public_channel,private_channel,im,mpim") -> List[Dict[str, Any]]:
        """List all channels/conversations of specified types."""
        all_channels = []
        cursor = None
        
        while True:
            params = {"types": types, "limit": 200}
            if cursor:
                params["cursor"] = cursor
                
            response = await self._make_request("GET", "/conversations.list", params=params)
            all_channels.extend(response.get("channels", []))
            
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
                
        return all_channels
    
    async def fetch_messages(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: Optional[float] = None,
        latest: Optional[float] = None,
        include_threads: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch messages from a channel with optional date filtering."""
        all_messages = []
        cursor = None
        
        while len(all_messages) < limit:
            params = {
                "channel": channel_id,
                "limit": min(limit - len(all_messages), 200),  # Max 200 per request
            }
            
            if oldest:
                params["oldest"] = str(oldest)
            if latest:
                params["latest"] = str(latest)
            if cursor:
                params["cursor"] = cursor
                
            response = await self._make_request("GET", "/conversations.history", params=params)
            messages = response.get("messages", [])
            
            if not messages:
                break
                
            if include_threads:
                # Process threads for messages that have replies
                for msg in messages:
                    if "thread_ts" in msg and "reply_count" in msg and msg["reply_count"] > 0:
                        msg["replies"] = await self._fetch_thread_replies(
                            channel_id=channel_id,
                            thread_ts=msg["thread_ts"]
                        )
            
            all_messages.extend(messages)
            cursor = response.get("response_metadata", {}).get("next_cursor")
            
            if not cursor or len(all_messages) >= limit:
                break
                
        return all_messages[:limit]
    
    async def _fetch_thread_replies(self, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Fetch all replies in a thread."""
        response = await self._make_request(
            "GET",
            "/conversations.replies",
            params={
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 1000  # Max allowed by Slack
            }
        )
        # Skip the first message as it's the parent
        return response.get("messages", [])[1:] if response.get("messages") else []
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get detailed information about a user."""
        response = await self._make_request("GET", "/users.info", params={"user": user_id})
        return response.get("user", {})
    
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users in the workspace."""
        response = await self._make_request("GET", "/users.list")
        return response.get("members", [])
    
    async def get_channel_info(self, channel_id: str) -> Dict[str, Any]:
        """Get detailed information about a channel."""
        response = await self._make_request(
            "GET",
            "/conversations.info",
            params={"channel": channel_id}
        )
        return response.get("channel", {})
    
    # High-level ingestion methods
    
    async def get_channel_messages(
        self,
        channel_id: str,
        days_back: int = 7,
        include_threads: bool = True
    ) -> List[Dict[str, Any]]:
        """Get messages from a channel from the last N days."""
        oldest = (datetime.utcnow() - timedelta(days=days_back)).timestamp()
        return await self.fetch_messages(
            channel_id=channel_id,
            oldest=oldest,
            include_threads=include_threads
        )
    
    async def get_all_recent_messages(
        self,
        days_back: int = 7,
        channel_types: str = "public_channel,private_channel",
        include_threads: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent messages from all channels of specified types."""
        channels = await self.list_channels(types=channel_types)
        all_messages = {}
        
        for channel in channels:
            try:
                channel_id = channel["id"]
                channel_name = channel.get("name", f"channel_{channel_id[:8]}")
                
                logger.info(f"Fetching messages from channel: {channel_name}")
                messages = await self.get_channel_messages(
                    channel_id=channel_id,
                    days_back=days_back,
                    include_threads=include_threads
                )
                
                if messages:
                    all_messages[channel_name] = messages
                    logger.info(f"Fetched {len(messages)} messages from {channel_name}")
                else:
                    logger.info(f"No messages found in {channel_name}")
                    
            except Exception as e:
                logger.error(f"Error processing channel {channel.get('name', 'unknown')}: {str(e)}")
                continue
                
        return all_messages
    
    # Utility methods for data processing
    
    @staticmethod
    def format_message_for_ingestion(message: Dict[str, Any]) -> Dict[str, Any]:
        """Format a Slack message for the ingestion pipeline."""
        return {
            "id": message.get("ts"),
            "channel": message.get("channel"),
            "user": message.get("user"),
            "text": message.get("text", ""),
            "thread_ts": message.get("thread_ts"),
            "type": message.get("type"),
            "subtype": message.get("subtype"),
            "attachments": message.get("attachments", []),
            "blocks": message.get("blocks", []),
            "reactions": message.get("reactions", []),
            "reply_count": message.get("reply_count", 0),
            "is_starred": message.get("is_starred", False),
            "pinned_to": message.get("pinned_to", []),
            "edited": message.get("edited", {}).get("ts") if message.get("edited") else None,
            "deleted": message.get("deleted", False),
            "processed_at": datetime.utcnow().isoformat(),
            "replies": [
                SlackService.format_message_for_ingestion(reply)
                for reply in message.get("replies", [])
            ]
        }
