"""
Centralized authentication client for all services.
Handles token management, refresh, and storage for various providers.
"""
import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import msal
from msal import ConfidentialClientApplication
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from LoggingPipeline.Logging.logSystem import AuditLogger

class AuthClient:
    """Centralized authentication client for handling OAuth and API tokens."""
    
    def __init__(self, logger: Optional[AuditLogger] = None):
        """Initialize the auth client with optional logger."""
        self.logger = logger or AuditLogger(platform="auth")
        self._tokens: Dict[str, Dict[str, Any]] = {}
        self._init_providers()
    
    def _init_providers(self) -> None:
        """Initialize provider-specific configurations."""
        self.providers = {
            "outlook": {
                "client_id": os.getenv("OUTLOOK_CLIENT_ID"),
                "authority": f"https://login.microsoftonline.com/{os.getenv('OUTLOOK_TENANT_ID')}",
                "scope": ["https://graph.microsoft.com/.default"],
                "secret": os.getenv("OUTLOOK_CLIENT_SECRET"),
            },
            "slack": {
                "token": os.getenv("SLACK_BOT_TOKEN"),
            }
        }
    
    async def get_token(self, provider: str) -> str:

        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Check if we have a valid cached token
        token_info = self._tokens.get(provider, {})
        if token_info and self._is_token_valid(token_info):
            return token_info["access_token"]
        
        # Get a new token
        try:
            if provider == "outlook":
                token = await self._get_outlook_token()
            elif provider == "slack":
                token = await self._get_slack_token()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache the token
            self._tokens[provider] = {
                "access_token": token,
                "expires_at": datetime.utcnow() + timedelta(minutes=50)  # Default 1h expiry with 10min buffer
            }
            
            return token
            
        except Exception as e:
            await self.logger.log_query({
                "status": "FAILED",
                "error": f"Failed to get {provider} token: {str(e)}",
                "platform": provider
            })
            raise
    
    async def _get_outlook_token(self) -> str:
        """Get a token for Microsoft Graph API."""
        config = self.providers["outlook"]
        
        app = ConfidentialClientApplication(
            client_id=config["client_id"],
            authority=config["authority"],
            client_credential=config["secret"]
        )
        
        result = app.acquire_token_silent(
            scopes=config["scope"],
            account=None
        )
        
        if not result:
            result = app.acquire_token_for_client(
                scopes=config["scope"]
            )
        
        if "access_token" not in result:
            raise ValueError(f"Failed to get token: {result.get('error_description')}")
            
        return result["access_token"]
    
    async def _get_slack_token(self) -> str:
        """Get token for Slack API."""
        return self.providers["slack"]["token"]
    
    def _is_token_valid(self, token_info: Dict[str, Any]) -> bool:
        """Check if a token is still valid."""
        expires_at = token_info.get("expires_at")
        if not expires_at:
            return False
        return datetime.utcnow() < expires_at

# Global instance for easy access
auth_client = AuthClient()
