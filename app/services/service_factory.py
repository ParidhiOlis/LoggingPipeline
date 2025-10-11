"""
Service factory for creating and managing service instances with dependency injection.
"""
from typing import Dict, Type, Any, Optional
from LoggingPipeline.Logging.logSystem import AuditLogger
from .base_service import BaseService
from .slack_service import SlackService
from .outlook_service import OutlookService
from .sharepoint_service import SharePointService

class ServiceFactory:
    """Factory class for creating and managing service instances."""
    
    _instance = None
    _services: Dict[str, BaseService] = {}
    _logger: Optional[AuditLogger] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceFactory, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, logger: AuditLogger):
        cls._logger = logger
        
        # Initialize all services
        cls._services = {
            'slack': SlackService(logger=logger),
            'outlook': OutlookService(logger=logger),
            'sharepoint': SharePointService(logger=logger)
        }
    
    @classmethod
    def get_service(cls, service_name: str) -> BaseService:
        if service_name.lower() not in cls._services:
            raise ValueError(f"Service '{service_name}' not found. Available services: {list(cls._services.keys())}")
        return cls._services[service_name.lower()]
    
    @classmethod
    def get_all_services(cls) -> Dict[str, BaseService]:
        return cls._services
    
    @classmethod
    def register_service(cls, name: str, service: BaseService):
        cls._services[name.lower()] = service
        
    @classmethod
    def get_logger(cls) -> AuditLogger:
        if cls._logger is None:
            raise RuntimeError("ServiceFactory has not been initialized. Call initialize() first.")
        return cls._logger

# Helper function for FastAPI dependency injection
def get_service(service_name: str):
    def _get_service():
        return ServiceFactory.get_service(service_name)
    return _get_service
