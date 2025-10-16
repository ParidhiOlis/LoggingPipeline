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
        """Initialize the service factory with a logger.
        
        Args:
            logger: Instance of AuditLogger to be used by all services
        """
        cls._logger = logger
        
        # Initialize all services
        cls._services = {
            'slack': SlackService(logger=logger),
            'outlook': OutlookService(logger=logger),
            'sharepoint': SharePointService(logger=logger)
        }
    
    @classmethod
    def get_service(cls, service_name: str) -> BaseService:
        """Get a service instance by name.
        
        Args:
            service_name: Name of the service to retrieve (e.g., 'slack', 'outlook')
            
        Returns:
            The requested service instance
            
        Raises:
            ValueError: If the service is not found
        """
        if service_name.lower() not in cls._services:
            raise ValueError(f"Service '{service_name}' not found. Available services: {list(cls._services.keys())}")
        return cls._services[service_name.lower()]
    
    @classmethod
    def get_all_services(cls) -> Dict[str, BaseService]:
        """Get all registered services.
        
        Returns:
            Dictionary of all registered services
        """
        return cls._services
    
    @classmethod
    def register_service(cls, name: str, service: BaseService):
        """Register a new service.
        
        Args:
            name: Name to register the service under
            service: Service instance to register
        """
        cls._services[name.lower()] = service
        
    @classmethod
    def get_logger(cls) -> AuditLogger:
        """Get the logger instance used by the factory.
        
        Returns:
            The AuditLogger instance
            
        Raises:
            RuntimeError: If the factory hasn't been initialized
        """
        if cls._logger is None:
            raise RuntimeError("ServiceFactory has not been initialized. Call initialize() first.")
        return cls._logger

# Helper function for FastAPI dependency injection
def get_service(service_name: str):
    """Dependency function for FastAPI to inject services.
    
    Args:
        service_name: Name of the service to inject
        
    Returns:
        A function that returns the requested service
    """
    def _get_service():
        return ServiceFactory.get_service(service_name)
    return _get_service
