"""Utility functions for handling and sanitizing error messages to remove sensitive information"""
import re
from typing import Optional

def sanitize_error(error_msg: Optional[str]) -> str:
    if not error_msg:
        return ""
        
    # Remove any potential sensitive data patterns
    patterns = [
        (r'password=[^&\s]+', 'password=***'),
        (r'token=[^&\s]+', 'token=***'),
        (r'api[_-]?key=[^&\s]+', 'api_key=***'),
        (r'bearer\s+[^\s]+', 'bearer ***'),
        (r'basic\s+[^\s]+', 'basic ***'),
        (r'\b\d{4}[-\.\s]?\d{4}[-\.\s]?\d{4}[-\.\s]?\d{4}\b', '****-****-****-****'),  # Credit card
        (r'\b\d{3}-?\d{2}-?\d{4}\b', '***-**-****'),  # SSN
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '***@***.***'),  # Email
    ]
    
    sanitized = str(error_msg)
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Truncate to prevent log injection and excessive logging
    return sanitized[:1000]  # Limit error message length
