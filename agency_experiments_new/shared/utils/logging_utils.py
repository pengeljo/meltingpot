"""
Logging utilities for agency experiments.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = 'INFO', 
                 format_string: Optional[str] = None,
                 include_timestamp: bool = True) -> logging.Logger:
    """
    Set up logging for agency experiments.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
    
    Returns:
        Configured logger
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            format_string = '%(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Return logger
    logger = logging.getLogger('agency_experiments')
    return logger