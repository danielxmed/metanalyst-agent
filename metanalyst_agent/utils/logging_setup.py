"""
Logging setup utilities for Metanalyst-Agent

Provides centralized logging configuration for the multi-agent system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    logger_name: str = "metanalyst-agent"
) -> logging.Logger:
    """
    Setup logging configuration for Metanalyst-Agent
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def setup_agent_logger(agent_name: str, base_logger: logging.Logger) -> logging.Logger:
    """
    Setup logger for a specific agent
    
    Args:
        agent_name: Name of the agent
        base_logger: Base logger to inherit configuration from
        
    Returns:
        Agent-specific logger
    """
    
    agent_logger = logging.getLogger(f"{base_logger.name}.{agent_name}")
    agent_logger.setLevel(base_logger.level)
    
    # Copy handlers from base logger
    for handler in base_logger.handlers:
        agent_logger.addHandler(handler)
    
    agent_logger.propagate = False
    
    return agent_logger