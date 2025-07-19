"""Logging configuration for Metanalyst-Agent"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the Metanalyst-Agent system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create main logger
    logger = logging.getLogger("metanalyst_agent")
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set up specific loggers for different components
    _setup_component_loggers(numeric_level)
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}, Console: {enable_console}")
    
    return logger


def _setup_component_loggers(log_level: int):
    """Set up loggers for specific components"""
    
    # Agent loggers
    for agent_name in ["supervisor", "researcher", "processor", "analyst", "writer", "reviewer", "editor"]:
        agent_logger = logging.getLogger(f"metanalyst_agent.agents.{agent_name}")
        agent_logger.setLevel(log_level)
    
    # Tool loggers
    for tool_category in ["research", "processing", "analysis", "handoff"]:
        tool_logger = logging.getLogger(f"metanalyst_agent.tools.{tool_category}")
        tool_logger.setLevel(log_level)
    
    # Graph logger
    graph_logger = logging.getLogger("metanalyst_agent.graph")
    graph_logger.setLevel(log_level)
    
    # State logger
    state_logger = logging.getLogger("metanalyst_agent.state")
    state_logger.setLevel(log_level)
    
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("tavily").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component
    
    Args:
        name: Name of the component/module
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"metanalyst_agent.{name}")


def log_execution_metrics(
    logger: logging.Logger,
    phase: str,
    agent: str,
    metrics: dict,
    level: int = logging.INFO
):
    """
    Log execution metrics in a structured format
    
    Args:
        logger: Logger instance
        phase: Current execution phase
        agent: Current agent name
        metrics: Dictionary of metrics to log
        level: Logging level
    """
    
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    logger.log(level, f"METRICS | Phase: {phase} | Agent: {agent} | {metrics_str}")


def log_agent_handoff(
    logger: logging.Logger,
    from_agent: str,
    to_agent: str,
    reason: str,
    context: str = ""
):
    """
    Log agent handoff events
    
    Args:
        logger: Logger instance
        from_agent: Source agent
        to_agent: Target agent
        reason: Reason for handoff
        context: Additional context
    """
    
    context_str = f" | Context: {context}" if context else ""
    logger.info(f"HANDOFF | {from_agent} → {to_agent} | Reason: {reason}{context_str}")


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: dict,
    phase: str = "unknown",
    agent: str = "unknown"
):
    """
    Log errors with additional context information
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
        phase: Current phase when error occurred
        agent: Current agent when error occurred
    """
    
    context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
    logger.error(
        f"ERROR | Phase: {phase} | Agent: {agent} | Error: {str(error)} | Context: {context_str}",
        exc_info=True
    )