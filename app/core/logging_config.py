"""
Centralized Logging Configuration

This module provides a centralized logging setup that:
1. Configures loguru for structured logging
2. Intercepts standard Python logging calls
3. Ensures all modules log to the same files consistently
"""

import os
import sys
from pathlib import Path
from loguru import logger
import logging


class InterceptHandler(logging.Handler):
    """Handler to intercept standard Python logging calls and route them to loguru"""
    
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    """Configure centralized logging for the entire application"""
    
    # Remove default loguru logger to avoid duplicate logs
    logger.remove()
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure console logging (colored and formatted)
    logger.add(
        sink=sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file logging with rotation
    logger.add(
        sink="logs/app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="1 day",
        retention="7 days",
        compression="zip",
        enqueue=True,  # Makes logging thread-safe
        backtrace=True,
        diagnose=True
    )
    
    # Add error-specific logging
    logger.add(
        sink="logs/errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add debug logging (optional, controlled by environment variable)
    debug_enabled = os.environ.get("DEBUG", "false").lower() == "true"
    if debug_enabled:
        logger.add(
            sink="logs/debug_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="3 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
    
    # Intercept standard Python logging calls
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set specific loggers to appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    
    # Log the configuration
    logger.info("📝 Centralized logging configured")
    logger.info(f"📂 Log files will be saved to: {logs_dir.absolute()}")
    logger.info(f"🐛 Debug logging: {'enabled' if debug_enabled else 'disabled'}")


def get_logger(name: str = None):
    """Get a logger instance that works with the centralized logging system"""
    if name:
        return logger.bind(name=name)
    return logger


# Configure logging when this module is imported
setup_logging() 