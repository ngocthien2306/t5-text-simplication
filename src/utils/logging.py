import logging
import os
from datetime import datetime
from typing import Optional, Union, Dict

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    config: Optional[Dict] = None
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (default: "INFO")
        log_file: Path to log file (optional)
        config: Configuration dictionary (optional)
    """
    # Convert string level to logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Create formatters and handlers
    format_str = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # If config is provided and has logging settings
    if config and "logging" in config:
        log_config = config["logging"]
        
        # Override level if specified in config
        if "level" in log_config:
            level = getattr(logging, log_config["level"].upper())
            
        # Add file handler if results_dir is specified
        if "results_dir" in log_config:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(log_config["results_dir"], f"run_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "run.log")
            handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    # Set level for specific loggers if needed
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get logger with given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)