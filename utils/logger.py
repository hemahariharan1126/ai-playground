import logging
import sys
import os

def setup_logger(name="ai_playground", level=logging.INFO):
    """Sets up a standard logger with a clean format."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if logger already has handlers to avoid duplicate logs
    if not logger.handlers:
        # Create console handler and set level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to ch
        ch.setFormatter(formatter)
        
        # Add ch to logger
        logger.addHandler(ch)
        
    return logger

# Default logger instance
logger = setup_logger()

if __name__ == "__main__":
    logger.info("Logger initialized successfully!")
