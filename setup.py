import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def main():
    """
    Set up project directories
    """
    # Data directories
    data_dirs = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/feedback",
        "data/logs"
    ]
    
    # Create directories
    for directory in data_dirs:
        ensure_dir(directory)
    
    logger.info("Project setup complete")

if __name__ == "__main__":
    main() 