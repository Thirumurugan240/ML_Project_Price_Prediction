import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs',LOG_FILE)
os.makedirs(log_path,exist_ok=True)

log_file_path = os.path.join(log_path,LOG_FILE)
# Configure logging
logging.basicConfig(
    filename=log_file_path,
    filemode='a',  # Append mode
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("Logging has started")
