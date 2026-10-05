import logging
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="RAG Memory"):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  

    logger.setLevel(logging.DEBUG)

    #  file handler
    log_file = os.path.join(LOG_DIR, f"{datetime.now().date()}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    #  console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(
        logging.WARNING if os.getenv("DAENA_CLI") == "1" else logging.DEBUG
    )
    #  format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] → %(message)s",
        datefmt="%H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger