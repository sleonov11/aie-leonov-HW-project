import os
import logging

MODEL_PATH = os.getenv('MODEL_PATH', 'artifacts/model_pipeline.pkl')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '8000'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def setup_logger():
    logger = logging.getLogger('churn_service')
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger