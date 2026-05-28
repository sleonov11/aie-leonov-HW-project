import joblib
import logging

logger = logging.getLogger('churn_service')

def load_model(path: str):
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise