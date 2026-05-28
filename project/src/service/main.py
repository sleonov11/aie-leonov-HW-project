from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import time
from collections import defaultdict
from .utils import setup_logger, MODEL_PATH, SERVICE_PORT
from .model import load_model
from contextlib import asynccontextmanager

logger = setup_logger()
request_counter = defaultdict(int)
response_times = []

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model
    model = load_model(MODEL_PATH)
    logger.info("Service started")
    yield
    # Shutdown
    logger.info("Service shutting down")

app = FastAPI(title="Customer Churn Prediction Service", lifespan=lifespan)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    request_counter[request.url.path] += 1
    response_times.append(duration)
    return response

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/metrics")
async def metrics():
    total = sum(request_counter.values())
    avg_time = sum(response_times) / len(response_times) if response_times else 0.0
    return {
        "total_requests": total,
        "requests_per_endpoint": dict(request_counter),
        "avg_response_time_sec": round(avg_time, 4)
    }

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame(request.data)
        probabilities = model.predict_proba(df)[:, 1].tolist()
        predictions = model.predict(df).tolist()
        logger.info(f"Predicted {len(df)} records")
        return {"predictions": predictions, "probabilities": probabilities}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.service.main:app", host="0.0.0.0", port=SERVICE_PORT, reload=True)