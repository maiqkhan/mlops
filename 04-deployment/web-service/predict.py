from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
import uvicorn
from dotenv import load_dotenv
import mlflow
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


# Dictionary to store ML models
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.
    Code before yield runs on startup, code after yield runs on shutdown.
    """
    # Startup: Load the model
    try:
        model_uri = ""
        logger.info(f"Loading model from {model_uri}")
        ml_models["taxi_duration"] = mlflow.sklearn.load_model(model_uri=model_uri)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # You can choose to raise the exception to prevent app startup
        # or continue without the model (handle gracefully in endpoints)
        raise e

    yield  # App is running

    # Shutdown: Clean up resources
    logger.info("Shutting down and cleaning up resources")
    ml_models.clear()
    logger.info("Model cleanup completed")


def predict_ride_duration(features) -> float:
    """Predict ride duration using the loaded model"""
    if "taxi_duration" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        model = ml_models["taxi_duration"]
        prediction = model.predict(features)[0]
        return float(prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed") from e


app = FastAPI(
    title="NYC Taxi Trip Duration Prediction API", version="1.0", lifespan=lifespan
)


@app.post("/predict")
def output_ride_duration(ride: dict):
    """Predict ride duration endpoint"""

    try:
        ride_df = pd.DataFrame([ride])
        logger.info(f"Received ride data: {ride_df}")

        logger.info(f"Processing prediction for: {ride}")

        # Make prediction
        ride_duration = predict_ride_duration(ride_df)

        result = {
            "duration": ride_duration,
        }

        logger.info(f"Prediction result: {result}")
        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
