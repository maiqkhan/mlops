import pandas as pd
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from mlflow.sklearn import load_model
import logging
import argparse
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_input_data(taxi_type: str, month: int, year: int) -> pd.DataFrame:
    """
    Returns the input file path for the given taxi type, month, and year.
    """
    input_data = pd.read_parquet(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    )

    raw_data = input_data.copy()
    logger.info(
        f"Loaded {len(raw_data)} rows of {taxi_type} data for {year}-{month:02d}"
    )

    raw_data["tpep_dropoff_datetime"] = pd.to_datetime(
        raw_data["tpep_dropoff_datetime"]
    )
    raw_data["tpep_pickup_datetime"] = pd.to_datetime(raw_data["tpep_pickup_datetime"])

    # Calculate trip duration
    raw_data["duration"] = (
        raw_data["tpep_dropoff_datetime"] - raw_data["tpep_pickup_datetime"]
    )
    raw_data["duration"] = raw_data["duration"].dt.total_seconds() / 60

    # Filter out durations less than 1 minute and greater than 60 minutes
    clean_data = raw_data.query("duration >= 1 and duration <= 60").copy()

    categorical_cols = ["PULocationID", "DOLocationID"]
    for col in categorical_cols:
        clean_data[col] = clean_data[col].astype(str)

    return clean_data


def get_prediction_model(s3_uri: str) -> Pipeline:
    """
    Returns the prediction model loaded from the specified S3 URI.
    """
    logger.info(f"Loading model from {s3_uri}")
    duration_model = load_model(model_uri=s3_uri)
    logger.info("Model loaded successfully")
    return duration_model


def apply_model(duration_model: Pipeline, clean_df: pd.DataFrame) -> np.ndarray:
    """
    Applies the model to the DataFrame and returns predictions.
    """
    logger.info("Applying model to DataFrame")
    return duration_model.predict(clean_df)


def score_model(taxi_type: str, month: int, year: int, s3_uri: str) -> float:
    """
    Scores the model by calculating the RMSE on the input data.
    """
    clean_df = generate_input_data(taxi_type, month, year)
    duration_model = get_prediction_model(s3_uri)

    # Apply the model to get predictions
    predictions = apply_model(duration_model, clean_df)

    # Calculate RMSE
    rmse = root_mean_squared_error(clean_df["duration"], predictions)
    logger.info(f"RMSE for {taxi_type} data in {year}-{month:02d}: {rmse}")

    print(predictions.mean())


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Score the taxi duration prediction model."
    )

    parser.add_argument(
        "--taxi_type",
        type=str,
        required=True,
        help="Type of taxi data to score (e.g., yellow, green)",
    )

    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month of the data to score (1-12)",
    )

    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of the data to score (e.g., 2021)",
    )

    parser.add_argument(
        "--s3_uri",
        type=str,
        required=True,
        help="S3 URI of the model to load (e.g., s3://mlops-zoomcamp/models/taxi_duration_model)",
    )

    args = parser.parse_args()

    score_model(
        taxi_type=args.taxi_type,
        month=args.month,
        year=args.year,
        s3_uri=args.s3_uri,
    )
