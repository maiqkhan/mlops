import requests
from dagster import asset, AssetExecutionContext  # import the `dagster` library
from .resources import ExtractFileName

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow


@asset(name="raw_taxi_trip_data")
def raw_data(context: AssetExecutionContext, config: ExtractFileName) -> pd.DataFrame:
    """
    Getting an individaul month's taxi trip data from configured directory
    """

    df = pd.read_parquet(config.f_name)

    context.log.info(f"Dataframe shape: {df.shape}")  # 3_403_766

    return df


@asset(name="cleansed_taxi_trip_data")
def cleansed_data(
    context: AssetExecutionContext, raw_taxi_trip_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans the taxi trip data
    """

    raw_data = raw_taxi_trip_data.copy()

    context.log.info(f"Raw data shape: {raw_data.columns}")
    # Convert pickup and dropoff datetime columns to datetime
    raw_data["tpep_dropoff_datetime"] = pd.to_datetime(
        raw_data["tpep_dropoff_datetime"]
    )
    raw_taxi_trip_data["tpep_pickup_datetime"] = pd.to_datetime(
        raw_taxi_trip_data["tpep_pickup_datetime"]
    )

    # Calculate trip duration
    raw_data["duration"] = (
        raw_data["tpep_dropoff_datetime"] - raw_data["tpep_pickup_datetime"]
    )
    raw_data["duration"] = raw_data["duration"].dt.total_seconds() / 60

    context.log.info(
        f"Dataframe shape before duration within hour filter: {raw_data.shape}"
    )

    # Filter out durations less than 1 minute and greater than 60 minutes
    clean_data = raw_data.query("duration >= 1 and duration <= 60")

    context.log.info(
        f"Dataframe shape after duration within hour filter: {clean_data.shape}"  # 3_316_216
    )

    # Convert LocationID to string for OneHotEncoding later on
    categorical_cols = ["PULocationID", "DOLocationID"]
    for col in categorical_cols:
        clean_data[col] = clean_data[col].astype(str)

    return clean_data


@asset(name="taxi_trip_lin_reg_model")
def train_lin_reg_model(
    context: AssetExecutionContext, cleansed_taxi_trip_data: pd.DataFrame
) -> Pipeline:
    """
    Train a linear regression model on the cleansed taxi trip data
    """

    enc = OneHotEncoder(drop="first", handle_unknown="ignore")

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "cat",
                            OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ["PULocationID", "DOLocationID"],
                        ),
                        ("num", StandardScaler(), ["trip_distance"]),
                    ],
                    remainder="passthrough",
                ),
            ),
            ("regressor", LinearRegression()),
        ]
    )

    # Define features and target variable
    X = cleansed_taxi_trip_data[["PULocationID", "DOLocationID", "trip_distance"]]

    y = cleansed_taxi_trip_data["duration"]

    # split the arrays into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    context.log.info(
        f"Model training complete. Model Intercept: {pipeline.named_steps['regressor'].intercept_}"
    )

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate RMSE
    rmse = root_mean_squared_error(y_test, y_pred)
    context.log.info(f"RMSE: {rmse}")

    return pipeline


@asset(name="taxi_trip_registered_model")
def register_lin_reg_model(
    context: AssetExecutionContext, taxi_trip_lin_reg_model: Pipeline
) -> None:

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=train_lin_reg_model,
            artifact_path="model",
            registered_model_name="nyc_taxi_trip_duration_model",
        )
