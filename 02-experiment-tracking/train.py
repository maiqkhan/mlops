import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

@click.option(
    "--uri_path",
    default="./output",
    help="URI string of the local tracking server db"
)

def run_train(data_path: str, uri_path: str):

    mlflow.set_tracking_uri(uri_path)
    mlflow.set_experiment("nyc-taxi-homework-experiment")

    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        
        mlflow.set_tag("developer", "Mustafa")

        mlflow.log_param("train-data-path", f"{data_path}/train.pkl")
        mlflow.log_param("validation-data-path", f"{data_path}/val.pkl")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()