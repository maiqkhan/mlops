import datetime 
import time 
import random 
import logging 
import uuid 
import pytz
import pandas as pd
import io
import psycopg2
import joblib

from evidently import Dataset, DataDefinition, Report, Regression
from evidently.metrics import ValueDrift, DatasetMissingValueCount
from evidently.presets import DataDriftPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
    drop table if exists dummy_metrics;
    create table dummy_metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_value float
    );
"""

reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin', 'rb') as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')

num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

data_definition = DataDefinition(
    numerical_columns=num_features + ["prediction"],
    categorical_columns=cat_features,
    regression=[Regression(target=None, prediction="prediction")],
    
    )


report = Report(
    metrics=[
        ValueDrift(column="prediction"),
        DataDriftPreset(),
        DatasetMissingValueCount()
    ]
)

begin = datetime.datetime(2022, 2, 1, 0, 0)




def prep_db():
    # with psycopg2.connect("host=localhost port=5432 user=postgres password=example") as conn:
    #     res = conn.execute("SELECT 1 FROM pg_database where datname='test'")
    #     if len(res.fetchall()) == 0:
    #         conn.execute("create database test;")
    #     with psycopg2.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
    #         conn.execute(create_table_statement)
    
    
    conn = psycopg2.connect("host=localhost port=5432 user=postgres password=example")
    conn.autocommit = True  # Required for CREATE DATABASE
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            if len(cur.fetchall()) == 0:
                cur.execute("CREATE DATABASE test")
    finally:
        conn.close()

    # Second connection to the test database for table creation
    with psycopg2.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_statement)


def calculate_metrics_psql(curr, i):
    current_data = raw_data[(raw_data.lpep_pickup_datetime <= (begin + datetime.timedelta(i))) & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

    current_data = current_data.fillna(0).copy()
    current_data['prediction'] = model.predict(current_data[num_features + cat_features])

    report_reference_data = Dataset.from_pandas(reference_data, data_definition=data_definition)
    report_current_data = Dataset.from_pandas(current_data, data_definition=data_definition)

    taxi_report = report.run(reference_data= report_reference_data, current_data = report_current_data)

    taxi_report_dict = taxi_report.dict()
    

    prediction_drift = taxi_report_dict['metrics'][0]['value']
    num_drifted_columns = taxi_report_dict['metrics'][1]['value']['count']
    share_of_missing_vals = taxi_report_dict['metrics'][-1]['value']['share']


    print('inserting record')
    curr.execute(
        "insert into dummy_metrics(timestamp ,prediction_drift, num_drifted_columns, share_missing_value) values (%s, %s, %s, %s)",
        (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_of_missing_vals)
    )


def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    with psycopg2.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        for i in range(0, 27):
            with conn.cursor() as curr:
                calculate_metrics_psql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")

if __name__ == "__main__":
    main()


