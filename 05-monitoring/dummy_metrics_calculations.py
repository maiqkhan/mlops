import datetime 
import time 
import random 
import logging 
import uuid 
import pytz
import pandas as pd
import io
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
    drop table if exists dummy_metrics;
    create table dummy_metrics(
        timestamp timestamp,
        value1 integer,
        value2 varchar,
        value3 float
    );
"""

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


def calculate_dummy_metrics_psql(curr):
    value1 = rand.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = rand.random()

    print('inserting record')
    curr.execute(
        "insert into dummy_metrics(timestamp ,value1, value2, value3) values (%s, %s, %s, %s)",
        (datetime.datetime.now(pytz.timezone('Canada/Eastern')), value1, value2, value3)
    )


def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    with psycopg2.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        for i in range(0, 100):
            with conn.cursor() as curr:
                calculate_dummy_metrics_psql(curr)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")

if __name__ == "__main__":
    main()