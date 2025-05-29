from fastapi import FastAPI
import joblib
import pandas as pd
import json
import uvicorn

with open("lin_reg.joblib", "rb") as f_in:
    lin_reg_pipeline = joblib.load(f_in)


def predict_ride_duration(features) -> float:

    return lin_reg_pipeline.predict(features)[0]


app = FastAPI(title="NYC Taxi Trip Duration Prediction API", version="1.0")


@app.post("/predict")
def output_ride_duration(ride: dict):

    ride_df = pd.DataFrame([ride])

    ride_duration = predict_ride_duration(ride_df)

    result = {
        "duration": float(ride_duration),
    }

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
