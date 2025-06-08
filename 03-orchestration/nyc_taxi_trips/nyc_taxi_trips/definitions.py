from dagster import (
    Definitions,
    load_assets_from_modules,
    load_asset_checks_from_modules,
    EnvVar,
)

from . import assets, resources


all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "scoring_dataset": resources.scoring_dataset_config(
            input_taxi_type="yellow",
            input_dataset_year=2023,
            input_dataset_month=3,
            model_uri=EnvVar("MODEL_URI"),
            model_name="nyc_taxi_trip_duration_model",
            output_bucket_name=EnvVar("OUTPUT_BUCKET_NAME"),
        )
    },
)
