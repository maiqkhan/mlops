from dagster import Config, ConfigurableResource, ResourceDependency


class ExtractFileName(Config):
    f_name: str


class scoring_dataset_config(ConfigurableResource):
    input_taxi_type: str
    input_dataset_year: int
    input_dataset_month: int
    model_uri: str
    model_name: str
    output_bucket_name: str
