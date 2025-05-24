from setuptools import find_packages, setup

setup(
    name="nyc_taxi_trips",
    install_requires=["dagster", "dagster-cloud"],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
