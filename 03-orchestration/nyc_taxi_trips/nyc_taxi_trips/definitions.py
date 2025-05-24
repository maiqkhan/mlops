from dagster import (
    Definitions,
    load_assets_from_modules,
    load_asset_checks_from_modules,
    EnvVar,
)

from . import assets


all_assets = load_assets_from_modules([assets])

defs = Definitions(assets=all_assets)
