import json
import os

import requests
from dagster import asset # import the `dagster` library

@asset # add the asset decorator to tell Dagster this is an asset
def topstory_ids() -> None:
    """
    Getting the story ids for the top 100 news stories
    """
    
    newstories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_new_story_ids = requests.get(newstories_url).json()[:100]

    os.makedirs("data", exist_ok=True)
    with open("data/topstory_ids.json", "w") as f:
        json.dump(top_new_story_ids, f)