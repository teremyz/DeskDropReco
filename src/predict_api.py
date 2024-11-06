"""
This FastAPI application serves as a recommendation prediction API for Deskdrop, an internal
communications platform. It leverages both an XGBoost-based model and a popularity-based model
to provide users with relevant article recommendations based on interaction data.

Classes and functions:
- predict: API endpoint function for generating recommendations based on user input.
"""

import logging
import pickle
import pandas as pd

from fastapi import FastAPI

logging.basicConfig(filename="prediction_web_sevice.log", level=logging.INFO)


app = FastAPI()
logging.info("Endpoint is up")
model = pickle.load(open("model/xgb_model.pkl", "rb"))
popularity_model = pickle.load(open("model/popularity_model.pkl", "rb"))
logging.info("Models in memory")
articles = pd.read_parquet("feature_stores/articles.parquet")
logging.info("articles in memory")
unique_train_content_ids = pickle.load(open("model/unique_train_content_ids.pkl", "rb"))
unique_train_preson_ids = pickle.load(open("model/unique_train_preson_ids.pkl", "rb"))
logging.info("IDs in memory")


@app.get("/predict")
def predict(
    dateTime: str,
    eventType: str,
    contentId: int,
    personId: int,
    userRegion: str,
    userCountry: str,
    lastContentId: str,
    lastEventType: str,
    top_n: int,
):
    """
    Generates article recommendations for a user based on their interaction data and profile.

    Args:
        dateTime (str): The timestamp of the interaction in the format "YYYY-MM-DD HH:MM:SS".
        eventType (str): Type of user interaction (e.g., VIEW, LIKE).
        contentId (int): The unique identifier for the content item.
        personId (int): The unique identifier for the user.
        userRegion (str): The region code for the user (e.g., "SP").
        userCountry (str): The country code for the user (e.g., "BR").
        lastContentId (str): The unique identifier for the last content the user interacted with.
        lastEventType (str): The event type for the user's last interaction.
        top_n (int): Number of top recommendations to return.

    Returns:
        dict: A dictionary containing the recommended content items and their associated scores.

    Example json: {
        "dateTime":"2017-02-17 16:12:27", "eventType":"VIEW", "contentId":-5781461435447152359,
        "personId":-9223121837663643404,"userRegion":"SP","userCountry":"BR",
        "lastContentId":"-6728844082024523776","lastEventType":"VIEW", "top_n": 10
    }
    """

    logging.info("Getting the json...")
    inputs = pd.DataFrame(
        {
            "dateTime": dateTime,
            "eventType": eventType,
            "contentId": contentId,
            "personId": personId,
            "userRegion": userRegion,
            "userCountry": userCountry,
            "lastContentId": lastContentId,
            "lastEventType": lastEventType,
        },
        index=[0],
    )

    logging.info("Endpoint is predicting...")
    if contentId in unique_train_content_ids and personId in unique_train_preson_ids:
        logging.info("Xgb Model is predicting...")
        predictions = model.recommend_items(df=inputs, articles=articles, top_n=top_n)

    else:
        logging.info("Popularity Model is predicting...")
        predictions = dict(
            content=popularity_model.recommend_items(topn=top_n), preds=[1] * top_n
        )

    logging.info("Recommendations are ready!")

    return predictions
