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
