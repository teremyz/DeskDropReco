import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.core.evaluation import calculate_metrics


class ModelTrainingPipeline:
    def __init__(
        self, data_loader, saver, model, top_n, split_date, event_type_strength
    ):
        """
        Initializes the ModelTrainingPipeline with data loader, saver, model, and configuration parameters.

        Args:
            data_loader: An instance of CsvDataLoader to load interactions and articles data.
            saver: An instance of PickleSaver to save the trained model and other artifacts.
            model: The recommendation model to be trained.
            top_n (int): The number of top recommendations to consider.
            split_date (str): The date to split data into training and testing sets.
            event_type_strength (dict): A mapping of event types to their respective strength scores.
        """
        self.data_loader = data_loader
        self.saver = saver
        self.model = model
        self.top_n = top_n
        self.split_date = split_date
        self.event_type_strength = event_type_strength

    def training_preprocess(self, interactions_df, event_type_strength):
        """
        Processes interactions data by applying event strength and log transformation,
        then splits the data into training and testing sets.

        Args:
            interactions_df (pd.DataFrame): DataFrame containing user interactions with articles.
            event_type_strength (dict): A mapping of event types to their respective strength scores.

        Returns:
            tuple: A tuple containing the processed training and testing DataFrames.
        """
        interactions_df["eventStrength"] = interactions_df["eventType"].apply(
            lambda x: event_type_strength[x]
        )

        interactions_train, interactions_test = self.train_test_split(
            interactions_df, self.split_date
        )

        interactions_train = (
            interactions_train.groupby(["personId", "contentId"])["eventStrength"]
            .sum()
            .apply(lambda x: np.log(1 + x))
            .reset_index()
        )
        interactions_test = (
            interactions_test.groupby(["personId", "contentId"])["eventStrength"]
            .sum()
            .apply(lambda x: np.log(1 + x))
            .reset_index()
        )

        return interactions_train, interactions_test

    def create_labels(self, test):
        """
        Generates labels for the testing data, identifying the content each user interacted with.

        Args:
            test (pd.DataFrame): The testing DataFrame of user interactions.

        Returns:
            pd.DataFrame: A DataFrame with 'personId' and a list of 'contentId' that each user interacted with.
        """
        return (
            test.copy()
            .reset_index()[["personId", "contentId"]]
            .groupby("personId")["contentId"]
            .agg(list)
            .reset_index()
        )

    def train_test_split(self, interactions_df, split_date):
        """
        Splits interactions data into training and testing sets based on a specified date.

        Args:
            interactions_df (pd.DataFrame): DataFrame containing all user interactions.
            split_date (str): The date to split data into training and testing sets.

        Returns:
            tuple: A tuple containing the training and testing DataFrames.
        """
        interactions_df["dateTime"] = pd.to_datetime(
            interactions_df["timestamp"], unit="s"
        )
        interactions_train = interactions_df[interactions_df["dateTime"] < split_date]
        interactions_test = interactions_df[interactions_df["dateTime"] >= split_date]
        return interactions_train, interactions_test


class PopularityModelTrainingPipeline(ModelTrainingPipeline):
    def run(self):
        """
        Executes the training pipeline for the popularity model. Loads interaction data, preprocesses it, trains the model,
        and evaluates model predictions.

        Args:
            None

        Returns:
            None
        """
        interactions_df = self.data_loader.get_interactions()

        interactions_train, interactions_test = self.training_preprocess(
            interactions_df, self.event_type_strength
        )

        interactions_labels = self.create_labels(test=interactions_test)

        self.model.train(interactions_train)

        interactions_labels["preds"] = interactions_labels.apply(
            lambda x: self.model.recommend_items(topn=self.top_n), axis=1
        )

        print(
            calculate_metrics(
                prediction_col="preds", interactions_labels=interactions_labels
            )
        )

        self.saver.save(self.model, file_name="popularity_model.pkl")


class MatrixFactorizationTrainingPipeline(ModelTrainingPipeline):
    """
    Encodes user and content IDs to a numerical format using label encoding, saving encoders for future use.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing interaction data with 'personId' and 'contentId' columns.
        saver (object): Instance of a saver class responsible for persisting label encoders.

    Returns:
        pd.DataFrame: DataFrame with encoded 'personId' and 'contentId' columns.
    """

    def label_encode_ids(self, interactions_df, saver):
        person_le = LabelEncoder()
        interactions_df["personId"] = person_le.fit_transform(
            interactions_df["personId"]
        )
        saver.save(person_le, file_name="personId_label_encoder.pkl")

        content_le = LabelEncoder()
        interactions_df["contentId"] = content_le.fit_transform(
            interactions_df["contentId"]
        )
        saver.save(content_le, file_name="contentId_label_encoder.pkl")

        return interactions_df

    def run(self):
        """
        Executes the matrix factorization training pipeline. This includes data preprocessing,
        encoding IDs, training the model, generating recommendations, evaluating predictions,
        and saving the trained model.

        Args:
            None

        Returns:
            None
        """
        interactions_df = self.data_loader.get_interactions()

        self.label_encode_ids(interactions_df=interactions_df, saver=self.saver)

        interactions_train, interactions_test = self.training_preprocess(
            interactions_df, self.event_type_strength
        )

        interactions_labels = self.create_labels(test=interactions_test)

        self.model.train(interactions_train, interactions_test)

        interactions_labels["preds"] = [
            self.model.recommend_item(
                user_id=person_id,
                item_ids=interactions_train.contentId.unique(),
                top_n=self.top_n,
            )
            for person_id in interactions_labels["personId"].tolist()
        ]

        print(
            calculate_metrics(
                prediction_col="preds", interactions_labels=interactions_labels
            )
        )

        self.saver.save(self.model, "mf_model.pkl")


class XGBModelTrainingPipeline(ModelTrainingPipeline):
    def training_preprocess(self, interactions_df, event_type_strength):
        """
        Preprocesses interactions data by sorting, adding contextual columns, and calculating event strength.

        Args:
            interactions_df (pd.DataFrame): DataFrame containing interaction data with columns for person, content, and timestamp.
            event_type_strength (dict): Dictionary mapping event types to their corresponding strength values.

        Returns:
            pd.DataFrame: Preprocessed interactions DataFrame with new columns for 'lastContentId', 'lastEventType',
                          and 'eventStrength'.
        """
        # Preprocess interactions
        interactions_df = interactions_df.sort_values(["personId", "timestamp"])
        interactions_df["lastContentId"] = (
            interactions_df.groupby("personId")["contentId"]
            .shift(1)
            .fillna(-999)
            .astype(int)
            .astype(str)
        )
        interactions_df["lastContentId"] = np.where(
            interactions_df["lastContentId"] == "-999",
            "NONE",
            interactions_df["lastContentId"],
        )
        interactions_df["lastEventType"] = interactions_df.groupby("personId")[
            "eventType"
        ].shift(1)
        interactions_df["eventStrength"] = interactions_df["eventType"].apply(
            lambda x: event_type_strength[x]
        )

        return interactions_df

    def preprocess_articles(self, articles_df, interactions_df):
        """
        Preprocesses articles data by filtering relevant records and merging additional attributes to interactions.

        Args:
            articles_df (pd.DataFrame): DataFrame containing article data with columns like 'eventType', 'contentId', and 'timestamp'.
            interactions_df (pd.DataFrame): DataFrame containing interactions data to merge article information into.

        Returns:
            tuple: (pd.DataFrame, pd.DataFrame) - training and test interaction data after preprocessing.
        """
        # Preprocess articles
        articles_df = articles_df[articles_df["eventType"] == "CONTENT SHARED"]
        articles_df["dateTime"] = pd.to_datetime(articles_df["timestamp"], unit="s")
        articles_df["authorPersonId"] = (
            articles_df["authorPersonId"].astype(int).astype(str)
        )
        articles_df = articles_df.merge(
            interactions_df[["contentId"]].groupby("contentId").count(),
            how="inner",
            on="contentId",
        )
        interactions_df = interactions_df.merge(
            articles_df[
                [
                    "contentId",
                    "authorPersonId",
                    "authorRegion",
                    "authorCountry",
                    "lang",
                    "text",
                ]
            ],
            how="left",
            on="contentId",
        )

        interactions_train, interactions_test = self.train_test_split(
            interactions_df, self.split_date
        )

        return interactions_train, interactions_test

    def run(self):
        """
        Executes the XGBoost model training pipeline. Loads interaction and article data, preprocesses it,
        trains the model, generates recommendations, evaluates predictions, and saves the model.

        Args:
            None

        Returns:
            None
        """
        interactions_df = self.data_loader.get_interactions()
        articles_df = self.data_loader.get_articles()

        interactions_df = self.training_preprocess(
            interactions_df.copy(), self.event_type_strength
        )

        interactions_train, interactions_test = self.preprocess_articles(
            articles_df.copy(), interactions_df.copy()
        )

        self.model.train(interactions_train.copy())
        articles_fs = pd.read_parquet("feature_stores/articles.parquet")

        interactions_test["preds"] = [
            self.model.recommend_items(
                df=interactions_test.iloc[[idx]][
                    [
                        "dateTime",
                        "contentId",
                        "personId",
                        "userRegion",
                        "userCountry",
                        "lastContentId",
                        "lastEventType",
                    ]
                ],
                articles=articles_fs,
                top_n=self.top_n,
            )["content"]
            for idx in range(len(interactions_test))
        ]

        interactions_test["contentId"] = interactions_test["contentId"].apply(
            lambda x: [x]
        )
        print(
            calculate_metrics(
                prediction_col="preds", interactions_labels=interactions_test
            )
        )

        self.saver.save(self.model, file_name="xgb_model.pkl")
