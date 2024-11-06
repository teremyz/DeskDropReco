"""
This script runs the main training pipeline for a recommendation system based on
XGBoost and a pre-trained matrix factorization model. The system is designed for
Deskdrop's user-article interaction data, using features from both interactions and
content attributes to provide article recommendations.

Classes and functions used:
- CsvDataLoader: Loads interaction and article data for the recommendation system.
- PickleSaver: Handles the storage of trained models.
- XGBModelTrainingPipeline: Manages data loading, training, and saving processes.
- XGBCustomModel: Implements a recommendation model using XGBoost and matrix factorization.
"""

import typer
import os
import pickle
from xgboost import XGBRegressor

from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import XGBModelTrainingPipeline
from src.core.model import XGBCustomModel
from src.core.utils import load_params


def main(config_path="config.yaml"):
    """
    Main function to initialize configuration, load data, set up the XGBoost and matrix
    factorization-based recommendation model, and run the training pipeline.

    Args:
        config_path (str): The path to the YAML configuration file. Default is "config.yaml".

    Returns:
        None
    """
    config = load_params(config_path)

    training_pipeline = XGBModelTrainingPipeline(
        data_loader=CsvDataLoader(
            interactions_path=config.basic.interactions_path,
            articles_path=config.basic.articles_path,
        ),
        saver=PickleSaver(path=config.basic.artifact_dir),
        model=XGBCustomModel(
            model=XGBRegressor(**config.xgb_params),
            mf_model=pickle.load(
                open(os.path.join(config.basic.artifact_dir, "mf_model.pkl"), "rb")
            ),
            person_le=pickle.load(
                open(
                    os.path.join(
                        config.basic.artifact_dir, "personId_label_encoder.pkl"
                    ),
                    "rb",
                )
            ),
            content_le=pickle.load(
                open(
                    os.path.join(
                        config.basic.artifact_dir, "contentId_label_encoder.pkl"
                    ),
                    "rb",
                )
            ),
        ),
        top_n=config.basic.top_n,
        split_date=config.basic.split_date,
        event_type_strength=config.event_type_strength,
    )

    training_pipeline.run()


if __name__ == "__main__":
    typer.run(main)
