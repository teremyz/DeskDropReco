"""
This script runs the main training pipeline for a popularity-based recommendation system
on Deskdrop's user-article interaction data. It loads the configuration, prepares the data,
initializes the popularity recommendation model, and executes the training pipeline.

Classes and functions used:
- CsvDataLoader: Loads interaction and article data for the recommendation system.
- PickleSaver: Saves the trained model.
- PopularityModelTrainingPipeline: Manages the data loading, training, and saving processes.
- PopularityRecommender: Implements a popularity-based recommendation model.
"""

import typer
from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import PopularityModelTrainingPipeline
from src.core.model import PopularityRecommender
from src.core.utils import load_params


def main(config_path="config.yaml"):
    """
    Main function to initialize configuration settings, data loading, and
    popularity-based recommendation model, then runs the training pipeline.

    Args:
        config_path (str): The path to the YAML configuration file. Default is "config.yaml".

    Returns:
        None
    """
    config = load_params(config_path)

    training_pipeline = PopularityModelTrainingPipeline(
        data_loader=CsvDataLoader(
            interactions_path=config.basic.interactions_path,
            articles_path=config.basic.articles_path,
        ),
        saver=PickleSaver(path=config.basic.artifact_dir),
        model=PopularityRecommender(),
        top_n=config.basic.top_n,
        split_date=config.basic.split_date,
        event_type_strength=config.event_type_strength,
    )

    training_pipeline.run()


if __name__ == "__main__":
    typer.run(main)
