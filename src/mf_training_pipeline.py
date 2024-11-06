"""
This script executes the Deskdrop recommendation system's main training pipeline.
It loads configuration settings, prepares data, initializes the recommendation model,
and runs the training pipeline, outputting a trained model ready for making predictions
on user-article interactions.

Classes and functions used:
- CsvDataLoader: Loads interaction and article data.
- PickleSaver: Saves the trained model.
- MatrixFactorizationTrainingPipeline: Manages the data loading, training, and model saving processes.
- PytorchMatrixFactorizationModel: Wrapper for the recommendation model.
- MFAdvanced: Defines the matrix factorization model architecture.
"""

import typer
from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import MatrixFactorizationTrainingPipeline
from src.core.model import PytorchMatrixFactorizationModel, MFAdvanced
from src.core.utils import load_params


def main(config_path="config.yaml"):
    """
    Main function that loads configuration settings, initializes the data and model,
    and runs the training pipeline for a recommendation model on user-article interactions.

    Args:
        config_path (str): The path to the configuration file in YAML format. Default is "config.yaml".

    Returns:
        None
    """
    config = load_params(config_path)

    num_users = len(
        CsvDataLoader(
            interactions_path=config.basic.interactions_path,
            articles_path=config.basic.articles_path,
        )
        .get_interactions()
        .personId.unique()
    )

    num_items = len(
        CsvDataLoader(
            interactions_path=config.basic.interactions_path,
            articles_path=config.basic.articles_path,
        )
        .get_interactions()
        .contentId.unique()
    )

    training_pipeline = MatrixFactorizationTrainingPipeline(
        data_loader=CsvDataLoader(
            interactions_path=config.basic.interactions_path,
            articles_path=config.basic.articles_path,
        ),
        saver=PickleSaver(path=config.basic.artifact_dir),
        model=PytorchMatrixFactorizationModel(
            model=MFAdvanced(
                num_users=num_users,
                num_items=num_items,
                emb_dim=config.matrix_factorization.emb_dim,
                init=config.matrix_factorization.init,
                bias=config.matrix_factorization.bias,
                sigmoid=config.matrix_factorization.sigmoid,
            ),
            batch_size=config.matrix_factorization.batch_size,
            lr=config.matrix_factorization.lr,
            num_epochs=config.matrix_factorization.num_epochs,
            num_workers=config.matrix_factorization.num_workers,
            device=config.matrix_factorization.device,
        ),
        top_n=config.basic.top_n,
        split_date=config.basic.split_date,
        event_type_strength=config.event_type_strength,
    )

    training_pipeline.run()


if __name__ == "__main__":
    typer.run(main)
