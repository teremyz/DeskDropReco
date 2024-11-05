import logging
import os
import pickle
from xgboost import XGBRegressor

from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import XGBModelTrainingPipeline
from src.core.model import XGBCustomModel
from src.core.utils import load_params


logging.basicConfig(level=logging.INFO)


def main(config_path="config.yaml"):
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
    main()
