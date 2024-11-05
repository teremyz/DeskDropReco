import logging
from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import PopularityModelTrainingPipeline
from src.core.model import PopularityRecommender
from src.core.utils import load_params


logging.basicConfig(level=logging.INFO)


def main(config_path="config.yaml"):
    config = load_params(config_path)

    training_pipeline = PopularityModelTrainingPipeline(
        data_loader=CsvDataLoader(
            interactions_path="data/users_interactions.csv",
            articles_path="data/shared_articles.csv",
        ),
        saver=PickleSaver(path="./"),
        model=PopularityRecommender(),
        top_n=config.basic.top_n,
        split_date=config.basic.split_date,
        event_type_strength=config.event_type_strength,
    )

    training_pipeline.run()


if __name__ == "__main__":
    main()
