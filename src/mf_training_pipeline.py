import logging
from src.core.loaders import CsvDataLoader, PickleSaver
from src.core.pipelines import MatrixFactorizationTrainingPipeline
from src.core.model import PytorchMatrixFactorizationModel, MFAdvanced
from src.core.utils import load_params


logging.basicConfig(level=logging.INFO)


def main(config_path="config.yaml"):
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
        saver=PickleSaver(path="./"),
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
    main()
