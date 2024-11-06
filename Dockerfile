FROM python:3.10.15-slim

RUN apt-get update && apt-get install -y gcc

RUN pip install -U pip
RUN pip install poetry==1.5.1


WORKDIR /app

COPY ["pyproject.toml", "poetry.lock", "./"]

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

RUN mkdir model
RUN mkdir feature_stores
RUN mkdir src
COPY ["src/", "src/"]
COPY ["model/xgb_model.pkl", "model/popularity_model.pkl" , "model/unique_train_content_ids.pkl", "model/unique_train_preson_ids.pkl", "model/"]
COPY ["feature_stores/articles.parquet", "feature_stores/"]

CMD ["fastapi", "run", "src/predict_api.py", "--port", "8000"]

EXPOSE 80
