import pandas as pd
import pickle
import os


class CsvDataLoader:
    def __init__(self, interactions_path, articles_path):
        self.interactions_path = interactions_path
        self.articles_path = articles_path

    def get_interactions(self):
        return pd.read_csv(self.interactions_path)

    def get_articles(self):
        return pd.read_csv(self.articles_path)


class PickleSaver:
    def __init__(self, path):
        self.path = path

    def save(self, artifact, file_name):
        pickle.dump(artifact, open(os.path.join(self.path, file_name), "wb"))
