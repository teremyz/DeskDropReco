import pandas as pd
import pickle
import os


class CsvDataLoader:
    def __init__(self, interactions_path, articles_path):
        """
        Initializes the CsvDataLoader with the file paths for interactions and articles.

        Args:
            interactions_path (str): The file path for the interactions CSV file.
            articles_path (str): The file path for the articles CSV file.

        Returns:
            None
        """
        self.interactions_path = interactions_path
        self.articles_path = articles_path

    def get_interactions(self):
        """
        Loads the interaction data from the CSV file.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing interaction data.
        """
        return pd.read_csv(self.interactions_path)

    def get_articles(self):
        """
        Loads the article data from the CSV file.

        Args:
            None

        Returns:
            pd.DataFrame: A DataFrame containing article data.
        """
        return pd.read_csv(self.articles_path)


class PickleSaver:
    def __init__(self, path):
        """
        Initializes the PickleSaver with a specified directory path for storing pickle files.

        Args:
            path (str): The directory path where pickle files will be saved.

        Returns:
            None
        """
        self.path = path

    def save(self, artifact, file_name):
        """
        Saves the given artifact to a pickle file with the specified file name.

        Args:
            artifact (object): The Python object (e.g., model, data) to be saved.
            file_name (str): The name of the file where the artifact will be saved.

        Returns:
            None
        """
        pickle.dump(artifact, open(os.path.join(self.path, file_name), "wb"))
