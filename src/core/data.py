from torch.utils.data import Dataset


class ContentDataset(Dataset):
    """
    Summary:
    This class represents a dataset used for training a machine learning model. It stores user-content interactions and their corresponding event strengths (ratings).
    The dataset is intended for use with PyTorch DataLoader to load batches for training.

    Args:
        df (pd.DataFrame): A DataFrame containing the user-content interaction data. The DataFrame should have the following columns:
                           - "personId" (int): The ID of the user interacting with the content.
                           - "contentId" (int): The ID of the content being interacted with.
                           - "eventStrength" (float): The strength or rating of the event (interaction) between the user and the content.

    Returns:
        None
    """

    def __init__(self, df):
        super().__init__()
        self.df = df[["personId", "contentId", "eventStrength"]]
        self.x_user_content = list(zip(df["personId"].values, df["contentId"].values))
        self.y_rating = self.df["eventStrength"].values

    def __len__(self):
        """
        Summary:
        Returns the number of samples in the dataset.

        Args:
            None

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Summary:
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - (tuple): A tuple (personId, contentId) representing the user and content IDs.
                - (float): The event strength (rating) for the user-content interaction.
        """
        return self.x_user_content[idx], self.y_rating[idx]
