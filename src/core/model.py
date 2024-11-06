import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from src.core.data import ContentDataset
from torch import nn

from sklearn.preprocessing import TargetEncoder


def sigmoid_range(x, low, high):
    return torch.sigmoid(x) * (high - low) + low


class MFAdvanced(nn.Module):
    """
    Matrix Factorization model with user and item bias, weight initialization, and optional sigmoid range.

    Args:
        num_users (int): The number of unique users in the dataset.
        num_items (int): The number of unique items in the dataset.
        emb_dim (int): The dimensionality of the embedding space.
        init (bool): Flag indicating whether to initialize embeddings with a uniform distribution.
        bias (bool): Flag indicating whether to include user and item biases.
        sigmoid (bool): Flag indicating whether to apply a sigmoid range transformation to the output.

    """

    def __init__(self, num_users, num_items, emb_dim, init, bias, sigmoid):
        super().__init__()
        self.bias = bias
        self.sigmoid = sigmoid
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        if bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users))
            self.item_bias = nn.Parameter(torch.zeros(num_items))
            self.offset = nn.Parameter(torch.zeros(1))
        if init:
            self.user_emb.weight.data.uniform_(0.0, 0.05)
            self.item_emb.weight.data.uniform_(0.0, 0.05)

    def forward(self, user, item):
        """
        Forward pass to compute the interaction score for given user and item.

        Args:
            user (torch.LongTensor): Tensor of user indices.
            item (torch.LongTensor): Tensor of item indices.

        Returns:
            torch.FloatTensor: Interaction scores for each user-item pair, with optional sigmoid range.
        """
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        element_product = (user_emb * item_emb).sum(1)
        if self.bias:
            user_b = self.user_bias[user]
            item_b = self.item_bias[item]
            element_product += user_b + item_b + self.offset
        if self.sigmoid:
            return sigmoid_range(element_product, 0, 5.5)
        return element_product


class PopularityRecommender:
    """
    A simple recommender system based on the popularity of items, using event strength to rank content.

    Summary:
    This class provides functionality to train a popularity-based recommendation model and recommend
    items based on their popularity (the number of interactions).

    Methods:
        train (interactions_df): Trains the recommender model using interaction data, ranking content
                                  by the number of interactions.
        recommend_items (topn): Recommends the top-N most popular items based on interaction data.

    """

    def train(self, interactions_df):
        """
        Trains the popularity-based recommendation model by ranking items based on their interaction counts.

        Args:
            interactions_df (pd.DataFrame): A dataframe containing user interaction data with at least
                                             columns 'contentId' and 'eventStrength'.

        Returns:
            None: The method updates the internal popularity list of items for future recommendations.
        """
        global_top = (
            interactions_df[["contentId", "eventStrength"]]
            .groupby("contentId")
            .count()["eventStrength"]
            .sort_values(ascending=False)
            .reset_index()
        )
        self.popularity_list = global_top["contentId"].tolist()

    def recommend_items(self, topn=5):
        """
        Recommends the top-N most popular items based on the trained model.

        Args:
            topn (int): The number of top recommendations to return. Default is 5.

        Returns:
            list: A list of the top-N most popular content IDs.
        """
        return self.popularity_list[:topn]


class PytorchMatrixFactorizationModel:
    """
    A PyTorch-based matrix factorization model for collaborative filtering.
    This model learns user-item embeddings and predicts user preferences for unseen items.

    Summary:
    This class provides methods for training a matrix factorization model using PyTorch, evaluating its performance,
    recommending items for users, and saving/loading the model. It uses the Adam optimizer and Mean Squared Error loss.

    Methods:
        __init__ (model, batch_size, lr, num_epochs, num_workers=4, device="cpu"): Initializes the model with given parameters.
        prerpocess_train (interactions_train, interactions_test): Prepares training and validation datasets.
        train (interactions_train, interactions_test): Trains the model using the training data and evaluates it on the validation set.
        recommend_item (user_id, item_ids, top_n): Recommends top-N items for a given user.
        save_pytorch_model (path): Saves the model to the specified path.
        load_pytorch_model (path): Loads the model from the specified path.
        get_user_embeddings (user_ids): Retrieves the embeddings for a given set of users.
        get_item_embeddings (item_ids): Retrieves the embeddings for a given set of items.
    """

    def __init__(self, model, batch_size, lr, num_epochs, num_workers=4, device="cpu"):
        """
        Initializes the PyTorch Matrix Factorization Model with the specified parameters.

        Args:
            model (nn.Module): The matrix factorization model architecture.
            batch_size (int): The batch size for training.
            lr (float): The learning rate for optimization.
            num_epochs (int): The number of epochs to train the model.
            num_workers (int, optional): The number of workers for data loading. Default is 4.
            device (str, optional): The device to run the model on (e.g., "cpu" or "cuda"). Default is "cpu".

        Returns:
            None
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

    def prerpocess_train(self, interactions_train, interactions_test):
        """
        Preprocesses the training and test interaction datasets into DataLoader objects.

        Args:
            interactions_train (pd.DataFrame): The training interactions data.
            interactions_test (pd.DataFrame): The test interactions data.

        Returns:
            tuple: A tuple containing the training and validation DataLoader objects.
        """
        ds_train = ContentDataset(interactions_train)
        ds_val = ContentDataset(interactions_test)
        dl_train = DataLoader(
            ds_train, self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        dl_val = DataLoader(
            ds_val, self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        return dl_train, dl_val

    def train(self, interactions_train, interactions_test):
        """
        Trains the matrix factorization model on the training data and evaluates it on the validation data.

        Args:
            interactions_train (pd.DataFrame): The training interactions data.
            interactions_test (pd.DataFrame): The test interactions data.

        Returns:
            None
        """
        dl_train, dl_val = self.prerpocess_train(interactions_train, interactions_test)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.epoch_train_losses, self.epoch_val_losses = [], []

        for i in range(self.num_epochs):
            train_losses, val_losses = [], []
            # Training
            self.model.train()
            for xb, yb in dl_train:
                xUser = xb[0].to(self.device, dtype=torch.long)
                xItem = xb[1].to(self.device, dtype=torch.long)
                yRatings = yb.to(self.device, dtype=torch.float)
                preds = self.model(xUser, xItem)
                loss = loss_fn(preds, yRatings)
                train_losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            # Evaluation
            self.model.eval()
            for xb, yb in dl_val:
                xUser = xb[0].to(self.device, dtype=torch.long)
                xItem = xb[1].to(self.device, dtype=torch.long)
                yRatings = yb.to(self.device, dtype=torch.float)
                preds = self.model(xUser, xItem)
                loss = loss_fn(preds, yRatings)
                val_losses.append(loss.item())
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            self.epoch_train_losses.append(epoch_train_loss)
            self.epoch_val_losses.append(epoch_val_loss)
            s = (
                f"Epoch: {i}, Train Loss: {epoch_train_loss:0.3f}, "
                f"Val Loss: {epoch_val_loss:0.3f}"
            )
            print(s)

    def recommend_item(self, user_id, item_ids, top_n):
        """
        Recommends top-N items for a given user.

        Args:
            user_id (int): The user ID for whom recommendations are being generated.
            item_ids (list): A list of item IDs to recommend from.
            top_n (int): The number of top recommendations to return.

        Returns:
            list: A list of the top-N recommended item IDs for the given user.
        """
        self.model.eval()

        # Convert IDs to tensors and move to device
        xUser = torch.tensor(
            [user_id] * len(item_ids), device=self.device, dtype=torch.long
        )
        xItem = torch.tensor(item_ids, device=self.device, dtype=torch.long)

        # Get the prediction
        with torch.no_grad():
            predictions = self.model(xUser, xItem).cpu().numpy()

        predictions = [(x, y) for (x, y) in zip(item_ids, predictions)]

        sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        return [t[0] for t in sorted_predictions[:top_n]]

    def save_pytorch_model(self, path):
        """
        Saves the trained PyTorch model to the specified file path.

        Args:
            path (str): The path where the model should be saved.

        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)

    def load_pytorch_model(self, path):
        """
        Loads a pre-trained PyTorch model from the specified file path.

        Args:
            path (str): The path from which the model should be loaded.

        Returns:
            None
        """
        self.model.load_state_dict(torch.load(path))

    def get_user_embeddings(self, user_ids):
        """
        Retrieves the embeddings for the specified user IDs.

        Args:
            user_ids (list): A list of user IDs to retrieve embeddings for.

        Returns:
            np.ndarray: A numpy array containing the user embeddings for the specified user IDs.
        """
        user_id = torch.tensor(user_ids, device=self.device)
        return self.model.user_emb(user_id).detach().cpu().numpy()

    def get_item_embeddings(self, item_ids):
        """
        Retrieves the embeddings for the specified item IDs.

        Args:
            item_ids (list): A list of item IDs to retrieve embeddings for.

        Returns:
            np.ndarray: A numpy array containing the item embeddings for the specified item IDs.
        """
        item_id = torch.tensor(item_ids, device=self.device)
        return self.model.item_emb(item_id).detach().cpu().numpy()


class XGBCustomModel:
    """
    A custom machine learning model that integrates matrix factorization embeddings
    with additional feature engineering for content recommendation.

    Summary:
    This class uses XGBoost for content recommendation predictions. It incorporates matrix
    factorization embeddings for both user and content, applies target encoding for categorical
    variables, and performs preprocessing on input features before training and prediction.

    Methods:
        __init__(self, model, mf_model, person_le, content_le): Initializes the model with necessary components.
        preprocess_inputs(self, df): Preprocesses the input DataFrame by encoding user and content IDs, applying
                                      target encoding, and adding embeddings for users and content.
        train(self, df): Trains the model using the processed input data.
        predict(self, df): Makes predictions using the trained model.
        recommend_items(self, df, articles, top_n): Recommends top-N items based on the model's predictions.
    """

    def __init__(self, model, mf_model, person_le, content_le):
        """
        Initializes the custom model with required components.

        Args:
            model (XGBClassifier or similar): The base model used for predictions.
            mf_model (MatrixFactorizationModel): The matrix factorization model for generating embeddings.
            person_le (LabelEncoder): Label encoder for person (user) IDs.
            content_le (LabelEncoder): Label encoder for content (article) IDs.

        Returns:
            None
        """
        self.model = model
        self.mf_model = mf_model
        self.person_le = person_le
        self.content_le = content_le
        self.target_enc = TargetEncoder()
        self.target_enc_cols = [
            "userRegion",
            "userCountry",
            "lastContentId",
            "lastEventType",
            "authorPersonId",
            "authorRegion",
            "authorCountry",
        ]
        self.target = "eventStrength"
        self.trained = False
        self.embedding_cols = [f"content_embedding{x}" for x in range(32)] + [
            f"user_embedding{x}" for x in range(32)
        ]
        self.other_cols = ["EngDummy", "PtDummy"]
        self.region_big = [
            "IL",
            "NSW",
            "ON",
            "CA",
            "NJ",
            "RJ",
            "GA",
            "TX",
            "NY",
            "MG",
            "SP",
        ]

    def preprocess_inputs(self, df):
        """
        Preprocesses the input DataFrame by encoding categorical features, adding matrix factorization embeddings,
        and handling missing or incorrect data.

        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions and metadata.

        Returns:
            pd.DataFrame: The preprocessed DataFrame ready for model training or prediction.
        """
        df["personId"] = self.person_le.transform(df["personId"])
        df["contentId"] = self.content_le.transform(df["contentId"])

        df["userRegion"] = np.where(df["userRegion"] == "13", np.nan, df["userRegion"])
        df["userRegion"] = np.where(df["userRegion"] == "?", np.nan, df["userRegion"])

        df["userRegion"] = df["userRegion"].apply(
            lambda x: x if x in self.region_big else np.nan
        )

        user_embeddings = self.mf_model.get_user_embeddings(
            user_ids=df.personId.tolist()
        )
        df[[f"user_embedding{x}" for x in range(len(user_embeddings[0]))]] = (
            user_embeddings
        )

        item_embeddings = self.mf_model.get_item_embeddings(
            item_ids=df.contentId.tolist()
        )
        df[[f"content_embedding{x}" for x in range(len(item_embeddings[0]))]] = (
            item_embeddings
        )

        df["EngDummy"] = np.where(df["lang"] == "en", 1, 0)
        df["PtDummy"] = np.where(df["lang"] == "pt", 1, 0)

        df[self.target_enc_cols] = df[self.target_enc_cols].fillna("NONE")

        if self.trained:
            df[self.target_enc_cols] = self.target_enc.transform(
                df[self.target_enc_cols]
            )

        else:
            df[self.target_enc_cols] = self.target_enc.fit_transform(
                df[self.target_enc_cols], df[self.target]
            )

        return df

    def train(self, df):
        """
        Trains the model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The preprocessed DataFrame containing features and target variable.

        Returns:
            None
        """
        df = self.preprocess_inputs(df)
        self.model.fit(
            df[self.embedding_cols + self.other_cols + self.target_enc_cols],
            df[self.target],
        )
        self.trained = True

    def predict(self, df):
        """
        Makes predictions for the provided DataFrame using the trained model.

        Args:
            df (pd.DataFrame): The preprocessed DataFrame containing features for prediction.

        Returns:
            np.ndarray: The predicted event strengths for each row in the DataFrame.
        """
        df = self.preprocess_inputs(df)
        return self.model.predict(
            df[self.embedding_cols + self.other_cols + self.target_enc_cols]
        )

    def recommend_items(self, df, articles, top_n):
        """
        Recommends top-N items for each user based on predicted event strengths.

        Args:
            df (pd.DataFrame): The DataFrame containing user-content interactions.
            articles (pd.DataFrame): The articles DataFrame used to cross-join with the user interactions.
            top_n (int): The number of top recommendations to return.

        Returns:
            dict: A dictionary containing predicted strengths and recommended content IDs.
        """
        df = df.rename(columns={"contentId": "currentContentId"})
        df = df.merge(articles, how="cross")

        df["preds"] = self.predict(df)

        df = df.sort_values("preds", ascending=False)

        df["realContentId"] = self.content_le.inverse_transform(df["contentId"])

        return {
            "preds": df["preds"].tolist()[:top_n],
            "content": df["realContentId"].tolist()[:top_n],
        }
