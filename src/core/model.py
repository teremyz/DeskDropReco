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
    """Matrix factorization + user & item bias, weight init., sigmoid_range"""

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
    def train(self, interactions_df):
        global_top = (
            interactions_df[["contentId", "eventStrength"]]
            .groupby("contentId")
            .count()["eventStrength"]
            .sort_values(ascending=False)
            .reset_index()
        )
        self.popularity_list = global_top["contentId"].tolist()

    def recommend_items(self, topn=5):
        return self.popularity_list[:topn]


class PytorchMatrixFactorizationModel:
    def __init__(self, model, batch_size, lr, num_epochs, num_workers=4, device="cpu"):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

    def prerpocess_train(self, interactions_train, interactions_test):
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
        torch.save(self.model.state_dict(), path)

    def load_pytorch_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def get_user_embeddings(self, user_ids):
        user_id = torch.tensor(user_ids, device=self.device)
        return self.model.user_emb(user_id).detach().cpu().numpy()

    def get_item_embeddings(self, item_ids):
        item_id = torch.tensor(item_ids, device=self.device)
        return self.model.item_emb(item_id).detach().cpu().numpy()


class XGBCustomModel:
    def __init__(self, model, mf_model, person_le, content_le):
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
        df = self.preprocess_inputs(df)
        self.model.fit(
            df[self.embedding_cols + self.other_cols + self.target_enc_cols],
            df[self.target],
        )
        self.trained = True

    def predict(self, df):
        df = self.preprocess_inputs(df)
        return self.model.predict(
            df[self.embedding_cols + self.other_cols + self.target_enc_cols]
        )

    def recommend_items(self, df, articles, top_n):
        df = df.rename(columns={"contentId": "currentContentId"})
        df = df.merge(articles, how="cross")

        df["preds"] = self.predict(df)

        df = df.sort_values("preds", ascending=False)

        df["realContentId"] = self.content_le.inverse_transform(df["contentId"])

        return {
            "preds": df["preds"].tolist()[:top_n],
            "content": df["realContentId"].tolist()[:top_n],
        }
