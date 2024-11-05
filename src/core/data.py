from torch.utils.data import Dataset


class ContentDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df[["personId", "contentId", "eventStrength"]]
        self.x_user_content = list(zip(df["personId"].values, df["contentId"].values))
        self.y_rating = self.df["eventStrength"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x_user_content[idx], self.y_rating[idx]
