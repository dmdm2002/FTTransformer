from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, categorical_features, numerical_features, y):
        super().__init__()

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        categorical_features = self.categorical_features[idx]
        numerical_feature = self.numerical_features[idx]
        y = self.y[idx]

        return categorical_features, numerical_feature, y