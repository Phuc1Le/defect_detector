import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
class BottleDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = Path(row.filepath.replace("\\", "/"))
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row.label)