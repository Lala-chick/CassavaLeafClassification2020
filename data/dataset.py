from torch.utils.data import Dataset
import cv2
import torch
import pandas as pd

class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = self.df.iloc[index]["label"]
        path = "{}/{}".format(self.data_root, self.df.iloc[index]["image_id"])

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label:
            return img, target
        else:
            return img