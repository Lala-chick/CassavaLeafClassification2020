import cv2
import pandas as pandas
from torch.utils.data import DataLoader
from .dataset import CassavaDataset
from .augmentations import *

def train_dataloader(
    df, args, trn_idx, data_root="../CassavaLeafClassification2020/train_images/"
):

    _train = df.loc[trn_idx, :].reset_index(drop=True)
    train_ds = CassavaDataset(
        _train,
        data_root,
        transforms=get_train_transforms(args),
        output_label=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return train_loader


def valid_dataloader(
    df, args, val_idx, data_root="../CassavaLeafClassification2020/train_images/"
):
    _valid = df.loc[val_idx, :].reset_index(drop=True)
    valid_ds = CassavaDataset(
        _valid,
        data_root,
        transforms=get_valid_transforms(args),
        output_label=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.valid_bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return valid_loader


def test_dataloader(
    df, args, model, data_root="../CassavaLeafClassification2020/test_images"
):
    _test = df.reset_index(drop=True)
    if model == "ViT":
        test_ds = CassavaDataset(
            _test,
            data_root,
            transforms=get_inference_Vit_transforms(args),
            output_label=False,
        )
    else:
        test_ds = CassavaDataset(
            _test,
            data_root,
            transforms=get_inference_transforms(args),
            output_label=False,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return test_loader
