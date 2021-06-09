import argparse
import pandas as pd
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from scipy.special import softmax

from data import test_dataloader
from models import *

def inference(model, test_loader, device):
    preds = []
    model.eval()
    test_tqdm = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    for images in test_tqdm:
        images = images.to(device)
        preds.extend(model(images).detach().cpu().numpy())
    return preds

def test_time_augment(model, paths, device, test_loader, cfg):
    preds = []
    for pretrained_model in paths:
        model.to(device)
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        with torch.no_grad():
            for i in range(cfg.tta):
                preds += [inference(model, test_loader, device)]
    preds = np.mean(preds, axis=0)
    return preds

def main(cfg):
    df = pd.read_csv("../CassavaLeafClassification2020/sample_submission.csv")
    test = df.copy()
    cfg_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg_device)
    if cfg.ViT:
        test_loader_vit = test_dataloader(test, cfg, "ViT", data_root="../CassavaLeafClassification2020/test_images")
    if cfg.efficientnet_b3 or cfg.efficientnet_b4 or cfg.resnet_50 or cfg.resnext_50 or cfg.resnext_101: 
        test_loader = test_dataloader(test, cfg, "others", data_root="../CassavaLeafClassification2020/test_images")
    
    final_preds = None
    if cfg.efficientnet_b3:
        effnet_b3 = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/effnet_b3/*.pth"):
            effnet_b3.append(filepath)
        model = EffNetClassifier("tf_efficientnet_b3_ns", 5)
        b3_outcomes = test_time_augment(model, effnet_b3, device, test_loader, cfg)
        b3_outcomes = pd.concat([df['image_id'], pd.DataFrame(b3_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = b3_outcomes.drop('image_id', axis=1).to_numpy()
        else:
            final_preds += b3_outcomes.drop('image_id', axis=1).to_numpy()
    
    if cfg.efficientnet_b4: 
        effnet_b4 = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/effnet_b4/*.pth"):
            effnet_b4.append(filepath)
        model = EffNetClassifier("tf_efficientnet_b4_ns", 5)
        b4_outcomes = test_time_augment(model, effnet_b4, device, test_loader, cfg)
        b4_outcomes = pd.concat([df['image_id'], pd.DataFrame(b4_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = b4_outcomes.drop("image_id", axis=1).to_numpy()
        else:
            final_preds += b4_outcomes.drop("image_id", axis=1).to_numpy()


    if cfg.resnet_50: 
        resnet_50 = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/resnet50d/*.pth"):
            resnet_50.append(filepath)
        model = ResNetClassifier("resnet50d", 5)
        resnet_outcomes = test_time_augment(model, resnet_50, device, test_loader, cfg)
        resnet_outcomes = pd.concat([df['image_id'], pd.DataFrame(resnet_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = resnet_outcomes.drop("image_id", axis=1).to_numpy()
        else:
            final_preds += resnet_outcomes.drop("image_id", axis=1).to_numpy()
 

    if cfg.resnext_50: 
        resnext_50 = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/resnext50/*.pth"):
            resnext_50.append(filepath)
        model = ResNetClassifier("resnext50d_32x4d", 5)
        resnext_outcomes = test_time_augment(model, resnext_50, device, test_loader, cfg)
        resnext_outcomes = pd.concat([df['image_id'], pd.DataFrame(resnext_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = resnext_outcomes.drop("image_id", axis=1).to_numpy()
        else:
            final_preds += resnext_outcomes.drop("image_id", axis=1).to_numpy()

    
    if cfg.resnext_101: 
        resnext_101 = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/resnext101/*.pth"):
            resnext_101.append(filepath)
        model = ResNetClassifier("ig_resnext101_32x8d", 5)
        resnext101_outcomes = test_time_augment(model, resnext_101, device, test_loader, cfg)
        resnext101_outcomes = pd.concat([df['image_id'], pd.DataFrame(resnext101_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = resnext101_outcomes.drop("image_id", axis=1).to_numpy()
        else:
            final_preds += resnext101_outcomes.drop("image_id", axis=1).to_numpy()

    if cfg.ViT: 
        ViT = []
        for filepath in glob.iglob(f"../CassavaLeafClassification2020/pretrained/ViT/*.pth"):
            ViT.append(filepath)
        model = ViTClassifier("vit_base_patch16_384", 5)
        vit_outcomes = test_time_augment(model, ViT, device, test_loader_vit, cfg)
        vit_outcomes = pd.concat([df['image_id'], pd.DataFrame(vit_outcomes)], axis=1).sort_values(['image_id'])
        if final_preds is None:
            final_preds = vit_outcomes.drop("image_id", axis=1).to_numpy()
        else:
            final_preds += vit_outcomes.drop("image_id", axis=1).to_numpy()

    final_preds = softmax(final_preds).argmax(1)
    submit = pd.DataFrame({'image_id': df['image_id'].values, 'label': final_preds})
    submit.to_csv('./submission.csv', index=False)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--efficientnet_b3", type=bool, default=True)
    cli_parser.add_argument("--efficientnet_b4", type=bool, default=False)
    cli_parser.add_argument("--resnet_50", type=bool, default=False)
    cli_parser.add_argument("--resnext_50", type=bool, default=True)
    cli_parser.add_argument("--resnext_101", type=bool, default=False)
    cli_parser.add_argument("--ViT", type=bool, default=True)
    cli_parser.add_argument("--test_bs", type=int, default=32)
    cli_parser.add_argument("--tta", type=int, default=3)
    cli_parser.add_argument("--num_workers", type=int, default=4)
    cli_parser.add_argument("--img_size", type=int, default=512)
    cli_parser.add_argument("--vit_img", type=int, default=384)

    cfg = cli_parser.parse_args()

    main(cfg)