import argparse
import random
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch import nn

from data import train_dataloader, valid_dataloader, cutmix, fmix
from models import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    epoch,
    model,
    loss_fn,
    optimizer,
    train_loader,
    device,
    scaler,
    cfg,
    scheduler=None,
    schd_batch_update=False,
):
    model.train()

    running_loss = None

    pbar = tqdm(
        enumerate(train_loader), total=len(train_loader), position=0, leave=True
    )
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        mix_decision = np.random.rand()
        if mix_decision < 0.25:
            imgs, image_labels = cutmix(imgs, image_labels, 1.0)
        elif mix_decision >= 0.25 and mix_decision < 0.5:
            imgs, image_labels = fmix(
                imgs,
                image_labels,
                alpha=1.0,
                decay_power=5.0,
                shape=(cfg.img_size, cfg.img_size),
                device=device,
            )
        # print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs.float())  # output = model(input)
            # print(image_preds.shape, exam_pred.shape)

            if mix_decision < 0.50:
                loss = loss_fn(image_preds, image_labels[0]) * image_labels[
                    2
                ] + loss_fn(image_preds, image_labels[1]) * (1.0 - image_labels[2])
            else:
                loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * 0.99 + loss.item() * 0.01

            if ((step + 1) % cfg.accum_iter == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % cfg.verbose_step == 0) or (
                (step + 1) == len(train_loader)
            ):
                description = f"epoch {epoch} loss: {running_loss:.4f}"

                pbar.set_description(description)

    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(
    epoch,
    model,
    loss_fn,
    val_loader,
    device,
    cfg,
    scheduler=None,
    schd_loss_update=False,
):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % cfg.verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f"epoch {epoch} loss: {loss_sum/sample_num:.4f}"
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()
    print("validation multi-class accuracy = {:.4f}".format(accuracy))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()

    return accuracy


def is_resnet(model):
    resnet_models = [
        "resnest26d",
        "resnest50d",
        "resnest50d_1s4x24d",
        "resnest50d_4s2x40d",
        "resnest101e",
        "resnest200e",
        "resnest269e",
        "resnet18",
        "resnet18d",
        "resnet26",
        "resnet26d",
        "resnet34",
        "resnet34d",
        "resnet50",
        "resnet50d",
        "resnet101",
        "resnet101d",
        "resnet101d_320",
        "resnet152",
        "resnet152d",
        "resnet152d_320",
        "resnet200",
        "resnet200d",
        "resnet200d_320",
        "resnetblur18",
        "resnetblur50",
        "resnext50_32x4d",
        "resnext50d_32x4d",
        "resnext101_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d",
        "ig_resnext101_32x8d",
        "ig_resnext101_32x16d",
        "ig_resnext101_32x32d",
        "ig_resnext101_32x48d",
    ]

    if model in resnet_models:
        return True
    return False


def is_effnet(model):
    effnet_models = [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b1_pruned",
        "efficientnet_b2",
        "efficientnet_b2_pruned",
        "efficientnet_b2a",
        "efficientnet_b3",
        "efficientnet_b3_pruned",
        "efficientnet_b3a",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_b8",
        "efficientnet_cc_b0_4e",
        "efficientnet_cc_b0_8e",
        "efficientnet_cc_b1_8e",
        "efficientnet_el",
        "efficientnet_em",
        "efficientnet_es",
        "efficientnet_l2",
        "efficientnet_lite0",
        "efficientnet_lite1",
        "efficientnet_lite2",
        "efficientnet_lite3",
        "efficientnet_lite4",
        "tf_efficientnet_b0",
        "tf_efficientnet_b0_ap",
        "tf_efficientnet_b0_ns",
        "tf_efficientnet_b1",
        "tf_efficientnet_b1_ap",
        "tf_efficientnet_b1_ns",
        "tf_efficientnet_b2",
        "tf_efficientnet_b2_ap",
        "tf_efficientnet_b2_ns",
        "tf_efficientnet_b3",
        "tf_efficientnet_b3_ap",
        "tf_efficientnet_b3_ns",
        "tf_efficientnet_b4",
        "tf_efficientnet_b4_ap",
        "tf_efficientnet_b4_ns",
        "tf_efficientnet_b5",
        "tf_efficientnet_b5_ap",
        "tf_efficientnet_b5_ns",
        "tf_efficientnet_b6",
        "tf_efficientnet_b6_ap",
        "tf_efficientnet_b6_ns",
        "tf_efficientnet_b7",
        "tf_efficientnet_b7_ap",
        "tf_efficientnet_b7_ns",
        "tf_efficientnet_b8",
        "tf_efficientnet_b8_ap",
        "tf_efficientnet_cc_b0_4e",
        "tf_efficientnet_cc_b0_8e",
        "tf_efficientnet_cc_b1_8e",
        "tf_efficientnet_el",
        "tf_efficientnet_em",
        "tf_efficientnet_es",
        "tf_efficientnet_l2_ns",
        "tf_efficientnet_l2_ns_475",
        "tf_efficientnet_lite0",
        "tf_efficientnet_lite1",
        "tf_efficientnet_lite2",
        "tf_efficientnet_lite3",
        "tf_efficientnet_lite4",
    ]

    if model in effnet_models:
        return True
    return False


def is_ViT(model):
    vit_models = [
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_patch32_384",
        "vit_base_resnet26d_224",
        "vit_base_resnet50d_224",
        "vit_huge_patch16_224",
        "vit_huge_patch32_384",
        "vit_large_patch16_224",
        "vit_large_patch16_384",
        "vit_large_patch32_384",
        "vit_small_patch16_224",
        "vit_small_resnet26d_224",
        "vit_small_resnet50d_s3_224",
    ]
    if model in vit_models:
        return True
    return False


def main(cfg):
    train = pd.read_csv("../CassavaLeafClassification2020/train.csv")
    seed_everything(cfg.seed)
    cfg_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    folds = StratifiedKFold(
        n_splits=cfg.fold_num, shuffle=True, random_state=cfg.seed
    ).split(np.arange(train.shape[0]), train.label.values)
    save_path = "../CassavaLeafClassification2020/pretrained/" + cfg.model
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        train_loader = train_dataloader(
            train,
            cfg,
            trn_idx,
            data_root="../CassavaLeafClassification2020/train_images",
        )
        valid_loader = valid_dataloader(
            train,
            cfg,
            val_idx,
            data_root="../CassavaLeafClassification2020/train_images",
        )

        device = torch.device(cfg_device)

        if is_resnet(cfg.model):
            model = ResNetClassifier(
                cfg.model, train.label.nunique(), pretrained=True
            ).to(device)
        elif is_effnet(cfg.model):
            model = EffNetClassifier(
                cfg.model, train.label.nunique(), pretrained=True
            ).to(device)
        else:
            model = ViTClassifier(
                cfg.model, train.label.nunique(), pretrained=True
            ).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1
        )
        loss_fn = nn.CrossEntropyLoss().to(device)

        best_acc = 0
        best_epoch = 0

        for epoch in range(cfg.epoch):
            train_one_epoch(
                epoch,
                model,
                loss_fn,
                optimizer,
                train_loader,
                device,
                scaler,
                cfg,
                scheduler=scheduler,
                schd_batch_update=False,
            )
            with torch.no_grad():
                epoch_acc = valid_one_epoch(
                    epoch, model, loss_fn, valid_loader, device, cfg
                )
            if epoch_acc > best_acc:
                torch.save(
                    model.state_dict(),
                    save_path + "{}_fold_{}_{}.pth".format(cfg.model, fold, epoch),
                )
                best_acc = epoch_acc
                best_epoch = epoch
                print("This model is saved")

        del model, optimizer, train_loader, valid_loader, scaler, scheduler
        torch.cuda.empty_cache()
        print("Best Accuracy: {} in epoch {}".format(best_acc, best_epoch))


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--model", type=str, default="tf_efficientnet_b3_ns")
    cli_parser.add_argument("--train_bs", type=int, default=16)
    cli_parser.add_argument("--valid_bs", type=int, default=32)
    cli_parser.add_argument("--test_bs", type=int, default=32)
    cli_parser.add_argument("--epoch", type=int, default=10)
    cli_parser.add_argument("--fold_num", type=int, default=5)
    cli_parser.add_argument("--lr", type=float, default=1e-4)
    cli_parser.add_argument("--weight_decay", type=float, default=1e-6)
    cli_parser.add_argument("--num_workers", type=int, default=4)
    cli_parser.add_argument("--accum_iter", type=int, default=2)
    cli_parser.add_argument("--verbose_step", type=int, default=1)
    cli_parser.add_argument("--img_size", type=int, default=512)
    cli_parser.add_argument("--seed", type=int, default=209)
    cli_parser.add_argument("--T_0", type=int, default=10)
    cli_parser.add_argument("--min_lr", type=float, default=1e-6)
    cli_parser.add_argument("--vit_img", type=int, default=384)

    cfg = cli_parser.parse_args()

    main(cfg)
