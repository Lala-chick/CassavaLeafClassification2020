import cv2
from torch.utils.data import Dataset, DataLoader
import augmentations

def get_img(path):
    img_bgr = cv2.imread(path)
    img_rgb = img_bgr[:,:,::-1]
    return img_rgb

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
            target = self.df.iloc[index]['label']
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])

        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label:
            return img, target
        else:
            return img

def train_dataloader(df, args, trn_idx, data_root='../CassavaLeafClassification2020/data/train_images/'):
    
    _train = df.loc[trn_idx,:].reset_index(drop=True)
    train_ds = CassavaDataset(_train, data_root, transforms=augmentations.get_train_transforms(args), output_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )
    return train_loader

def valid_dataloader(df, args, val_idx, data_root="../CassavaLeafClassification2020/data/train_images/"):
    _valid = df.loc[val_idx,:].reset_index(drop=True)
    valid_ds = CassavaDataset(_valid, data_root, transforms=augmentations.get_valid_transforms(args), output_label=True)

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.valid_bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )
    return valid_loader

def test_dataloader(df, args, model, data_root="../CassavaLeafClassification2020/data/test_images"):
    _test = df.reset_index(drop=True)
    if model == "ViT":
        test_ds = CassavaDataset(_test, data_root, transforms=augmentations.get_inference_Vit_transforms(args), output_label=False)
    else:
        test_ds = CassavaDataset(_test, data_root, transforms=augmentations.get_inference_transforms(args), output_label=False)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_bs,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )
    return test_loader