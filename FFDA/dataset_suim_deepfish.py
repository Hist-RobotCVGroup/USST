import os
from PIL import Image
from matplotlib import transforms
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

# -------------------------------
# 1. 通用 Segmentation Dataset
# -------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir,transform = None, num_classes=8, image_size=(256,256),dataset_name="suim", mode="train"):
        """
        img_dir: 图像文件夹
        mask_dir: 分割 mask 文件夹
        num_classes: 类别数 (SUIM=8)
        image_size: resize 大小
        mode: "train" / "val"
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.num_classes = num_classes
        self.transform_img = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
        self.transform_mask = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
        ])
        if mode == "train":
            self.aug = T.RandomHorizontalFlip(p=0.5)
        else:
            self.aug = None
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transform = transform
        self.dataset_name = dataset_name.lower()

        # SUIM 的 colormap（示例，实际需按官方定义修改）
        self.suim_colormap = {
            (0, 0, 0): 0,       # background
            (0, 0, 1): 1,     # human diver
            (0, 255, 0): 2,     # fish
            (0, 255, 255): 3,   # reef
            (255, 0, 0): 4,     # robot
            (255, 0, 255): 5,   # wreck
            (255, 255, 0): 6,   # vegetation
            (255, 255, 255): 7  # others
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform_img(img)
        mask_path = self.mask_paths[idx]

        if self.dataset_name == "suim":
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask)
            label_mask = np.zeros(mask.shape[:2], dtype=np.int64)
            for rgb, class_id in self.suim_colormap.items():
                matches = np.all(mask == rgb, axis=-1)
                label_mask[matches] = class_id
            mask = torch.from_numpy(label_mask).long()

        elif self.dataset_name == "deepfish":
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.int64)
            mask = torch.from_numpy(mask).long()

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        mask = self.transform_mask(mask)
        print(img.shape, mask.shape)

        return img, mask

# -------------------------------
# 2. SUIM Dataset
# -------------------------------
def get_suim_dataloader(root, batch_size=4, image_size=(256,256), split_ratio=0.8):
    """
    root: 数据集根目录，假设有 root/images/ root/masks/
    """
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    dataset = SegmentationDataset(img_dir, mask_dir, num_classes=8,dataset_name="suim", image_size=image_size, mode="train")

    n_train = int(split_ratio * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

# -------------------------------
# 3. DeepFish Dataset
# -------------------------------
def get_deepfish_dataloader(root, batch_size=4, image_size=(256,256), split_ratio=0.8):
    """
    root: 数据集根目录，假设有 root/images/ root/masks/
    """
    img_dir = os.path.join(root, "images/valid")
    mask_dir = os.path.join(root, "masks/valid")
    dataset = SegmentationDataset(img_dir, mask_dir, num_classes=8,dataset_name="deepfish", image_size=image_size, mode="train")

    n_train = int(split_ratio * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader
