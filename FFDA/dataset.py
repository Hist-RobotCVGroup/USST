import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

# -------------------------------
# 1. 通用 Segmentation Dataset
# -------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes=8, image_size=(256, 256), dataset_name="suim", mode="train"):
        """
        img_dir: 图像文件夹路径
        mask_dir: 掩码文件夹路径  
        num_classes: 类别数量
        image_size: 图像尺寸 (H, W)
        dataset_name: 数据集名称 ("suim" 或 "deepfish")
        mode: 模式 ("train" 或 "val")
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.dataset_name = dataset_name.lower()
        self.mode = mode
        
        # 检查目录是否存在
        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory does not exist: {img_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory does not exist: {mask_dir}")
        
        # 获取文件列表
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir) 
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # 确保文件数量匹配
        if len(self.img_paths) != len(self.mask_paths):
            print(f"Warning: Image count ({len(self.img_paths)}) and mask count ({len(self.mask_paths)}) do not match!")
            # 取较小值
            min_len = min(len(self.img_paths), len(self.mask_paths))
            self.img_paths = self.img_paths[:min_len]
            self.mask_paths = self.mask_paths[:min_len]
        
        if len(self.img_paths) == 0:
            raise ValueError("No images found in the specified directories")
        
        print(f"Loaded {len(self.img_paths)} images and {len(self.mask_paths)} masks for {dataset_name} {mode} set")
        
        # 图像转换
        self.transform_img = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # SUIM 的 colormap (根据实际数据集调整)
        self.suim_colormap = {
            (0, 0, 0): 0,       # background - 黑色
            (0, 0, 255): 1,     # human diver - 蓝色
            (0, 255, 0): 2,     # fish - 绿色
            (0, 255, 255): 3,   # reef - 青色
            (255, 0, 0): 4,     # robot - 红色
            (255, 0, 255): 5,   # wreck - 紫色
            (255, 255, 0): 6,   # vegetation - 黄色
            (255, 255, 255): 7  # others - 白色
        }

    def __len__(self):
        return len(self.img_paths)

    def rgb_to_label(self, mask_rgb):
        """将RGB掩码转换为标签掩码"""
        label_mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        
        for rgb, class_id in self.suim_colormap.items():
            # 将RGB值转换为numpy数组进行比较
            rgb_array = np.array(rgb).reshape(1, 1, 3)
            # 允许一定的颜色容差
            matches = np.all(np.abs(mask_rgb - rgb_array) < 10, axis=-1)
            label_mask[matches] = class_id
        
        return label_mask

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            if self.dataset_name == "suim":
                # SUIM数据集使用RGB掩码
                mask = Image.open(mask_path).convert("RGB")
                mask_np = np.array(mask)
                mask_label = self.rgb_to_label(mask_np)
                mask_pil = Image.fromarray(mask_label.astype(np.uint8))
                
            elif self.dataset_name == "deepfish":
                # DeepFish数据集使用灰度掩码
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask, dtype=np.int64)
                # 将非零值设为1（前景），零值为0（背景）
                mask_np = (mask_np > 0).astype(np.int64)
                mask_np[mask_np == 1] = 6
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        except Exception as e:
            print(f"Error loading image/mask pair {img_path}, {mask_path}: {e}")
            # 返回默认值
            img = Image.new("RGB", self.image_size, color=(0, 0, 0))
            mask_pil = Image.new("L", self.image_size, color=0)
        
        # 应用图像转换
        img_tensor = self.transform_img(img)
        
        # 应用掩码转换
        mask_tensor = torch.from_numpy(np.array(mask_pil.resize(self.image_size, Image.NEAREST))).long()
        
        # 数据增强（仅训练时）
        if self.mode == "train":
            # 随机水平翻转
            if torch.rand(1) > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[-1])
                mask_tensor = torch.flip(mask_tensor, dims=[-1])
            
            # 随机垂直翻转
            if torch.rand(1) > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[-2])
                mask_tensor = torch.flip(mask_tensor, dims=[-2])
        
        # 确保掩码值在有效范围内
        if self.dataset_name == "deepfish":
            mask_tensor = mask_tensor % 2  # DeepFish只有2个类别
        else:
            mask_tensor = mask_tensor % self.num_classes  # SUIM有8个类别
        
        return img_tensor, mask_tensor

# -------------------------------
# 2. SUIM Dataset Loader
# -------------------------------
def get_suim_dataloader(root, batch_size=4, image_size=(256, 256), split_ratio=0.8):
    """
    加载SUIM数据集
    root: 数据集根目录，包含images和masks子目录
    """
    img_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    
    # 创建完整数据集
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        num_classes=8,
        image_size=image_size,
        dataset_name="suim",
        mode="train"
    )
    
    # 分割训练集和验证集
    n_total = len(full_dataset)
    n_train = int(split_ratio * n_total)
    n_val = n_total - n_train
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

# -------------------------------
# 3. DeepFish Dataset Loader  
# -------------------------------
def get_deepfish_dataloader(root, batch_size=4, image_size=(256, 256), split_ratio=0.8):
    """
    加载DeepFish数据集
    root: 数据集根目录，包含images/valid和masks/valid子目录
    """
    img_dir = os.path.join(root, "images", "valid")
    mask_dir = os.path.join(root, "masks", "valid")
    
    # 创建完整数据集
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        # num_classes=2,  # DeepFish只有2个类别
        num_classes=8,  # DeepFish只有2个类别
        image_size=image_size,
        dataset_name="deepfish",
        mode="train"
    )
    
    # 分割训练集和验证集
    n_total = len(full_dataset)
    n_train = int(split_ratio * n_total)
    n_val = n_total - n_train
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    try:
        train_loader, val_loader = get_suim_dataloader("E:/数据集/SUIM数据集/train_val/", batch_size=2)
        print("SUIM dataset loaded successfully")
        
        for i, (img, mask) in enumerate(train_loader):
            print(f"Batch {i}: images {img.shape}, masks {mask.shape}")
            if i == 1:  # 只查看前两个批次
                break
                
    except Exception as e:
        print(f"Error: {e}")