import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from doubao_dataset import get_suim_dataloader, get_deepfish_dataloader

# -------------------------------
# 1. U-Net 定义
# -------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1 = DoubleConv(512, 256)  # 512 = 256(上采样) + 256(跳跃连接)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)  # 256 = 128(上采样) + 128(跳跃连接)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)   # 128 = 64(上采样) + 64(跳跃连接)
        
        # 最终输出层
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)        # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        
        # 解码路径
        d1 = self.up1(e4)        # [B, 256, H/4, W/4]
        d1 = torch.cat([d1, e3], dim=1)  # [B, 512, H/4, W/4]
        d1 = self.dec1(d1)       # [B, 256, H/4, W/4]
        
        d2 = self.up2(d1)        # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)       # [B, 128, H/2, W/2]
        
        d3 = self.up3(d2)        # [B, 64, H, W]
        d3 = torch.cat([d3, e1], dim=1)  # [B, 128, H, W]
        d3 = self.dec3(d3)       # [B, 64, H, W]
        
        output = self.final(d3)  # [B, num_classes, H, W]
        return output

# -------------------------------
# 2. Fixed FDA 实现
# -------------------------------
def fixed_fda(src_img, trg_img, beta=0.1):
    """
    固定带宽的FDA风格迁移
    """
    # 确保输入尺寸一致
    if src_img.shape[-2:] != trg_img.shape[-2:]:
        # 调整目标图像尺寸以匹配源图像
        trg_img = torch.nn.functional.interpolate(
            trg_img, size=src_img.shape[-2:], mode='bilinear', align_corners=False
        )
    
    # 傅里叶变换
    fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
    fft_trg = torch.fft.fft2(trg_img, dim=(-2, -1))
    
    # 提取幅度和相位
    amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
    amp_trg = torch.abs(fft_trg)
    
    # 创建频率掩码
    H, W = src_img.shape[-2:]
    y_freq = torch.fft.fftfreq(H).to(src_img.device).reshape(-1, 1)
    x_freq = torch.fft.fftfreq(W).to(src_img.device).reshape(1, -1)
    freq_dist = torch.sqrt(y_freq**2 + x_freq**2)
    
    # 低频替换
    mask = (freq_dist < beta).float()
    amp_mixed = mask * amp_trg + (1 - mask) * amp_src
    
    # 逆傅里叶变换
    fft_mixed = amp_mixed * torch.exp(1j * pha_src)
    mixed_img = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
    
    return mixed_img

# -------------------------------
# 3. Learnable FDA 实现
# -------------------------------
class LearnableFDA(nn.Module):
    def __init__(self, max_bandwidth=0.1):
        super().__init__()
        self.max_bandwidth = max_bandwidth
        self.bandwidth_predictor = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),  # 添加BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),  # 添加BatchNorm
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),   # 增加中间层
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, src_img, trg_img,strength=1.0):
        # 确保输入尺寸一致
        if src_img.shape[-2:] != trg_img.shape[-2:]:
            trg_img = torch.nn.functional.interpolate(
                trg_img, size=src_img.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # 预测带宽
        combined_input = torch.cat([src_img, trg_img], dim=1)
        b_pred = self.bandwidth_predictor(combined_input) * self.max_bandwidth

        # 控制强度
        b_pred = self.min_bandwidth + (b_pred - self.min_bandwidth) * strength

        # 傅里叶变换
        fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
        fft_trg = torch.fft.fft2(trg_img, dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_trg = torch.abs(fft_trg)
        
        # 创建软频率掩码
        H, W = src_img.shape[-2:]
        y_freq = torch.fft.fftfreq(H).to(src_img.device).reshape(-1, 1)
        x_freq = torch.fft.fftfreq(W).to(src_img.device).reshape(1, -1)
        freq_dist = torch.sqrt(y_freq**2 + x_freq**2)
        
        # 使用sigmoid创建平滑过渡
        soft_mask = torch.sigmoid((b_pred.view(-1, 1, 1) - freq_dist) * 10)
        # amp_mixed = soft_mask * amp_trg + (1 - soft_mask) * amp_src
        soft_mask_expanded = soft_mask.unsqueeze(1)  # 增加通道维度 [batch_size, 1, H, W]
        soft_mask_expanded = soft_mask_expanded.expand_as(amp_trg)  # 扩展为 [batch_size, channels, H, W]
        amp_mixed = soft_mask_expanded * amp_trg + (1 - soft_mask_expanded) * amp_src
        
        # 逆傅里叶变换
        fft_mixed = amp_mixed * torch.exp(1j * pha_src)
        mixed_img = torch.fft.ifft2(fft_mixed, dim=(-2, -1)).real
        
        return mixed_img, b_pred

# -------------------------------
# 4. 评估指标 mIoU + PixelAcc
# -------------------------------
@torch.no_grad()
def compute_miou(model, dataloader, num_classes, device):
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        # 确保输入尺寸正确
        if imgs.shape[-2:] != (256, 256):
            imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear')
        
        preds = model(imgs)
        preds = preds.argmax(1)
        
        # 调整预测和真实标签的尺寸
        if preds.shape[-2:] != masks.shape[-2:]:
            preds = torch.nn.functional.interpolate(
                preds.unsqueeze(1).float(), size=masks.shape[-2:], mode='nearest'
            ).squeeze(1).long()
        
        preds_np = preds.cpu().numpy().flatten()
        masks_np = masks.cpu().numpy().flatten()
        
        # 更新混淆矩阵
        for p, g in zip(preds_np, masks_np):
            if g < num_classes and p < num_classes:
                hist[g, p] += 1
    
    # 计算mIoU和像素精度
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iou)
    pixel_acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    
    return pixel_acc, miou, iou

# -------------------------------
# 5. 主训练循环
# -------------------------------
def train_model(mode, src_loader, trg_loader, val_loader, num_classes=8, epochs=10, device="cuda"):
    model = UNet(num_classes=num_classes).to(device)
    fda_module = LearnableFDA().to(device) if mode == "learnable_fda" else None
    
    # 参数设置
    params = list(model.parameters())
    if fda_module:
        params += list(fda_module.parameters())
    
    optimizer = optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        if fda_module:
            fda_module.train()
        
        total_loss = 0
        batch_count = 0
        
        # 创建数据迭代器，确保两个数据集长度匹配
        min_len = min(len(src_loader), len(trg_loader))
        src_iter = iter(src_loader)
        trg_iter = iter(trg_loader)
        
        for _ in range(min_len):
            try:
                src_data = next(src_iter)
                trg_data = next(trg_iter)
            except StopIteration:
                break
                
            src_img, src_mask = src_data
            trg_img, _ = trg_data
            
            src_img, src_mask, trg_img = src_img.to(device), src_mask.to(device), trg_img.to(device)
            
            # 尺寸检查和调整
            if src_img.shape[-2:] != (256, 256):
                src_img = torch.nn.functional.interpolate(src_img, size=(256, 256), mode='bilinear')
            if trg_img.shape[-2:] != (256, 256):
                trg_img = torch.nn.functional.interpolate(trg_img, size=(256, 256), mode='bilinear')
            if src_mask.shape[-2:] != (256, 256):
                src_mask = torch.nn.functional.interpolate(
                    src_mask.unsqueeze(1).float(), size=(256, 256), mode='nearest'
                ).squeeze(1).long()
            
            # 风格迁移
            if mode == "baseline":
                stylized_img = src_img
            elif mode == "fixed_fda":
                stylized_img = fixed_fda(src_img, trg_img)
            elif mode == "learnable_fda":
                stylized_img, _ = fda_module(src_img, trg_img)
            
            # 前向传播
            preds = model(stylized_img)
            loss = criterion(preds, src_mask)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        pixel_acc, miou, iou = compute_miou(model, val_loader, num_classes, device)
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"[{mode}] Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, PixelAcc={pixel_acc:.4f}, mIoU={miou:.4f}")
    
    return model

# -------------------------------
# 6. 主入口
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
        # 加载数据集
        print("Loading SUIM dataset...")
        src_train, src_val = get_suim_dataloader("E:/数据集/SUIM数据集/train_val/", batch_size=4, image_size=(256, 256))
        
        print("Loading DeepFish dataset...")
        trg_train, trg_val = get_deepfish_dataloader("E:/数据集/DeepFish/Segmentation/", batch_size=4, image_size=(256, 256))
        
        print(f"SUIM dataset: {len(src_train.dataset)} training samples, {len(src_val.dataset)} validation samples")
        print(f"DeepFish dataset: {len(trg_train.dataset)} training samples, {len(trg_val.dataset)} validation samples")
        
        # 测试一个批次以确保尺寸正确
        print("\nTesting batch dimensions:")
        for (src_img, src_mask), (trg_img, _) in zip(src_train, trg_train):
            print(f"Source image shape: {src_img.shape}, mask shape: {src_mask.shape}")
            print(f"Target image shape: {trg_img.shape}")
            break

        # === 三种方式对比 ===
        # print("\n" + "="*50)
        # print("=== Baseline Training ===")
        # print("="*50)
        # train_model("baseline", src_train, trg_train, src_val, num_classes=8, epochs=20, device=device)

        # print("\n" + "="*50)
        # print("=== Fixed FDA Training ===")
        # print("="*50)
        # train_model("fixed_fda", src_train, trg_train, src_val, num_classes=8, epochs=20, device=device)

        print("\n" + "="*50)
        print("=== Learnable FDA Training ===")
        print("="*50)
        train_model("learnable_fda", src_train, trg_train, src_val, num_classes=8, epochs=20, device=device)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()