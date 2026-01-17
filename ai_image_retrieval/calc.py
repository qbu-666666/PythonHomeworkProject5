import os
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import shutil, sys, subprocess
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

# ==================== 设备设置 ====================
def _detect_gpu_and_cuda():
    has_nvidia_tool = shutil.which('nvidia-smi') is not None
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        return torch.device('cuda')

    # 打印诊断信息，帮助定位为何 PyTorch 未检测到 CUDA
    if has_nvidia_tool:
        print("检测到 nvidia-smi，尝试读取 GPU 信息...")
        try:
            out = subprocess.check_output(['nvidia-smi', '-L'], stderr=subprocess.STDOUT, encoding='utf-8')
            print("nvidia-smi -L 输出:")
            print(out.strip())
        except Exception as e:
            print(f"无法运行 nvidia-smi -L: {e}")
        try:
            smi_info = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, encoding='utf-8')
            print("nvidia-smi 概要（前几行）:")
            print('\n'.join(smi_info.splitlines()[:8]))
        except Exception:
            pass
    else:
        print("未检测到 nvidia-smi（系统可能没有安装 NVIDIA 驱动或非 NVIDIA GPU）。")

    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {cuda_available}")
    try:
        dev_count = torch.cuda.device_count()
    except Exception:
        dev_count = 0
    print(f"torch.cuda.device_count() = {dev_count}")
    print(f"torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")

    if has_nvidia_tool and not cuda_available:
        print("注意：检测到 GPU 但 torch.cuda 未启用。常见原因：PyTorch 不是带 CUDA 的二进制，或驱动/环境不匹配。")
        print("若要安装带 CUDA 的 PyTorch（推荐 CUDA 12.4，兼容你的 CUDA 13.1 驱动）：")
        print("  pip uninstall torch torchvision -y")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        # 可通过环境变量控制是否在此处退出（默认继续在 CPU 上运行）
        if os.environ.get('CONTINUE_ON_CPU', '1') == '0':
            print("环境变量 CONTINUE_ON_CPU=0，脚本退出以便你修复 CUDA 环境。")
            sys.exit(1)
        print("继续在 CPU 上运行（如需退出以修复环境，设置 CONTINUE_ON_CPU=0 后重试）。")

    return torch.device('cpu')

device = _detect_gpu_and_cuda()
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True  # 启用 cudnn benchmark 提升吞吐

# 创建本地图片文件夹
os.makedirs('local_images', exist_ok=True)

# ==================== 图像预处理 ====================
class DinoV2Transform:
    """DINOv2专用的图像预处理"""
    def __init__(self):
        # DINOv2的标准预处理
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, img_or_path):
        if isinstance(img_or_path, str):
            img = Image.open(img_or_path).convert('RGB')
        else:  # bytes
            img = Image.open(BytesIO(img_or_path)).convert('RGB')
        
        # 保持长宽比调整到合适尺寸
        img = self._smart_resize(img)
        return self.transform(img).unsqueeze(0)  # 添加batch维度
    
    def _smart_resize(self, img):
        """保持长宽比的调整"""
        target_size = 518
        width, height = img.size
        
        # 计算缩放比例
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        img = img.resize((new_width, new_height), Image.BICUBIC)
        
        # 填充到518x518
        delta_w = target_size - new_width
        delta_h = target_size - new_height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        return ImageOps.expand(img, padding, fill=(0, 0, 0))

preprocess = DinoV2Transform()

# ==================== 基于 NumPy 的特征提取（新增） ====================
def extract_features_numpy(image_path, resize=(128, 128), bins=32):
    """使用 PIL + numpy 生成颜色直方图特征并 L2 归一化（返回 1D numpy 向量）"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(resize, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    hist_list = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0.0, 1.0), density=True)
        hist_list.append(h.astype(np.float32))
    feat = np.concatenate(hist_list)  # (3*bins,)
    norm = np.linalg.norm(feat)
    if norm > 1e-12:
        feat = feat / norm
    return feat

# ==================== 训练函数（延迟加载 model） ====================
def train_classifier_from_folder(train_dir='local_images/train', epochs=5, batch_size=16, lr=1e-3):
    if not os.path.isdir(train_dir):
        print(f"训练目录未找到: {train_dir}")
        return

    # dataset 使用 preprocess.transform（不加 batch 维）
    train_ds = datasets.ImageFolder(train_dir, transform=preprocess.transform)
    num_classes = len(train_ds.classes)
    if num_classes == 0:
        print("未发现类别，退出")
        return

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 延迟从 hub 加载 DINOv2（仅在训练时）
    print("从 hub 加载 DINOv2 模型（仅训练时）...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.eval()
    model.to(device)
    # 冻结主干
    for p in model.parameters():
        p.requires_grad = False
    model.train()

    # 获取特征维度（前向一次）
    with torch.no_grad():
        sample = torch.zeros((1, 3, 518, 518), device=device)
        out = model(sample)
        if isinstance(out, dict):
            if 'x_norm_clstoken' in out:
                feat = out['x_norm_clstoken']
            elif 'x_prenorm' in out:
                feat = out['x_prenorm'][:, 0]
            else:
                feat = out.mean(dim=1)
        elif out.dim() == 3:
            feat = out[:, 0]
        else:
            feat = out
        feat_dim = feat.shape[1]

    head = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"开始训练分类头: classes={num_classes}, feat_dim={feat_dim}, epochs={epochs}, batch_size={batch_size}")
    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for imgs, labels in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 主干前向（不计算梯度以节省显存/加速）
            with torch.no_grad():
                out = model(imgs)
                if isinstance(out, dict):
                    if 'x_norm_clstoken' in out:
                        feats = out['x_norm_clstoken']
                    elif 'x_prenorm' in out:
                        feats = out['x_prenorm'][:, 0]
                    else:
                        feats = out.mean(dim=1)
                elif out.dim() == 3:
                    feats = out[:, 0]
                else:
                    feats = out
                feats = F.normalize(feats, p=2, dim=1)

            logits = head(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    # 保存分类头
    torch.save({'head_state_dict': head.state_dict(), 'classes': train_ds.classes}, 'classifier.pth')
    print("训练完成，分类头已保存到 classifier.pth")

# ==================== 共同代码 ====================
# ==================== 特征提取函数 ====================
# 保留原来的 torch 提取函数但命名为 extract_features_torch（如需）
@torch.no_grad()
def extract_features_torch(image_tensor, model):
    image_tensor = image_tensor.to(device)
    features = model(image_tensor)
    if isinstance(features, dict):
        if 'x_norm_clstoken' in features:
            features = features['x_norm_clstoken']
        elif 'x_prenorm' in features:
            features = features['x_prenorm'][:, 0]
        else:
            features = features.mean(dim=1)
    elif features.dim() == 3:
        features = features[:, 0]
    features = F.normalize(features, p=2, dim=1)
    return features.cpu().numpy()

# ==================== 主程序 ====================
def main():
    # ===== 提取流程（基于 CPU + NumPy） =====
    print("扫描 local_images 文件夹中的本地图片...")
    img_files = [f for f in os.listdir('local_images') if f.lower().endswith(('.jpg', '.jpeg'))]
    if not img_files:
        print("local_images 中没有找到图片，退出")
        return
    img_files.sort()
    print(f"发现 {len(img_files)} 张本地图片")

    features = []
    valid_paths = []
    valid_captions = []

    for idx, fname in enumerate(img_files):
        local_path = os.path.join('local_images', fname)
        try:
            feat = extract_features_numpy(local_path)
            features.append(feat)
            valid_paths.append(local_path)
            valid_captions.append(fname)  # 以文件名作为 caption
            print(f"[{idx+1}/{len(img_files)}] {fname} 提取完成")
        except Exception as e:
            print(f"[{idx+1}] {fname} 处理失败: {e}")

    if features:
        feats = np.stack(features)
        # 再次确保归一化
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        np.save('db_features.npy', feats)
        np.save('db_paths.npy', np.array(valid_paths))
        np.save('db_captions.npy', np.array(valid_captions))
        print(f"完成，保存 {len(valid_paths)} 张图片的特征到 db_features.npy")
    else:
        print("没有有效图片可提取")

if __name__ == "__main__":
    main()