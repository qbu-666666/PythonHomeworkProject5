import os
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from dinov2_numpy import Dinov2Numpy

# 创建文件夹
os.makedirs('local_images', exist_ok=True)  # 下载图片存这里

# 预处理函数（优化版：保持长宽比，支持矩形图）
def preprocess_image(img_path_or_bytes):
    if isinstance(img_path_or_bytes, bytes):  # 从 bytes 下载
        img = Image.open(BytesIO(img_path_or_bytes)).convert('RGB')
    else:
        img = Image.open(img_path_or_bytes).convert('RGB')
    
    ratio = 224 / min(img.size)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size)
    
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr

# 加载模型
weights = np.load('vit-dinov2-base.npz', allow_pickle=True)
model = Dinov2Numpy(weights)

# 读取 csv（只取前 N 行，避免太大）
N = 500  # 你可以改成 100~1000，500 张下载+计算 ≈30~60 分钟
df = pd.read_csv('data.csv', nrows=N)

features = []
valid_paths = []
valid_captions = []

print(f"开始处理 {len(df)} 张图片...")
for idx, row in df.iterrows():
    url = row['image_url']
    caption = row['caption']
    
    print(f"[{idx+1}/{len(df)}] 下载 {url[:50]}...")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200 or len(response.content) < 1000:  # 过滤坏图
            print("  跳过（无效响应）")
            continue
        
        # 保存本地图片（文件名用索引）
        local_path = f'local_images/{idx:06d}.jpg'
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        # 提取特征
        pixel = preprocess_image(response.content)  # 直接从 bytes，节省磁盘
        feat = model(pixel)[0]
        
        features.append(feat)
        valid_paths.append(local_path)
        valid_captions.append(caption)
        
    except Exception as e:
        print(f"  跳过（错误: {e})")

# 转 numpy 并 L2 归一化
features = np.stack(features)
features = features / np.linalg.norm(features, axis=1, keepdims=True)

# 保存
np.save('db_features.npy', features)
np.save('db_paths.npy', np.array(valid_paths))
np.save('db_captions.npy', np.array(valid_captions))

print(f"完成！有效图片 {len(valid_paths)} 张（可能少于 {N}，因链接失效）")
print("特征和 caption 已保存，可用于 Django")