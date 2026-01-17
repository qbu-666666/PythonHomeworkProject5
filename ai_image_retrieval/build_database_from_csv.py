import os
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps

from dinov2_numpy import Dinov2Numpy

PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}

# 本地图片文件夹
LOCAL_IMAGES_DIR = 'local_images'

# 确保文件夹存在
os.makedirs(LOCAL_IMAGES_DIR, exist_ok=True)

# 优化预处理函数：保持长宽比 + pad 到 14 整除
def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    
    ps = 14
    short_side = min(img.size)
    # 选择最接近原比例的 14 倍数短边（避免极端尺寸）
    multiples = [ps * i for i in range(12, 40)]  # 168~560
    best_multiple = min(multiples, key=lambda m: abs(m - short_side))
    ratio = best_multiple / short_side
    new_size = (round(img.width * ratio), round(img.height * ratio))
    img = img.resize(new_size)
    
    # Pad 黑边到整除 14
    H, W = img.height, img.width
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    if pad_h or pad_w:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill='black')
    
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return arr

# 加载模型
weights = np.load('vit-dinov2-base.npz', allow_pickle=True)
model = Dinov2Numpy(weights)

# 从 CSV 读取前 500 行
N = 500
df = pd.read_csv('data.csv', nrows=N)
print(f"从 CSV 读取 {len(df)} 行，开始下载并处理图片...")

session = requests.Session()
session.proxies = PROXIES
session.headers.update({'User-Agent': 'Mozilla/5.0'})

features = []
valid_paths = []
captions = []

for idx, row in df.iterrows():
    url = row.get('image_url', None)
    caption = row.get('caption', "No caption") if pd.notna(row.get('caption')) else "No caption"
    if not isinstance(url, str) or not url:
        continue
    
    local_path = os.path.join(LOCAL_IMAGES_DIR, f'{idx:06d}.jpg')
    print(f"[{idx+1}/{len(df)}] 下载并处理 {url[:60]}...")
    
    try:
        # 下载图片
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            print(f"  文件太小，跳过")
            continue
        with open(local_path, 'wb') as f:
            f.write(resp.content)
        
        # 提取特征
        pixel = preprocess_image(local_path)
        feat = model(pixel)[0]
        
        features.append(feat)
        valid_paths.append(local_path)
        captions.append(caption)
        
    except requests.RequestException as e:
        print(f"  下载失败: {e}")
    except Exception as e:
        print(f"  处理失败: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

# 保存有效数据
if features:
    features = np.stack(features)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)  # L2 归一化
    
    np.save('db_features.npy', features)
    np.save('db_paths.npy', np.array(valid_paths))
    np.save('db_captions.npy', np.array(captions))
    
    print(f"\n完成！有效图片 {len(valid_paths)} 张")
    print("文件已保存：db_features.npy, db_paths.npy, db_captions.npy")
    print("图片路径在 db_paths.npy 中（相对或绝对路径）")
else:
    print("没有有效图片！请检查 local_images 文件夹是否有正常图片")

# 使用说明：
# 1. 把所有图片放进 local_images/ 文件夹（支持 jpg/jpeg/png/bmp）
# 2. 运行 python build_database_from_csv.py（或重命名为 build_local_database.py）
# 3. 时间：每张 ≈10~30秒（纯 CPU NumPy）
# 4. 完成后直接跑 python manage.py runserver 测试 Django 检索
# 5. caption 在结果页显示为文件名（可手动改 captions 列表为自定义描述）