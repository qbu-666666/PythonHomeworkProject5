import numpy as np
from PIL import Image
from dinov2_numpy import Dinov2Numpy  # 确认文件名正确

def preprocess_image(path, size=224):
    img = Image.open(path).convert('RGB').resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return arr

# 正确加载方式（移除 ['weights'].item()）
weights = np.load('vit-dinov2-base.npz', allow_pickle=True)  # 直接得到 dict-like NpzFile

# 如果想确认键名，可以先打印（调试用，正式运行可删）
print("可用键示例:", list(weights.keys())[:10])  # 应该看到 'embeddings.cls_token' 等

model = Dinov2Numpy(weights)

cat_feat = model(preprocess_image('demo_data/cat.jpg'))  # (1, 768)
dog_feat = model(preprocess_image('demo_data/dog.jpg'))  # (1, 768)

# L2 归一化后计算余弦相似度
cat_feat = cat_feat / np.linalg.norm(cat_feat, axis=1, keepdims=True)
dog_feat = dog_feat / np.linalg.norm(dog_feat, axis=1, keepdims=True)
sim = np.dot(cat_feat[0], dog_feat[0])
print("cat-dog cosine similarity:", sim)  # 预期 0.65~0.80（DINOv2 很强）
print("cat-cat similarity:", np.dot(cat_feat[0], cat_feat[0]))  # 应 ≈1.0

ref = np.load('demo_data/cat_dog_feature.npy')  # 路径根据你的位置调整
ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)

print("与官方 cat 特征相似度:", np.dot(cat_feat[0], ref[0]))
print("与官方 dog 特征相似度:", np.dot(dog_feat[0], ref[1]))