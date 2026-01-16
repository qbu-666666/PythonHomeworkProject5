from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import numpy as np
from PIL import Image
from io import BytesIO
from dinov2_numpy import Dinov2Numpy

# 全局加载模型和数据库（只加载一次）
weights = np.load('vit-dinov2-base.npz', allow_pickle=True)
model = Dinov2Numpy(weights)

db_features = np.load('db_features.npy')
db_paths = np.load('db_paths.npy')
db_captions = np.load('db_captions.npy', allow_pickle=True)

def preprocess_image(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert('RGB')
    else:
        img = img_path_or_pil.convert('RGB')
    
    ratio = 224 / min(img.size)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size)
    
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr

def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        upload_path = fs.path(filename)  # 完整本地路径

        # 提取查询特征
        query_feat = model(preprocess_image(upload_path))[0]
        query_feat = query_feat / np.linalg.norm(query_feat)

        # 计算相似度
        sims = np.dot(db_features, query_feat)
        topk = np.argsort(sims)[-10:][::-1]  # Top 10

        results = []
        for idx in topk:
            img_name = os.path.basename(db_paths[idx])  # 只取文件名，如 000001.jpg
            results.append({
                'img_url': f'/static/{img_name}',  # 静态服侍
                'sim': round(float(sims[idx]), 4),
                'caption': db_captions[idx]
            })

        # 查询图 URL（media 服侍）
        query_url = fs.url(filename)

        return render(request, 'results.html', {
            'results': results,
            'query_url': query_url,
            'query_name': file.name
        })

    return render(request, 'home.html')