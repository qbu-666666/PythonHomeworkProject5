import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from utils.feature_extractor_numpy import FeatureExtractorNumpy

def build_database(csv_path='assignments/data.csv', nrows=64, output_dir='data'):
    """构建图像数据库"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('media/db_images', exist_ok=True)
    
    # 读取CSV
    print(f"读取CSV文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path, nrows=nrows)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return
    
    # 初始化特征提取器
    extractor = FeatureExtractorNumpy('config.json')
    
    features_list = []
    paths_list = []
    captions_list = []
    
    print(f"开始处理 {len(df)} 张图片...")
    
    for idx, row in df.iterrows():
        url = row['image_url']
        caption = row.get('caption', '') if pd.notna(row.get('caption', '')) else ''
        
        try:
            print(f"处理第 {idx+1} 张图片: {url[:50]}...")
            
            # 下载图片
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"  下载失败: 状态码 {response.status_code}")
                continue
            
            # 保存图片
            img = Image.open(BytesIO(response.content)).convert('RGB')
            save_path = f"media/db_images/{idx:06d}.jpg"
            img.save(save_path)
            
            # 提取特征
            features = extractor.extract_features(img)
            if features is not None and len(features) > 0:
                features_list.append(features)
                paths_list.append(save_path)
                captions_list.append(caption)
                print(f"  处理成功")
            else:
                print(f"  特征提取失败")
                
        except Exception as e:
            print(f"处理失败 {idx}: {e}")
            continue
    
    # 保存数据库
    if features_list:
        features_array = np.stack(features_list)
        
        # 保存到文件
        np.save(os.path.join(output_dir, 'db_features.npy'), features_array)
        np.save(os.path.join(output_dir, 'db_paths.npy'), np.array(paths_list))
        np.save(os.path.join(output_dir, 'db_captions.npy'), np.array(captions_list))
        
        print(f"\n✅ 数据库构建完成！")
        print(f"  处理成功: {len(features_list)} 张图片")
        print(f"  特征维度: {features_array.shape}")
        print(f"  数据库文件保存在: {output_dir}/")
    else:
        print("\n❌ 没有成功处理任何图片")

if __name__ == "__main__":
    # 构建64张图片的数据库
    build_database('data.csv', nrows=64)