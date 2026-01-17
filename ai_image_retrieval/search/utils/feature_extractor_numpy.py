import numpy as np
from PIL import Image, ImageOps
import requests
from io import BytesIO
import os
import json

class FeatureExtractorNumpy:
    """纯NumPy的特征提取器"""
    
    def __init__(self, config_path='config.json'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print("初始化NumPy特征提取器...")
        
        # 导入Dinov2Numpy类
        # 注意：这里需要根据你的dinov2_numpy模块的位置调整导入
        try:
            # 尝试从项目根目录导入
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from dinov2_numpy import Dinov2Numpy
            
            # 加载DINOv2模型
            weights_path = self.config.get('weights_path', 'vit-dinov2-base.npz')
            if not os.path.exists(weights_path):
                # 尝试在上级目录查找
                weights_path = os.path.join(os.path.dirname(config_path), '..', 'assignments', 'vit-dinov2-base.npz')
            
            print(f"加载权重文件: {weights_path}")
            self.model = Dinov2Numpy(weights_path)
            
        except ImportError as e:
            print(f"无法导入Dinov2Numpy: {e}")
            # 创建一个简单的模拟模型用于测试
            class MockModel:
                def __init__(self):
                    self.embed_dim = 768
                def __call__(self, x):
                    # 返回随机特征
                    batch_size = x.shape[0]
                    return np.random.randn(batch_size, self.embed_dim)
                def preprocess_image(self, x):
                    # 简单归一化
                    if x.max() > 1.0:
                        x = x / 255.0
                    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                    return (x - mean) / std
            
            self.model = MockModel()
        
        print("✅ NumPy特征提取器初始化完成")
    
    def preprocess_image(self, image_input):
        """
        预处理图像
        Args:
            image_input: 可以是文件路径、URL或PIL Image
        Returns:
            预处理后的numpy数组 (1, 3, H, W)
        """
        # 加载图像
        if isinstance(image_input, str):
            if image_input.startswith('http'):
                # 下载图像
                try:
                    proxies = self.config.get('proxy', {})
                    response = requests.get(image_input, timeout=10, proxies=proxies)
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    print(f"下载图片失败 {image_input}: {e}")
                    raise
            else:
                # 本地文件
                img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            raise ValueError("不支持的输入类型")
        
        # 调整大小并填充到14的倍数
        img = self._resize_and_pad(img)
        
        # 转换为numpy数组
        img_array = np.array(img).astype(np.float32)
        
        # 转换为 (C, H, W) 格式
        img_array = img_array.transpose(2, 0, 1)  # (3, H, W)
        
        # 添加batch维度
        img_array = img_array[np.newaxis, ...]  # (1, 3, H, W)
        
        # 应用模型预处理（归一化）
        img_array = self.model.preprocess_image(img_array)
        
        return img_array
    
    def _resize_and_pad(self, img, target_divisible=14):
        """
        调整图像大小并填充到target_divisible的倍数
        """
        # 计算目标尺寸（保持长宽比）
        width, height = img.size
        target_size = 518  # DINOv2的标准输入
        
        # 计算缩放比例
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 调整大小
        img = img.resize((new_width, new_height), Image.BICUBIC)
        
        # 填充到target_divisible的倍数
        pad_w = (target_divisible - new_width % target_divisible) % target_divisible
        pad_h = (target_divisible - new_height % target_divisible) % target_divisible
        
        padding = (0, 0, pad_w, pad_h)  # 左, 上, 右, 下
        img = ImageOps.expand(img, padding, fill=0)
        
        return img
    
    def extract_features(self, image_input):
        """
        提取图像特征
        Args:
            image_input: 图像路径、URL或PIL Image
        Returns:
            特征向量 (embed_dim,)
        """
        try:
            # 预处理
            img_array = self.preprocess_image(image_input)
            
            # 提取特征
            features = self.model(img_array)  # (1, embed_dim)
            
            # 移除batch维度
            features = features[0]
            
            # L2归一化
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None