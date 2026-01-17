import numpy as np
import os

class SimpleSimilaritySearch:
    """简化版相似度搜索"""
    
    def __init__(self):
        print("初始化相似度搜索器...")
        
        # 加载数据库
        self.load_database()
        print(f"✅ 加载完成，数据库大小: {len(self.db_features)}")
    
    def load_database(self):
        """加载预计算的特征数据库"""
        try:
            # 尝试加载数据文件
            if os.path.exists('data/db_features.npy'):
                self.db_features = np.load('data/db_features.npy')
                self.db_paths = np.load('data/db_paths.npy', allow_pickle=True)
                self.db_captions = np.load('data/db_captions.npy', allow_pickle=True)
            elif os.path.exists('db_features.npy'):
                self.db_features = np.load('db_features.npy')
                self.db_paths = np.load('db_paths.npy', allow_pickle=True)
                self.db_captions = np.load('db_captions.npy', allow_pickle=True)
            else:
                print("警告: 未找到数据库文件，使用模拟数据")
                self.create_mock_database()
        except Exception as e:
            print(f"数据库加载失败: {e}")
            self.create_mock_database()
    
    def create_mock_database(self):
        """创建模拟数据库（64张图片）"""
        print("创建模拟数据库...")
        
        # 64张图片，每张768维特征
        self.db_features = np.random.randn(64, 768)
        
        # 归一化
        self.db_features = self.db_features / np.linalg.norm(self.db_features, axis=1, keepdims=True)
        
        # 模拟图片路径和描述
        self.db_paths = np.array([f"media/db_images/{i:03d}.jpg" for i in range(64)])
        self.db_captions = np.array([f"图片 {i+1}" for i in range(64)])
        
        # 保存模拟数据（可选）
        np.save('db_features.npy', self.db_features)
        np.save('db_paths.npy', self.db_paths)
        np.save('db_captions.npy', self.db_captions)
    
    def search(self, query_features, top_k=10):
        """搜索相似图片"""
        if len(self.db_features) == 0:
            return []
        
        # 归一化查询特征
        query_norm = np.linalg.norm(query_features)
        if query_norm > 0:
            query_features = query_features / query_norm
        
        # 计算余弦相似度
        similarities = np.dot(self.db_features, query_features)
        
        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 构建结果
        results = []
        for i, idx in enumerate(top_indices):
            sim = similarities[idx]
            
            # 计算置信度 (70%到100%)
            confidence = max(70, min(100, 70 + (sim * 30)))
            
            results.append({
                'index': int(idx),
                'img_path': self.db_paths[idx],
                'img_url': f"/media/db_images/{idx:03d}.jpg",
                'caption': str(self.db_captions[idx]) if idx < len(self.db_captions) else "图片",
                'similarity': float(sim),
                'confidence': round(confidence, 1)
            })
        
        return results