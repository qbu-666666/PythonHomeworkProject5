import os
import numpy as np
from django.shortcuts import render, redirect
from django.http import FileResponse, Http404, HttpResponse
from django.conf import settings
from .forms import ImageUploadForm
from .models import UploadedImage, SearchResult

# ==================== 模拟搜索系统 ====================
class SimpleSearchSystem:
    """简化搜索系统，使用 local_images 中的图片"""
    
    def __init__(self):
        print("初始化搜索系统...")
        
        # 检查 local_images 文件夹
        self.local_images_path = self._find_local_images()
        
        # 获取图片数量
        self.image_count = self._count_images()
        print(f"✅ 找到 {self.image_count} 张图片")
        
        # 模拟特征数据库（64张图片）
        self.db_features = np.random.randn(64, 768).astype(np.float32)
        self.db_features = self.db_features / np.linalg.norm(self.db_features, axis=1, keepdims=True)
        
        # 图片描述
        self.db_captions = [f"图片 {i+1}" for i in range(64)]
        
        print("✅ 搜索系统初始化完成")
    
    def _find_local_images(self):
        """查找 local_images 文件夹"""
        possible_paths = [
            'local_images',
            '../assignments/local_images',
            os.path.join(settings.BASE_DIR, 'local_images'),
            os.path.join(settings.BASE_DIR, '..', 'assignments', 'local_images'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 找到图片文件夹: {path}")
                return path
        
        print("⚠️  未找到 local_images 文件夹")
        return None
    
    def _count_images(self):
        """计算图片数量"""
        if not self.local_images_path or not os.path.exists(self.local_images_path):
            return 0
        
        # 统计jpg文件
        count = 0
        for filename in os.listdir(self.local_images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
        
        return count
    
    def extract_features(self, image_path_or_url):
        """模拟特征提取"""
        # 在实际项目中，这里会使用DINOv2提取真实特征
        features = np.random.randn(768)
        features = features / np.linalg.norm(features)
        return features
    
    def search(self, query_features, top_k=10):
        """搜索相似图片，过滤掉不存在的文件并确保最多返回 top_k 个有效结果"""
        similarities = np.dot(self.db_features, query_features)
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            sim = similarities[idx]
            # 计算置信度 (70%到100%)
            confidence = max(70, min(100, 70 + (sim * 30)))

            # 生成图片文件名和检查文件是否存在
            img_filename = f"{idx:06d}.jpg"
            valid_file = False

            if self.local_images_path and os.path.exists(self.local_images_path):
                candidate = os.path.join(self.local_images_path, img_filename)
                if os.path.exists(candidate):
                    valid_file = True
                else:
                    # 尝试其他扩展名
                    name_without_ext = os.path.splitext(img_filename)[0]
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        test_path = os.path.join(self.local_images_path, name_without_ext + ext)
                        if os.path.exists(test_path):
                            img_filename = os.path.basename(test_path)
                            valid_file = True
                            break

            if not valid_file:
                # 跳过不存在的图片
                continue

            img_url = f"/local_images/{img_filename}"

            results.append({
                'index': int(idx),
                'img_url': img_url,
                'caption': self.db_captions[idx % len(self.db_captions)],
                'sim': round(float(sim), 3),
                'confidence': round(confidence, 1)
            })

        return results

# ==================== 视图函数 ====================

# 全局搜索系统实例
_search_system = None

def get_search_system():
    """获取搜索系统实例"""
    global _search_system
    if _search_system is None:
        _search_system = SimpleSearchSystem()
    return _search_system

def home(request):
    """首页和搜索处理"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        
        # 检查输入
        has_file = 'image' in request.FILES and request.FILES['image']
        has_url = 'url' in request.POST and request.POST['url'].strip()
        
        if not has_file and not has_url:
            return render(request, 'search/home.html', {
                'form': form,
                'error': '请选择图片或输入图片URL'
            })
        
        try:
            system = get_search_system()
            
            # 提取特征
            if has_file:
                # 处理上传的图片
                uploaded_image = form.save()
                features = system.extract_features(uploaded_image.image.path)
                query_url = uploaded_image.image.url
                query_name = os.path.basename(uploaded_image.image.name)
                uploaded_image_obj = uploaded_image
            else:
                # 处理URL
                url = request.POST['url'].strip()
                features = system.extract_features(url)
                query_url = url
                query_name = 'URL图片'
                uploaded_image_obj = None
            
            # 搜索相似图片
            results = system.search(features, top_k=10)
            
            # 保存搜索结果（如果有上传图片）
            if uploaded_image_obj:
                for i, result in enumerate(results):
                    SearchResult.objects.create(
                        query_image=uploaded_image_obj,
                        result_index=i,
                        result_image_path=result['img_url'],
                        result_caption=result['caption'],
                        similarity_score=result['sim']
                    )
            
            # 渲染结果页面
            return render(request, 'search/results.html', {
                'query_url': query_url,
                'query_name': query_name,
                'results': results
            })
            
        except Exception as e:
            print(f"搜索出错: {e}")
            return render(request, 'search/home.html', {
                'form': form,
                'error': f'搜索失败: {str(e)}'
            })
    
    else:
        # GET请求：显示首页
        form = ImageUploadForm()
    
    return render(request, 'search/home.html', {'form': form})

def serve_local_image(request, filename):
    """提供 local_images 文件夹中的图片"""
    # 获取搜索系统（用于找到图片路径）
    system = get_search_system()
    
    if not system.local_images_path:
        raise Http404("图片文件夹不存在")
    
    # 构建完整路径
    image_path = os.path.join(system.local_images_path, filename)
    
    # 如果文件不存在，尝试其他格式
    if not os.path.exists(image_path):
        # 尝试不同的扩展名
        name_without_ext = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(system.local_images_path, name_without_ext + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
        else:
            # 如果还是找不到，返回一个占位SVG（避免页面出现破图）
            svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300" '
                   'viewBox="0 0 400 300"><rect width="100%" height="100%" fill="#f0f0f0"/>'
                   '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
                   'fill="#888" font-size="20">图片不存在</text></svg>')
            return HttpResponse(svg, content_type='image/svg+xml')
    
    # 返回图片文件
    return FileResponse(open(image_path, 'rb'), content_type='image/jpeg')

def search_by_url(request):
    """通过URL搜索（可选，用于保持URL兼容性）"""
    if request.method == 'POST':
        # 重定向到home处理
        request.method = 'POST'
        return home(request)
    return redirect('home')