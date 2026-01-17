from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('local_images/<str:filename>', views.serve_local_image, name='serve_local_image'),
    # 可选：保持search_by_url路径的兼容性
    path('search/url/', views.search_by_url, name='search_by_url'),
]