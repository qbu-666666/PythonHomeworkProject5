from django.db import models
from django.core.validators import FileExtensionValidator

class UploadedImage(models.Model):
    """用户上传的图片"""
    image = models.ImageField(
        upload_to='uploads/%Y/%m/%d/',
        validators=[FileExtensionValidator(['jpg', 'jpeg', 'png', 'bmp', 'gif'])]
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Image {self.id}"

class SearchResult(models.Model):
    """搜索结果"""
    query_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE, related_name='results')
    result_index = models.IntegerField()
    result_image_path = models.CharField(max_length=500)
    result_caption = models.TextField(blank=True)
    similarity_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-similarity_score']
    
    def __str__(self):
        return f"Result {self.result_index} for Image {self.query_image.id}"