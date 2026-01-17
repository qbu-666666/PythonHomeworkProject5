from django import forms
from .models import UploadedImage

class ImageUploadForm(forms.ModelForm):
    url = forms.URLField(
        required=False,
        label='或输入图片URL',
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://example.com/image.jpg'
        })
    )
    
    class Meta:
        model = UploadedImage
        fields = ['image']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].required = False
        self.fields['image'].widget.attrs.update({
            'class': 'form-control form-control-lg',
            'accept': 'image/*'
        })
    
    def clean(self):
        cleaned_data = super().clean()
        image = cleaned_data.get('image')
        url = cleaned_data.get('url')
        
        if not image and not url:
            raise forms.ValidationError('请选择图片或输入图片URL')
        
        return cleaned_data