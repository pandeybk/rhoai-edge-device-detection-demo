from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select an image to upload')
