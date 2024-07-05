from django.urls import path
from .views import image_upload_view

urlpatterns = [
    path('upload/', image_upload_view, name='image_upload'),
]
